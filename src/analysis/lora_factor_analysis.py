import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import requests
import tweepy
import praw
from newspaper import Article
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from typing import Dict, List, Tuple, Optional, Union
import warnings
import json
from datetime import datetime, timedelta
import os
from datasets import Dataset
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class LoRAFactorAnalysis:
    def __init__(self, config, model_name: str = "microsoft/DialoGPT-medium", use_4bit: bool = True):
        self.config = config
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        if config.TWITTER_BEARER_TOKEN:
            self.twitter_client = tweepy.Client(bearer_token=config.TWITTER_BEARER_TOKEN)
        else:
            self.twitter_client = None
            
        if all([config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET, config.REDDIT_USER_AGENT]):
            self.reddit_client = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT
            )
        else:
            self.reddit_client = None
        
        self.model = None
        self.tokenizer = None
        self._initialize_lora_model()
    
    def _initialize_lora_model(self):
        try:
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                bnb_config = None
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
        except Exception as e:
            print(f"Error initializing LoRA model: {e}")
            print("Falling back to basic sentiment analysis")
            self.model = None
            self.tokenizer = None
    
    def create_financial_dataset(self, texts: List[str], labels: List[str]) -> Dataset:
        formatted_texts = []
        
        for text, label in zip(texts, labels):
            formatted_text = f"""### Instruction:
Analyze the following financial text and provide sentiment analysis.

### Input:
{text}

### Response:
{label}

### End"""
            formatted_texts.append(formatted_text)
        
        return Dataset.from_dict({"text": formatted_texts})
    
    def fine_tune_lora(self, training_texts: List[str], training_labels: List[str], 
                      epochs: int = 3, batch_size: int = 4):
        if self.model is None:
            print("Model not initialized, skipping fine-tuning")
            return
        
        try:
            dataset = self.create_financial_dataset(training_texts, training_labels)
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            training_args = TrainingArguments(
                output_dir="./lora_financial_model",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="no",
                save_strategy="epoch",
                load_best_model_at_end=False,
                report_to=None
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator
            )
            
            trainer.train()
            
            print("LoRA fine-tuning completed successfully")
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
    
    def analyze_financial_sentiment_lora(self, text: str) -> Dict[str, float]:
        if self.model is None or self.tokenizer is None:
            return self._fallback_sentiment_analysis(text)
        
        try:
            prompt = f"""### Instruction:
Analyze the following financial text and provide sentiment analysis in JSON format with scores from -1 to 1.

### Input:
{text}

### Response:
"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    
                    required_fields = ['sentiment', 'confidence', 'relevance', 'risk_level']
                    for field in required_fields:
                        if field not in result:
                            result[field] = 0.0
                    
                    return result
                else:
                    return self._fallback_sentiment_analysis(text)
                    
            except json.JSONDecodeError:
                return self._fallback_sentiment_analysis(text)
                
        except Exception as e:
            print(f"Error in LoRA sentiment analysis: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, float]:
        if not text or len(text.strip()) < 10:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'relevance': 0.0,
                'risk_level': 0.0
            }
        
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        financial_keywords = [
            'bond', 'yield', 'interest', 'rate', 'treasury', 'credit', 'spread',
            'inflation', 'fed', 'federal reserve', 'monetary', 'policy', 'market',
            'investment', 'portfolio', 'risk', 'volatility', 'return', 'price'
        ]
        
        relevance = sum(1 for keyword in financial_keywords if keyword.lower() in text.lower()) / len(financial_keywords)
        
        risk_keywords = ['crash', 'fall', 'decline', 'drop', 'volatile', 'uncertainty', 'risk']
        risk_level = sum(1 for keyword in risk_keywords if keyword.lower() in text.lower()) / len(risk_keywords)
        
        return {
            'sentiment': sentiment['compound'],
            'confidence': abs(sentiment['compound']),
            'relevance': min(relevance, 1.0),
            'risk_level': min(risk_level, 1.0)
        }
    
    def get_market_factors(self, start_date: str, end_date: str) -> pd.DataFrame:
        factor_tickers = {
            'equity_market': 'SPY',
            'credit_spread': 'LQD',
            'high_yield_spread': 'HYG',
            'treasury_10y': '^TNX',
            'treasury_2y': '^UST2YR',
            'dollar_index': 'UUP',
            'gold': 'GLD',
            'oil': 'USO'
        }
        
        factors = {}
        for factor_name, ticker in factor_tickers.items():
            try:
                import yfinance as yf
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    factors[factor_name] = data['Adj Close'].pct_change()
            except:
                continue
        
        return pd.DataFrame(factors).dropna()
    
    def calculate_rolling_returns(self, prices: pd.DataFrame, windows: List[int] = [20, 60, 120]) -> pd.DataFrame:
        rolling_features = {}
        
        for ticker in prices.columns:
            for window in windows:
                rolling_features[f'{ticker}_rolling_{window}d'] = prices[ticker].pct_change(window)
                rolling_features[f'{ticker}_momentum_{window}d'] = prices[ticker] / prices[ticker].shift(window) - 1
                rolling_features[f'{ticker}_volatility_{window}d'] = prices[ticker].pct_change().rolling(window).std()
        
        return pd.DataFrame(rolling_features).dropna()
    
    def extract_sentiment_from_text(self, text: str) -> Dict[str, float]:
        return self.analyze_financial_sentiment_lora(text)
    
    def get_news_sentiment(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        search_terms = [ticker, f"{ticker} bond", f"{ticker} ETF"]
        all_articles = []
        
        for term in search_terms:
            try:
                url = f"https://newsapi.org/v2/everything?q={term}&from={days_back}days&sortBy=publishedAt&apiKey=YOUR_NEWS_API_KEY"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    all_articles.extend(articles)
            except:
                continue
        
        sentiments = []
        for article in all_articles[:20]:
            try:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_financial_sentiment_lora(text)
                sentiment['date'] = article.get('publishedAt', '')
                sentiment['source'] = article.get('source', {}).get('name', '')
                sentiments.append(sentiment)
            except:
                continue
        
        return pd.DataFrame(sentiments)
    

    
    def get_twitter_sentiment(self, ticker: str, count: int = 100) -> pd.DataFrame:
        if not self.twitter_client:
            print(f"No Twitter client configured. Set TWITTER_BEARER_TOKEN in environment variables.")
            return pd.DataFrame()
        
        try:
            query = f"{ticker} -is:retweet lang:en"
            print(f"Searching Twitter for: {query}")
            tweets = self.twitter_client.search_recent_tweets(query=query, max_results=count)
            
            sentiments = []
            if tweets.data:
                print(f"Found {len(tweets.data)} tweets for {ticker}")
                for tweet in tweets.data:
                    sentiment = self.analyze_financial_sentiment_lora(tweet.text)
                    sentiment['date'] = tweet.created_at
                    sentiment['tweet_id'] = tweet.id
                    sentiments.append(sentiment)
            else:
                print(f"No tweets found for {ticker}")
            
            return pd.DataFrame(sentiments)
        except Exception as e:
            print(f"Error fetching Twitter data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_reddit_sentiment(self, ticker: str, subreddits: List[str] = ['investing', 'stocks', 'wallstreetbets']) -> pd.DataFrame:
        if not self.reddit_client:
            print(f"No Reddit client configured. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in environment variables.")
            return pd.DataFrame()
        
        sentiments = []
        print(f"Searching Reddit for {ticker} in subreddits: {subreddits}")
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                posts = subreddit.search(ticker, limit=50, time_filter='week')
                post_list = list(posts)
                print(f"Found {len(post_list)} posts in r/{subreddit_name} for {ticker}")
                
                for post in post_list:
                    sentiment = self.analyze_financial_sentiment_lora(post.title + " " + post.selftext)
                    sentiment['date'] = post.created_utc
                    sentiment['subreddit'] = subreddit_name
                    sentiment['score'] = post.score
                    sentiments.append(sentiment)
            except Exception as e:
                print(f"Error fetching Reddit data from r/{subreddit_name} for {ticker}: {e}")
                continue
        
        print(f"Total Reddit posts analyzed for {ticker}: {len(sentiments)}")
        return pd.DataFrame(sentiments)
    
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        if self.model is None:
            return self._fallback_triplet_extraction(text)
        
        try:
            prompt = f"""### Instruction:
Extract subject-verb-object triplets from the following financial text.

### Input:
{text}

### Response:
"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            triplets = []
            lines = response.split('\n')
            for line in lines:
                if '(' in line and ')' in line:
                    try:
                        triplet_str = line.strip()
                        if triplet_str.startswith('(') and triplet_str.endswith(')'):
                            triplet_str = triplet_str[1:-1]
                            parts = triplet_str.split(',')
                            if len(parts) == 3:
                                subject = parts[0].strip().strip('"')
                                verb = parts[1].strip().strip('"')
                                obj = parts[2].strip().strip('"')
                                triplets.append((subject, verb, obj))
                    except:
                        continue
            
            return triplets if triplets else self._fallback_triplet_extraction(text)
            
        except Exception as e:
            print(f"Error in LoRA triplet extraction: {e}")
            return self._fallback_triplet_extraction(text)
    
    def _fallback_triplet_extraction(self, text: str) -> List[Tuple[str, str, str]]:
        doc = nlp(text)
        triplets = []
        
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    subject = None
                    object_ = None
                    
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child.text
                        elif child.dep_ in ["dobj", "pobj"]:
                            object_ = child.text
                    
                    if subject and object_:
                        triplets.append((subject, token.text, object_))
        
        return triplets
    
    def aggregate_sentiment_features(self, ticker: str) -> Dict[str, float]:
        news_sentiment = self.get_news_sentiment(ticker)
        twitter_sentiment = self.get_twitter_sentiment(ticker)
        reddit_sentiment = self.get_reddit_sentiment(ticker)
        
        all_sentiments = []
        if not news_sentiment.empty:
            all_sentiments.extend(news_sentiment['sentiment'].tolist())
        if not twitter_sentiment.empty:
            all_sentiments.extend(twitter_sentiment['sentiment'].tolist())
        if not reddit_sentiment.empty:
            all_sentiments.extend(reddit_sentiment['sentiment'].tolist())
        
        if not all_sentiments:
            return {
                'avg_sentiment': 0,
                'sentiment_volatility': 0,
                'sentiment_momentum': 0,
                'avg_confidence': 0,
                'avg_relevance': 0,
                'avg_risk_level': 0,
                'news_count': 0,
                'social_count': 0
            }
        
        return {
            'avg_sentiment': np.mean(all_sentiments),
            'sentiment_volatility': np.std(all_sentiments),
            'sentiment_momentum': np.mean(all_sentiments[-10:]) if len(all_sentiments) >= 10 else np.mean(all_sentiments),
            'avg_confidence': np.mean([s.get('confidence', 0) for s in all_sentiments]),
            'avg_relevance': np.mean([s.get('relevance', 0) for s in all_sentiments]),
            'avg_risk_level': np.mean([s.get('risk_level', 0) for s in all_sentiments]),
            'news_count': len(news_sentiment),
            'social_count': len(twitter_sentiment) + len(reddit_sentiment)
        }
    
    def save_lora_model(self, path: str = "./lora_financial_model"):
        if self.model is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"LoRA model saved to {path}")
    
    def load_lora_model(self, path: str = "./lora_financial_model"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            print(f"LoRA model loaded from {path}")
        except Exception as e:
            print(f"Error loading LoRA model: {e}")
            self._initialize_lora_model() 
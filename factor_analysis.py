import pandas as pd
import numpy as np
import yfinance as yf
import requests
import tweepy
import praw
from newspaper import Article
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline
import litellm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class FactorAnalysis:
    def __init__(self, config):
        self.config = config
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
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
        if not text or len(text.strip()) < 10:
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            'compound': sentiment['compound'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu'],
            'entities': entities
        }
    
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
                sentiment = self.extract_sentiment_from_text(text)
                sentiment['date'] = article.get('publishedAt', '')
                sentiment['source'] = article.get('source', {}).get('name', '')
                sentiments.append(sentiment)
            except:
                continue
        
        return pd.DataFrame(sentiments)
    
    def get_twitter_sentiment(self, ticker: str, count: int = 100) -> pd.DataFrame:
        if not self.twitter_client:
            return pd.DataFrame()
        
        try:
            query = f"{ticker} -is:retweet lang:en"
            tweets = self.twitter_client.search_recent_tweets(query=query, max_results=count)
            
            sentiments = []
            for tweet in tweets.data or []:
                sentiment = self.extract_sentiment_from_text(tweet.text)
                sentiment['date'] = tweet.created_at
                sentiment['tweet_id'] = tweet.id
                sentiments.append(sentiment)
            
            return pd.DataFrame(sentiments)
        except:
            return pd.DataFrame()
    
    def get_reddit_sentiment(self, ticker: str, subreddits: List[str] = ['investing', 'stocks', 'wallstreetbets']) -> pd.DataFrame:
        if not self.reddit_client:
            return pd.DataFrame()
        
        sentiments = []
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                posts = subreddit.search(ticker, limit=50, time_filter='week')
                
                for post in posts:
                    sentiment = self.extract_sentiment_from_text(post.title + " " + post.selftext)
                    sentiment['date'] = post.created_utc
                    sentiment['subreddit'] = subreddit_name
                    sentiment['score'] = post.score
                    sentiments.append(sentiment)
            except:
                continue
        
        return pd.DataFrame(sentiments)
    
    def analyze_central_bank_reports(self, keywords: List[str] = ['federal reserve', 'ECB', 'BOJ', 'BOE']) -> pd.DataFrame:
        sentiments = []
        
        for keyword in keywords:
            try:
                url = f"https://newsapi.org/v2/everything?q={keyword}&domains=reuters.com,bloomberg.com&sortBy=publishedAt&apiKey=YOUR_NEWS_API_KEY"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    
                    for article in articles[:10]:
                        text = f"{article.get('title', '')} {article.get('description', '')}"
                        sentiment = self.extract_sentiment_from_text(text)
                        sentiment['date'] = article.get('publishedAt', '')
                        sentiment['keyword'] = keyword
                        sentiments.append(sentiment)
            except:
                continue
        
        return pd.DataFrame(sentiments)
    
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
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
    
    def get_llm_insights(self, text: str, aspect: str) -> Dict[str, float]:
        try:
            if self.config.LLM_PROVIDER == 'openai':
                model = "gpt-4"
            else:
                model = "claude-3-sonnet-20240229"
            
            prompt = f"""
            Analyze the following text for {aspect} sentiment related to fixed income markets.
            Return a JSON with scores from -1 to 1 for: sentiment, confidence, relevance
            
            Text: {text[:1000]}
            """
            
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            import json
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except:
                return {'sentiment': 0, 'confidence': 0, 'relevance': 0}
                
        except Exception as e:
            return {'sentiment': 0, 'confidence': 0, 'relevance': 0}
    
    def aggregate_sentiment_features(self, ticker: str) -> Dict[str, float]:
        news_sentiment = self.get_news_sentiment(ticker)
        twitter_sentiment = self.get_twitter_sentiment(ticker)
        reddit_sentiment = self.get_reddit_sentiment(ticker)
        
        all_sentiments = []
        if not news_sentiment.empty:
            all_sentiments.extend(news_sentiment['compound'].tolist())
        if not twitter_sentiment.empty:
            all_sentiments.extend(twitter_sentiment['compound'].tolist())
        if not reddit_sentiment.empty:
            all_sentiments.extend(reddit_sentiment['compound'].tolist())
        
        if not all_sentiments:
            return {
                'avg_sentiment': 0,
                'sentiment_volatility': 0,
                'sentiment_momentum': 0,
                'news_count': 0,
                'social_count': 0
            }
        
        return {
            'avg_sentiment': np.mean(all_sentiments),
            'sentiment_volatility': np.std(all_sentiments),
            'sentiment_momentum': np.mean(all_sentiments[-10:]) if len(all_sentiments) >= 10 else np.mean(all_sentiments),
            'news_count': len(news_sentiment),
            'social_count': len(twitter_sentiment) + len(reddit_sentiment)
        } 
import pandas as pd
import numpy as np
import json
import requests
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class FinancialTrainingDataGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_sentiment_training_data(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        positive_examples = [
            ("Federal Reserve signals potential rate cuts, bond yields decline across the curve. Treasury prices rally as investors anticipate accommodative monetary policy.", 
             '{"sentiment": 0.8, "confidence": 0.9, "relevance": 0.95, "risk_level": 0.1}'),
            ("Corporate bond spreads tighten as credit quality improves. Investment-grade bonds show strong performance with declining default rates.", 
             '{"sentiment": 0.7, "confidence": 0.85, "relevance": 0.9, "risk_level": 0.2}'),
            ("Inflation data comes in below expectations, supporting bond market rally. Real yields decline as inflation concerns ease.", 
             '{"sentiment": 0.6, "confidence": 0.8, "relevance": 0.85, "risk_level": 0.15}'),
            ("Municipal bond market shows strength with strong demand from retail investors. Tax-exempt yields remain attractive relative to taxable alternatives.", 
             '{"sentiment": 0.5, "confidence": 0.75, "relevance": 0.8, "risk_level": 0.1}'),
            ("Emerging market bonds rally on improved economic fundamentals. Local currency debt benefits from stronger growth outlook.", 
             '{"sentiment": 0.6, "confidence": 0.8, "relevance": 0.85, "risk_level": 0.3}'),
        ]
        
        negative_examples = [
            ("Federal Reserve signals aggressive rate hikes, bond yields surge to multi-year highs. Treasury prices plummet as inflation concerns mount.", 
             '{"sentiment": -0.8, "confidence": 0.9, "relevance": 0.95, "risk_level": 0.8}'),
            ("Corporate bond spreads widen sharply as credit quality deteriorates. High-yield bonds face selling pressure amid recession fears.", 
             '{"sentiment": -0.7, "confidence": 0.85, "relevance": 0.9, "risk_level": 0.7}'),
            ("Inflation data exceeds expectations, triggering bond market sell-off. Real yields surge as inflation concerns intensify.", 
             '{"sentiment": -0.6, "confidence": 0.8, "relevance": 0.85, "risk_level": 0.6}'),
            ("Municipal bond market faces liquidity concerns as institutional investors reduce exposure. Tax-exempt yields rise amid credit quality worries.", 
             '{"sentiment": -0.5, "confidence": 0.75, "relevance": 0.8, "risk_level": 0.5}'),
            ("Emerging market bonds sell off on deteriorating economic fundamentals. Local currency debt suffers from weaker growth outlook.", 
             '{"sentiment": -0.6, "confidence": 0.8, "relevance": 0.85, "risk_level": 0.7}'),
        ]
        
        neutral_examples = [
            ("Federal Reserve maintains current policy stance, bond yields remain stable. Treasury prices show minimal movement as market awaits economic data.", 
             '{"sentiment": 0.0, "confidence": 0.7, "relevance": 0.85, "risk_level": 0.3}'),
            ("Corporate bond spreads remain unchanged as credit quality stays stable. Investment-grade bonds show mixed performance across sectors.", 
             '{"sentiment": 0.0, "confidence": 0.65, "relevance": 0.8, "risk_level": 0.4}'),
            ("Inflation data meets expectations, bond market shows limited reaction. Real yields remain stable as inflation outlook unchanged.", 
             '{"sentiment": 0.0, "confidence": 0.7, "relevance": 0.85, "risk_level": 0.3}'),
            ("Municipal bond market shows mixed performance with sector-specific movements. Tax-exempt yields remain in recent trading range.", 
             '{"sentiment": 0.0, "confidence": 0.6, "relevance": 0.75, "risk_level": 0.3}'),
            ("Emerging market bonds show mixed performance across regions. Local currency debt reflects varying economic conditions.", 
             '{"sentiment": 0.0, "confidence": 0.65, "relevance": 0.8, "risk_level": 0.5}'),
        ]
        
        for text, label in positive_examples + negative_examples + neutral_examples:
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def generate_triplet_training_data(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        triplet_examples = [
            ("The Federal Reserve raised interest rates by 25 basis points.", 
             '[("Federal Reserve", "raised", "interest rates"), ("Federal Reserve", "raised", "25 basis points")]'),
            ("Bond yields increased as inflation expectations rose.", 
             '[("Bond yields", "increased", "inflation expectations"), ("inflation expectations", "rose", "bond yields")]'),
            ("Corporate spreads widened due to credit quality concerns.", 
             '[("Corporate spreads", "widened", "credit quality concerns"), ("credit quality concerns", "caused", "spread widening")]'),
            ("Treasury prices fell when the Fed signaled rate hikes.", 
             '[("Treasury prices", "fell", "Fed signaling"), ("Fed", "signaled", "rate hikes")]'),
            ("Municipal bonds rallied after the tax reform announcement.", 
             '[("Municipal bonds", "rallied", "tax reform"), ("tax reform", "announced", "municipal rally")]'),
        ]
        
        for text, label in triplet_examples:
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def generate_earnings_call_data(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        earnings_examples = [
            ("Our fixed income portfolio performed well this quarter with strong returns across government and corporate bonds. Credit quality remains excellent with no defaults in our holdings.", 
             '{"sentiment": 0.7, "confidence": 0.85, "relevance": 0.9, "risk_level": 0.2}'),
            ("We experienced some volatility in our bond holdings due to interest rate fluctuations, but our duration management helped mitigate the impact.", 
             '{"sentiment": -0.2, "confidence": 0.75, "relevance": 0.85, "risk_level": 0.4}'),
            ("The high-yield portion of our portfolio underperformed as credit spreads widened, but we remain confident in our credit selection process.", 
             '{"sentiment": -0.3, "confidence": 0.8, "relevance": 0.9, "risk_level": 0.5}'),
            ("Our emerging market bond exposure delivered strong returns as local currencies appreciated against the dollar.", 
             '{"sentiment": 0.6, "confidence": 0.8, "relevance": 0.85, "risk_level": 0.3}'),
        ]
        
        for text, label in earnings_examples:
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def generate_central_bank_data(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        central_bank_examples = [
            ("The Federal Reserve remains committed to achieving maximum employment and price stability. We will continue to monitor economic data closely.", 
             '{"sentiment": 0.1, "confidence": 0.8, "relevance": 0.95, "risk_level": 0.2}'),
            ("Inflation remains elevated and we are prepared to take additional action if necessary to bring it back to our 2% target.", 
             '{"sentiment": -0.3, "confidence": 0.85, "relevance": 0.95, "risk_level": 0.6}'),
            ("The labor market continues to show strength with solid job gains and low unemployment rates.", 
             '{"sentiment": 0.4, "confidence": 0.8, "relevance": 0.85, "risk_level": 0.2}'),
            ("We expect economic growth to moderate in the coming quarters as monetary policy takes effect.", 
             '{"sentiment": -0.2, "confidence": 0.75, "relevance": 0.9, "risk_level": 0.4}'),
        ]
        
        for text, label in central_bank_examples:
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def generate_news_headlines_data(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        headline_examples = [
            ("Bond Market Rally Continues as Fed Signals Dovish Stance", 
             '{"sentiment": 0.6, "confidence": 0.8, "relevance": 0.9, "risk_level": 0.2}'),
            ("Treasury Yields Hit 16-Year High Amid Inflation Concerns", 
             '{"sentiment": -0.5, "confidence": 0.85, "relevance": 0.95, "risk_level": 0.7}'),
            ("Corporate Bond Spreads Tighten on Strong Earnings Reports", 
             '{"sentiment": 0.4, "confidence": 0.75, "relevance": 0.85, "risk_level": 0.3}'),
            ("Municipal Bond Market Faces Liquidity Challenges", 
             '{"sentiment": -0.3, "confidence": 0.7, "relevance": 0.8, "risk_level": 0.5}'),
            ("Emerging Market Bonds Rally on Improved Economic Data", 
             '{"sentiment": 0.5, "confidence": 0.8, "relevance": 0.85, "risk_level": 0.3}'),
        ]
        
        for text, label in headline_examples:
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def generate_social_media_data(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        social_examples = [
            ("Just bought more TLT. Fed is going to cut rates soon and bonds will rally hard! 🚀", 
             '{"sentiment": 0.7, "confidence": 0.6, "relevance": 0.8, "risk_level": 0.4}'),
            ("Bond yields are insane right now. This is going to crash the economy. 😱", 
             '{"sentiment": -0.8, "confidence": 0.5, "relevance": 0.7, "risk_level": 0.8}'),
            ("Corporate bonds looking good this quarter. Credit quality improving across the board.", 
             '{"sentiment": 0.5, "confidence": 0.7, "relevance": 0.8, "risk_level": 0.3}'),
            ("Municipal bonds are a safe haven right now. Tax advantages + stability = win", 
             '{"sentiment": 0.4, "confidence": 0.6, "relevance": 0.75, "risk_level": 0.2}'),
            ("EM bonds are too risky for me. Stick to US treasuries.", 
             '{"sentiment": -0.3, "confidence": 0.65, "relevance": 0.8, "risk_level": 0.6}'),
        ]
        
        for text, label in social_examples:
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def generate_comprehensive_training_data(self) -> Tuple[List[str], List[str]]:
        all_texts = []
        all_labels = []
        
        sources = [
            self.generate_sentiment_training_data,
            self.generate_triplet_training_data,
            self.generate_earnings_call_data,
            self.generate_central_bank_data,
            self.generate_news_headlines_data,
            self.generate_social_media_data
        ]
        
        for source_func in sources:
            texts, labels = source_func()
            all_texts.extend(texts)
            all_labels.extend(labels)
        
        combined = list(zip(all_texts, all_labels))
        np.random.shuffle(combined)
        all_texts, all_labels = zip(*combined)
        
        return list(all_texts), list(all_labels)
    
    def save_training_data(self, texts: List[str], labels: List[str], 
                          output_path: str = "./training_data"):
        import os
        
        os.makedirs(output_path, exist_ok=True)
        
        data = {"texts": texts, "labels": labels}
        with open(os.path.join(output_path, "training_data.json"), "w") as f:
            json.dump(data, f, indent=2)
        
        df = pd.DataFrame({"text": texts, "label": labels})
        df.to_csv(os.path.join(output_path, "training_data.csv"), index=False)
        
        with open(os.path.join(output_path, "texts.txt"), "w") as f:
            for text in texts:
                f.write(text + "\n")
        
        with open(os.path.join(output_path, "labels.txt"), "w") as f:
            for label in labels:
                f.write(label + "\n")
        
        print(f"Training data saved to {output_path}")
        print(f"Total samples: {len(texts)}")
    
    def load_training_data(self, data_path: str = "./training_data") -> Tuple[List[str], List[str]]:
        import os
        
        json_path = os.path.join(data_path, "training_data.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            return data["texts"], data["labels"]
        
        csv_path = os.path.join(data_path, "training_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df["text"].tolist(), df["label"].tolist()
        
        texts_path = os.path.join(data_path, "texts.txt")
        labels_path = os.path.join(data_path, "labels.txt")
        if os.path.exists(texts_path) and os.path.exists(labels_path):
            with open(texts_path, "r") as f:
                texts = [line.strip() for line in f.readlines()]
            with open(labels_path, "r") as f:
                labels = [line.strip() for line in f.readlines()]
            return texts, labels
        
        raise FileNotFoundError(f"No training data found in {data_path}")
    
    def validate_training_data(self, texts: List[str], labels: List[str]) -> bool:
        if len(texts) != len(labels):
            print("Error: Number of texts and labels don't match")
            return False
        
        if len(texts) == 0:
            print("Error: No training data provided")
            return False
        
        empty_texts = sum(1 for text in texts if not text.strip())
        if empty_texts > 0:
            print(f"Warning: {empty_texts} empty texts found")
        
        invalid_labels = 0
        for i, label in enumerate(labels):
            try:
                json.loads(label)
            except json.JSONDecodeError:
                invalid_labels += 1
                print(f"Warning: Invalid JSON in label {i}: {label}")
        
        if invalid_labels > 0:
            print(f"Warning: {invalid_labels} invalid JSON labels found")
        
        print(f"Training data validation complete: {len(texts)} samples")
        return True 
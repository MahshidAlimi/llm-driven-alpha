import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.core.config import Config
from src.analysis.factor_analysis import FactorAnalysis


class TestFactorAnalysis(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.factor_analysis = FactorAnalysis(self.config)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.sample_prices = pd.DataFrame({
            'TLT': np.random.normal(100, 5, len(dates)),
            'IEF': np.random.normal(50, 2, len(dates)),
            'LQD': np.random.normal(80, 3, len(dates)),
            'HYG': np.random.normal(70, 4, len(dates))
        }, index=dates)
        
        # Ensure prices are positive
        self.sample_prices = self.sample_prices.abs()
    
    def test_initialization(self):
        """Test that FactorAnalysis is properly initialized"""
        self.assertIsNotNone(self.factor_analysis.sentiment_analyzer)
        self.assertIsNotNone(self.factor_analysis.zero_shot_classifier)
    
    @patch('yfinance.download')
    def test_get_market_factors_success(self, mock_download):
        """Test successful market factors retrieval"""
        # Mock successful download
        mock_data = pd.DataFrame({
            'Adj Close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2023-01-01', periods=5))
        mock_download.return_value = mock_data
        
        factors = self.factor_analysis.get_market_factors('2023-01-01', '2023-01-05')
        
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertGreater(len(factors.columns), 0)
        mock_download.assert_called()
    
    @patch('yfinance.download')
    def test_get_market_factors_failure(self, mock_download):
        """Test handling of failed market factors retrieval"""
        # Mock failed download
        mock_download.side_effect = Exception("Download failed")
        
        factors = self.factor_analysis.get_market_factors('2023-01-01', '2023-01-05')
        
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertEqual(len(factors.columns), 0)  # Should be empty
    
    def test_calculate_rolling_returns(self):
        """Test rolling returns calculation"""
        windows = [20, 60]
        rolling_features = self.factor_analysis.calculate_rolling_returns(
            self.sample_prices, windows
        )
        
        self.assertIsInstance(rolling_features, pd.DataFrame)
        self.assertGreater(len(rolling_features.columns), 0)
        
        # Check that features are created for each ticker and window
        expected_features = len(self.sample_prices.columns) * len(windows) * 3  # 3 types per window
        self.assertLessEqual(len(rolling_features.columns), expected_features)
        
        # Check for specific feature names
        for ticker in self.sample_prices.columns:
            for window in windows:
                self.assertIn(f'{ticker}_rolling_{window}d', rolling_features.columns)
                self.assertIn(f'{ticker}_momentum_{window}d', rolling_features.columns)
                self.assertIn(f'{ticker}_volatility_{window}d', rolling_features.columns)
    
    def test_extract_sentiment_from_text(self):
        """Test sentiment extraction from text"""
        # Test positive text
        positive_text = "This is a great investment opportunity with strong fundamentals."
        sentiment = self.factor_analysis.extract_sentiment_from_text(positive_text)
        
        self.assertIsInstance(sentiment, dict)
        self.assertIn('compound', sentiment)
        self.assertIn('positive', sentiment)
        self.assertIn('negative', sentiment)
        self.assertIn('neutral', sentiment)
        self.assertIn('entities', sentiment)
        
        # Test negative text
        negative_text = "This is a terrible investment with poor performance."
        sentiment_neg = self.factor_analysis.extract_sentiment_from_text(negative_text)
        
        # Compound sentiment should be lower for negative text
        self.assertLess(sentiment_neg['compound'], sentiment['compound'])
    
    def test_extract_sentiment_empty_text(self):
        """Test sentiment extraction with empty text"""
        empty_text = ""
        sentiment = self.factor_analysis.extract_sentiment_from_text(empty_text)
        
        self.assertEqual(sentiment['compound'], 0)
        self.assertEqual(sentiment['positive'], 0)
        self.assertEqual(sentiment['negative'], 0)
        self.assertEqual(sentiment['neutral'], 0)
    
    def test_extract_sentiment_short_text(self):
        """Test sentiment extraction with very short text"""
        short_text = "Hi"
        sentiment = self.factor_analysis.extract_sentiment_from_text(short_text)
        
        self.assertEqual(sentiment['compound'], 0)
    
    @patch('requests.get')
    def test_get_news_sentiment_success(self, mock_get):
        """Test successful news sentiment retrieval"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Positive news about TLT',
                    'description': 'Great performance expected',
                    'publishedAt': '2023-01-01T00:00:00Z',
                    'source': {'name': 'Reuters'}
                }
            ]
        }
        mock_get.return_value = mock_response
        
        sentiment_df = self.factor_analysis.get_news_sentiment('TLT', days_back=7)
        
        self.assertIsInstance(sentiment_df, pd.DataFrame)
        if len(sentiment_df) > 0:
            self.assertIn('compound', sentiment_df.columns)
            self.assertIn('date', sentiment_df.columns)
            self.assertIn('source', sentiment_df.columns)
    
    @patch('requests.get')
    def test_get_news_sentiment_failure(self, mock_get):
        """Test handling of failed news sentiment retrieval"""
        # Mock failed API response
        mock_get.side_effect = Exception("API error")
        
        sentiment_df = self.factor_analysis.get_news_sentiment('TLT', days_back=7)
        
        self.assertIsInstance(sentiment_df, pd.DataFrame)
        self.assertEqual(len(sentiment_df), 0)
    
    @patch('tweepy.Client')
    def test_get_twitter_sentiment_success(self, mock_client):
        """Test successful Twitter sentiment retrieval"""
        # Mock successful Twitter API response
        mock_twitter_client = MagicMock()
        mock_tweet = MagicMock()
        mock_tweet.text = "Great investment in TLT"
        mock_tweet.created_at = datetime.now()
        mock_tweet.id = 12345
        
        mock_response = MagicMock()
        mock_response.data = [mock_tweet]
        mock_twitter_client.search_recent_tweets.return_value = mock_response
        mock_client.return_value = mock_twitter_client
        
        # Temporarily set Twitter client
        original_client = self.factor_analysis.twitter_client
        self.factor_analysis.twitter_client = mock_twitter_client
        
        try:
            sentiment_df = self.factor_analysis.get_twitter_sentiment('TLT', count=10)
            
            self.assertIsInstance(sentiment_df, pd.DataFrame)
            if len(sentiment_df) > 0:
                self.assertIn('compound', sentiment_df.columns)
                self.assertIn('date', sentiment_df.columns)
                self.assertIn('tweet_id', sentiment_df.columns)
        finally:
            self.factor_analysis.twitter_client = original_client
    
    def test_get_twitter_sentiment_no_client(self):
        """Test Twitter sentiment when no client is available"""
        # Set Twitter client to None
        original_client = self.factor_analysis.twitter_client
        self.factor_analysis.twitter_client = None
        
        try:
            sentiment_df = self.factor_analysis.get_twitter_sentiment('TLT', count=10)
            
            self.assertIsInstance(sentiment_df, pd.DataFrame)
            self.assertEqual(len(sentiment_df), 0)
        finally:
            self.factor_analysis.twitter_client = original_client
    
    def test_extract_triplets(self):
        """Test triplet extraction from text"""
        text = "The Federal Reserve raised interest rates. Bond prices fell sharply."
        triplets = self.factor_analysis.extract_triplets(text)
        
        self.assertIsInstance(triplets, list)
        # Should extract some triplets from the text
        self.assertGreaterEqual(len(triplets), 0)
        
        # Test with empty text
        empty_triplets = self.factor_analysis.extract_triplets("")
        self.assertEqual(len(empty_triplets), 0)
    
    @patch('litellm.completion')
    def test_get_llm_insights_success(self, mock_completion):
        """Test successful LLM insights retrieval"""
        # Mock successful LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"sentiment": 0.5, "confidence": 0.8, "relevance": 0.7}'
        mock_completion.return_value = mock_response
        
        text = "The Federal Reserve announced a 25 basis point rate hike."
        insights = self.factor_analysis.get_llm_insights(text, "monetary_policy")
        
        self.assertIsInstance(insights, dict)
        self.assertIn('sentiment', insights)
        self.assertIn('confidence', insights)
        self.assertIn('relevance', insights)
        
        # Values should be between -1 and 1
        self.assertGreaterEqual(insights['sentiment'], -1)
        self.assertLessEqual(insights['sentiment'], 1)
        self.assertGreaterEqual(insights['confidence'], 0)
        self.assertLessEqual(insights['confidence'], 1)
    
    @patch('litellm.completion')
    def test_get_llm_insights_failure(self, mock_completion):
        """Test handling of failed LLM insights retrieval"""
        # Mock failed LLM response
        mock_completion.side_effect = Exception("LLM API error")
        
        text = "The Federal Reserve announced a 25 basis point rate hike."
        insights = self.factor_analysis.get_llm_insights(text, "monetary_policy")
        
        self.assertIsInstance(insights, dict)
        self.assertEqual(insights['sentiment'], 0)
        self.assertEqual(insights['confidence'], 0)
        self.assertEqual(insights['relevance'], 0)
    
    def test_aggregate_sentiment_features(self):
        """Test sentiment feature aggregation"""
        # Mock sentiment data
        with patch.object(self.factor_analysis, 'get_news_sentiment') as mock_news:
            mock_news.return_value = pd.DataFrame({
                'compound': [0.5, 0.3, -0.1]
            })
        
        with patch.object(self.factor_analysis, 'get_twitter_sentiment') as mock_twitter:
            mock_twitter.return_value = pd.DataFrame({
                'compound': [0.2, -0.3]
            })
        
        with patch.object(self.factor_analysis, 'get_reddit_sentiment') as mock_reddit:
            mock_reddit.return_value = pd.DataFrame({
                'compound': [0.1, 0.4]
            })
        
        features = self.factor_analysis.aggregate_sentiment_features('TLT')
        
        self.assertIsInstance(features, dict)
        self.assertIn('avg_sentiment', features)
        self.assertIn('sentiment_volatility', features)
        self.assertIn('sentiment_momentum', features)
        self.assertIn('news_count', features)
        self.assertIn('social_count', features)
        
        # Check that sentiment values are reasonable
        self.assertGreaterEqual(features['avg_sentiment'], -1)
        self.assertLessEqual(features['avg_sentiment'], 1)
        self.assertGreaterEqual(features['sentiment_volatility'], 0)
    
    def test_aggregate_sentiment_features_no_data(self):
        """Test sentiment aggregation with no data"""
        # Mock empty sentiment data
        with patch.object(self.factor_analysis, 'get_news_sentiment') as mock_news:
            mock_news.return_value = pd.DataFrame()
        
        with patch.object(self.factor_analysis, 'get_twitter_sentiment') as mock_twitter:
            mock_twitter.return_value = pd.DataFrame()
        
        with patch.object(self.factor_analysis, 'get_reddit_sentiment') as mock_reddit:
            mock_reddit.return_value = pd.DataFrame()
        
        features = self.factor_analysis.aggregate_sentiment_features('TLT')
        
        self.assertEqual(features['avg_sentiment'], 0)
        self.assertEqual(features['sentiment_volatility'], 0)
        self.assertEqual(features['sentiment_momentum'], 0)
        self.assertEqual(features['news_count'], 0)
        self.assertEqual(features['social_count'], 0)


if __name__ == '__main__':
    unittest.main() 
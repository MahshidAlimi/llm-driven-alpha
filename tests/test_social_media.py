#!/usr/bin/env python3

import os
import unittest
import pandas as pd
from src.core.lora_config import LoRAConfig
from src.analysis.lora_factor_analysis import LoRAFactorAnalysis

class TestSocialMedia(unittest.TestCase):
    
    def setUp(self):
        self.config = LoRAConfig()
        self.factor_analysis = LoRAFactorAnalysis(self.config)
    
    def test_social_media_config(self):
        self.assertIsInstance(self.config.TWITTER_BEARER_TOKEN, (str, type(None)))
        self.assertIsInstance(self.config.REDDIT_CLIENT_ID, (str, type(None)))
        self.assertIsInstance(self.config.REDDIT_CLIENT_SECRET, (str, type(None)))
        self.assertIsInstance(self.config.REDDIT_USER_AGENT, (str, type(None)))
    
    def test_twitter_sentiment(self):
        twitter_data = self.factor_analysis.get_twitter_sentiment("TLT", count=10)
        self.assertIsInstance(twitter_data, pd.DataFrame)
    
    def test_reddit_sentiment(self):
        reddit_data = self.factor_analysis.get_reddit_sentiment("TLT")
        self.assertIsInstance(reddit_data, pd.DataFrame)

if __name__ == "__main__":
    unittest.main() 
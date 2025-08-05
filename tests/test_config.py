import unittest
import os
import tempfile
from unittest.mock import patch

from src.core.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary environment file for testing
        self.temp_env_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_env_file.write("""
OPENAI_API_KEY=test_openai_key
ANTHROPIC_API_KEY=test_anthropic_key
TWITTER_BEARER_TOKEN=test_twitter_token
REDDIT_CLIENT_ID=test_reddit_id
REDDIT_CLIENT_SECRET=test_reddit_secret
REDDIT_USER_AGENT=test_user_agent
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
MONGODB_URI=mongodb://test:27017/
REDIS_URL=redis://test:6379/
        """)
        self.temp_env_file.close()
    
    def tearDown(self):
        # Clean up temporary file
        os.unlink(self.temp_env_file.name)
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'TWITTER_BEARER_TOKEN': 'test_twitter_token',
        'REDDIT_CLIENT_ID': 'test_reddit_id',
        'REDDIT_CLIENT_SECRET': 'test_reddit_secret',
        'REDDIT_USER_AGENT': 'test_user_agent',
        'LLM_PROVIDER': 'openai',
        'LLM_MODEL': 'gpt-4'
    })
    def test_config_initialization(self):
        """Test that Config is properly initialized with environment variables"""
        config = Config()
        
        self.assertEqual(config.OPENAI_API_KEY, 'test_openai_key')
        self.assertEqual(config.ANTHROPIC_API_KEY, 'test_anthropic_key')
        self.assertEqual(config.TWITTER_BEARER_TOKEN, 'test_twitter_token')
        self.assertEqual(config.REDDIT_CLIENT_ID, 'test_reddit_id')
        self.assertEqual(config.REDDIT_CLIENT_SECRET, 'test_reddit_secret')
        self.assertEqual(config.REDDIT_USER_AGENT, 'test_user_agent')
        self.assertEqual(config.LLM_PROVIDER, 'openai')
        self.assertEqual(config.LLM_MODEL, 'gpt-4')
    
    def test_config_defaults(self):
        """Test that Config uses default values when environment variables are not set"""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            
            self.assertIsNone(config.OPENAI_API_KEY)
            self.assertIsNone(config.ANTHROPIC_API_KEY)
            self.assertIsNone(config.TWITTER_BEARER_TOKEN)
            self.assertIsNone(config.REDDIT_CLIENT_ID)
            self.assertIsNone(config.REDDIT_CLIENT_SECRET)
            self.assertIsNone(config.REDDIT_USER_AGENT)
            self.assertEqual(config.LLM_PROVIDER, 'openai')  # Default value
            self.assertEqual(config.LLM_MODEL, 'gpt-4')      # Default value
            self.assertEqual(config.MONGODB_URI, 'mongodb://localhost:27017/')  # Default value
            self.assertEqual(config.REDIS_URL, 'redis://localhost:6379/')       # Default value
    
    def test_fixed_income_universe(self):
        """Test that the fixed income universe is properly defined"""
        config = Config()
        
        self.assertIsInstance(config.FIXED_INCOME_UNIVERSE, list)
        self.assertGreater(len(config.FIXED_INCOME_UNIVERSE), 0)
        
        # Check for some expected ETFs
        expected_etfs = ['TLT', 'IEF', 'LQD', 'HYG', 'BND']
        for etf in expected_etfs:
            self.assertIn(etf, config.FIXED_INCOME_UNIVERSE)
    
    def test_sector_constraints(self):
        """Test that sector constraints are properly defined"""
        config = Config()
        
        self.assertIsInstance(config.SECTOR_CONSTRAINTS, dict)
        self.assertGreater(len(config.SECTOR_CONSTRAINTS), 0)
        
        # Check that all constraints are between 0 and 1
        for sector, constraint in config.SECTOR_CONSTRAINTS.items():
            self.assertGreaterEqual(constraint, 0)
            self.assertLessEqual(constraint, 1)
        
        # Check for expected sectors
        expected_sectors = ['government', 'corporate', 'emerging_markets', 'high_yield']
        for sector in expected_sectors:
            self.assertIn(sector, config.SECTOR_CONSTRAINTS)
    
    def test_backtest_parameters(self):
        """Test that backtest parameters are properly defined"""
        config = Config()
        
        self.assertIsInstance(config.BACKTEST_START_DATE, str)
        self.assertIsInstance(config.BACKTEST_END_DATE, str)
        self.assertIsInstance(config.REBALANCE_FREQUENCY, str)
        self.assertIsInstance(config.LOOKBACK_PERIOD, int)
        self.assertIsInstance(config.ROLLING_WINDOW, int)
        
        # Check that dates are in correct format
        from datetime import datetime
        try:
            datetime.strptime(config.BACKTEST_START_DATE, '%Y-%m-%d')
            datetime.strptime(config.BACKTEST_END_DATE, '%Y-%m-%d')
        except ValueError:
            self.fail("Date format should be YYYY-MM-DD")
        
        # Check that start date is before end date
        start_date = datetime.strptime(config.BACKTEST_START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(config.BACKTEST_END_DATE, '%Y-%m-%d')
        self.assertLess(start_date, end_date)
    
    def test_risk_parameters(self):
        """Test that risk parameters are properly defined"""
        config = Config()
        
        self.assertIsInstance(config.RISK_FREE_RATE, float)
        self.assertIsInstance(config.TARGET_VOLATILITY, float)
        self.assertIsInstance(config.MAX_WEIGHT, float)
        self.assertIsInstance(config.MIN_WEIGHT, float)
        
        # Check that parameters are reasonable
        self.assertGreaterEqual(config.RISK_FREE_RATE, 0)
        self.assertLessEqual(config.RISK_FREE_RATE, 1)
        
        self.assertGreater(config.TARGET_VOLATILITY, 0)
        self.assertLessEqual(config.TARGET_VOLATILITY, 1)
        
        self.assertGreater(config.MAX_WEIGHT, 0)
        self.assertLessEqual(config.MAX_WEIGHT, 1)
        
        self.assertGreaterEqual(config.MIN_WEIGHT, 0)
        self.assertLess(config.MIN_WEIGHT, config.MAX_WEIGHT)
    
    def test_rebalance_frequency_validation(self):
        """Test that rebalance frequency is valid"""
        config = Config()
        
        valid_frequencies = ['D', 'W', 'M', 'Q', 'Y']
        self.assertIn(config.REBALANCE_FREQUENCY, valid_frequencies)


if __name__ == '__main__':
    unittest.main() 
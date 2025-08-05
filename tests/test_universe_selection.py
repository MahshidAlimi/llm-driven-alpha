import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.core.config import Config
from src.core.universe_selection import FixedIncomeUniverse


class TestFixedIncomeUniverse(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.universe = FixedIncomeUniverse(self.config)
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.sample_prices = pd.DataFrame({
            'TLT': np.random.normal(100, 5, len(dates)),
            'IEF': np.random.normal(50, 2, len(dates)),
            'LQD': np.random.normal(80, 3, len(dates)),
            'HYG': np.random.normal(70, 4, len(dates)),
            'BND': np.random.normal(90, 2, len(dates))
        }, index=dates)
        
        # Ensure prices are positive
        self.sample_prices = self.sample_prices.abs()
    
    def test_initialization(self):
        """Test that the universe is properly initialized"""
        self.assertIsNotNone(self.universe.universe)
        self.assertIsNotNone(self.universe.sector_mapping)
        self.assertEqual(len(self.universe.universe), 18)  # Should have 18 ETFs
        self.assertIn('TLT', self.universe.universe)
        self.assertIn('IEF', self.universe.universe)
    
    def test_sector_mapping(self):
        """Test that sector mapping is correct"""
        self.assertEqual(self.universe.sector_mapping['TLT'], 'government')
        self.assertEqual(self.universe.sector_mapping['LQD'], 'corporate')
        self.assertEqual(self.universe.sector_mapping['HYG'], 'high_yield')
        self.assertEqual(self.universe.sector_mapping['EMB'], 'emerging_markets')
    
    @patch('yfinance.download')
    def test_get_historical_data_success(self, mock_download):
        """Test successful historical data download"""
        # Mock successful download
        mock_data = pd.DataFrame({
            'Adj Close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2023-01-01', periods=5))
        mock_download.return_value = mock_data
        
        result = self.universe.get_historical_data('2023-01-01', '2023-01-05')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), 0)
        mock_download.assert_called()
    
    @patch('yfinance.download')
    def test_get_historical_data_failure(self, mock_download):
        """Test handling of failed data download"""
        # Mock failed download
        mock_download.side_effect = Exception("Download failed")
        
        result = self.universe.get_historical_data('2023-01-01', '2023-01-05')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 0)  # Should be empty
    
    def test_calculate_metrics(self):
        """Test calculation of financial metrics"""
        metrics = self.universe.calculate_metrics(self.sample_prices)
        
        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual(len(metrics), len(self.sample_prices.columns))
        
        # Check that all required metrics are present
        required_metrics = ['volatility', 'sharpe_ratio', 'max_drawdown', 'avg_volume', 'sector']
        for metric in required_metrics:
            self.assertIn(metric, metrics.columns)
        
        # Check that volatility is positive
        self.assertTrue(all(metrics['volatility'] >= 0))
        
        # Check that max drawdown is negative or zero
        self.assertTrue(all(metrics['max_drawdown'] <= 0))
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create a price series with a clear drawdown
        prices = pd.Series([100, 110, 120, 80, 90, 100])
        max_dd = self.universe._calculate_max_drawdown(prices)
        
        # Should be negative (drawdown)
        self.assertLess(max_dd, 0)
        # Should be approximately -0.33 (from 120 to 80)
        self.assertAlmostEqual(max_dd, -0.333, places=2)
    
    @patch('yfinance.Ticker')
    def test_get_avg_volume_success(self, mock_ticker):
        """Test successful average volume retrieval"""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'averageVolume': 1000000}
        mock_ticker.return_value = mock_ticker_instance
        
        volume = self.universe._get_avg_volume('TLT')
        self.assertEqual(volume, 1000000)
    
    @patch('yfinance.Ticker')
    def test_get_avg_volume_failure(self, mock_ticker):
        """Test handling of failed volume retrieval"""
        mock_ticker.side_effect = Exception("API error")
        
        volume = self.universe._get_avg_volume('TLT')
        self.assertEqual(volume, 0)
    
    def test_filter_universe(self):
        """Test universe filtering"""
        # Test with reasonable parameters
        filtered = self.universe.filter_universe(
            self.sample_prices,
            min_volatility=0.01,
            max_volatility=0.50,
            min_sharpe=-1.0
        )
        
        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), len(self.sample_prices.columns))
        
        # Test with very restrictive parameters
        filtered_restrictive = self.universe.filter_universe(
            self.sample_prices,
            min_volatility=0.50,  # Very high minimum
            max_volatility=0.10,  # Very low maximum
            min_sharpe=2.0        # Very high Sharpe
        )
        
        # Should return fewer or no securities
        self.assertLessEqual(len(filtered_restrictive), len(filtered))
    
    def test_get_sector_weights(self):
        """Test sector weight calculation"""
        selected_tickers = ['TLT', 'IEF', 'LQD', 'HYG']
        sector_weights = self.universe.get_sector_weights(selected_tickers)
        
        self.assertIsInstance(sector_weights, dict)
        self.assertAlmostEqual(sum(sector_weights.values()), 1.0, places=5)
        
        # Check specific sectors
        self.assertIn('government', sector_weights)
        self.assertIn('corporate', sector_weights)
        self.assertIn('high_yield', sector_weights)
        
        # TLT and IEF are government, so government should have 0.5 weight
        self.assertEqual(sector_weights['government'], 0.5)
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_prices = pd.DataFrame()
        metrics = self.universe.calculate_metrics(empty_prices)
        
        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual(len(metrics), 0)
    
    def test_single_asset_data(self):
        """Test handling of single asset data"""
        single_prices = self.sample_prices[['TLT']]
        metrics = self.universe.calculate_metrics(single_prices)
        
        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual(len(metrics), 1)
        self.assertIn('TLT', metrics.index)


if __name__ == '__main__':
    unittest.main() 
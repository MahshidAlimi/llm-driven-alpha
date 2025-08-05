import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.core.config import Config
from src.core.universe_selection import FixedIncomeUniverse
from src.optimization.optimization import PortfolioOptimizer
from src.backtesting.backtest import Backtester

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.universe = FixedIncomeUniverse(self.config)
        self.optimizer = PortfolioOptimizer(self.config)
        self.backtester = Backtester(self.config)
        
        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.sample_prices = pd.DataFrame({
            'TLT': np.random.normal(100, 5, len(dates)),
            'IEF': np.random.normal(50, 2, len(dates)),
            'LQD': np.random.normal(80, 3, len(dates)),
            'HYG': np.random.normal(70, 4, len(dates))
        }, index=dates)
        
        # Ensure prices are positive
        self.sample_prices = self.sample_prices.abs()
    
    def test_end_to_end_workflow(self):
        """Test the complete end-to-end workflow"""
        # 1. Universe Selection
        selected_tickers = self.universe.filter_universe(self.sample_prices)
        self.assertIsInstance(selected_tickers, list)
        self.assertGreater(len(selected_tickers), 0)
        
        # 2. Portfolio Optimization
        filtered_prices = self.sample_prices[selected_tickers]
        returns = filtered_prices.pct_change().dropna()
        
        mvo_result = self.optimizer.mean_variance_optimization(returns)
        self.assertIsNotNone(mvo_result)
        self.assertIn('weights', mvo_result)
        self.assertIn('expected_return', mvo_result)
        self.assertIn('volatility', mvo_result)
        self.assertIn('sharpe_ratio', mvo_result)
        
        # 3. Backtesting
        weights = mvo_result['weights']
        weights_history = pd.DataFrame([weights] * len(filtered_prices), 
                                     index=filtered_prices.index)
        
        backtest_result = self.backtester.run_backtest(
            filtered_prices, 
            weights_history, 
            'M'
        )
        
        self.assertIsNotNone(backtest_result)
        self.assertIn('portfolio_values', backtest_result)
        self.assertIn('performance_metrics', backtest_result)
        self.assertIn('risk_metrics', backtest_result)
        
        # 4. Report Generation
        report = self.backtester.generate_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
    
    def test_multiple_optimization_strategies(self):
        """Test multiple optimization strategies"""
        selected_tickers = self.universe.filter_universe(self.sample_prices)
        filtered_prices = self.sample_prices[selected_tickers]
        returns = filtered_prices.pct_change().dropna()
        
        # Test Mean-Variance Optimization
        mvo_result = self.optimizer.mean_variance_optimization(returns)
        self.assertIsNotNone(mvo_result)
        
        # Test Risk Parity Optimization
        rp_result = self.optimizer.risk_parity_optimization(returns)
        self.assertIsNotNone(rp_result)
        
        # Test Sector Constrained Optimization
        sector_mapping = {ticker: 'government' for ticker in selected_tickers}
        sector_constraints = {'government': 1.0}
        
        sc_result = self.optimizer.sector_constrained_optimization(
            returns, sector_mapping, sector_constraints
        )
        self.assertIsNotNone(sc_result)
    
    def test_different_rebalancing_frequencies(self):
        """Test different rebalancing frequencies"""
        selected_tickers = self.universe.filter_universe(self.sample_prices)
        filtered_prices = self.sample_prices[selected_tickers]
        returns = filtered_prices.pct_change().dropna()
        
        mvo_result = self.optimizer.mean_variance_optimization(returns)
        weights = mvo_result['weights']
        weights_history = pd.DataFrame([weights] * len(filtered_prices), 
                                     index=filtered_prices.index)
        
        frequencies = ['D', 'W', 'M', 'Q']
        
        for freq in frequencies:
            result = self.backtester.run_backtest(
                filtered_prices, 
                weights_history, 
                freq
            )
            
            self.assertIsNotNone(result)
            self.assertIn('portfolio_values', result)
            self.assertIn('performance_metrics', result)
    
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Test with empty data
        empty_prices = pd.DataFrame()
        empty_result = self.universe.filter_universe(empty_prices)
        self.assertEqual(len(empty_result), 0)
        
        # Test with single asset
        single_prices = self.sample_prices[['TLT']]
        single_result = self.universe.filter_universe(single_prices)
        self.assertIsInstance(single_result, list)
        
        # Test optimization with insufficient data
        short_returns = returns.iloc[:10] if len(returns) > 10 else returns
        if len(short_returns) > 0:
            opt_result = self.optimizer.mean_variance_optimization(short_returns)
            # Should either return a result or None, but not crash
            if opt_result is not None:
                self.assertIn('weights', opt_result)
    
    def test_performance_metrics_consistency(self):
        """Test that performance metrics are consistent"""
        selected_tickers = self.universe.filter_universe(self.sample_prices)
        filtered_prices = self.sample_prices[selected_tickers]
        returns = filtered_prices.pct_change().dropna()
        
        mvo_result = self.optimizer.mean_variance_optimization(returns)
        weights = mvo_result['weights']
        weights_history = pd.DataFrame([weights] * len(filtered_prices), 
                                     index=filtered_prices.index)
        
        backtest_result = self.backtester.run_backtest(
            filtered_prices, 
            weights_history, 
            'M'
        )
        
        performance = backtest_result['performance_metrics']
        risk = backtest_result['risk_metrics']
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(performance['total_return'], -100)  # Not more than 100% loss
        self.assertLessEqual(performance['total_return'], 1000)     # Not more than 1000% gain
        
        self.assertGreaterEqual(risk['volatility'], 0)
        self.assertLessEqual(risk['max_drawdown'], 0)
        
        # Sharpe ratio should be finite
        self.assertTrue(np.isfinite(performance['sharpe_ratio']))
        
        # Win rate should be between 0 and 1
        self.assertGreaterEqual(performance['win_rate'], 0)
        self.assertLessEqual(performance['win_rate'], 1)

if __name__ == '__main__':
    unittest.main() 
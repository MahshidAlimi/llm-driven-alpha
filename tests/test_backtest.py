import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.core.config import Config
from src.backtesting.backtest import Backtester


class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.backtester = Backtester(self.config)
        
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.sample_prices = pd.DataFrame({
            'TLT': np.random.normal(100, 5, len(dates)),
            'IEF': np.random.normal(50, 2, len(dates)),
            'LQD': np.random.normal(80, 3, len(dates)),
            'HYG': np.random.normal(70, 4, len(dates))
        }, index=dates)
        
        # Ensure prices are positive and monotonically increasing
        self.sample_prices = self.sample_prices.abs()
        for col in self.sample_prices.columns:
            self.sample_prices[col] = self.sample_prices[col].cumsum() + 100
        
        # Create sample weights history
        self.sample_weights = pd.DataFrame({
            'TLT': [0.3, 0.3, 0.3, 0.3, 0.3],
            'IEF': [0.3, 0.3, 0.3, 0.3, 0.3],
            'LQD': [0.2, 0.2, 0.2, 0.2, 0.2],
            'HYG': [0.2, 0.2, 0.2, 0.2, 0.2]
        }, index=self.sample_prices.index[:5])
    
    def test_initialization(self):
        """Test that Backtester is properly initialized"""
        self.assertIsNotNone(self.backtester.config)
        self.assertEqual(self.backtester.results, {})
    
    def test_should_rebalance_daily(self):
        """Test daily rebalancing logic"""
        current_date = pd.Timestamp('2023-01-02')
        previous_date = pd.Timestamp('2023-01-01')
        
        result = self.backtester._should_rebalance(current_date, previous_date, 'D')
        self.assertTrue(result)
    
    def test_should_rebalance_monthly(self):
        """Test monthly rebalancing logic"""
        current_date = pd.Timestamp('2023-02-01')
        previous_date = pd.Timestamp('2023-01-31')
        
        result = self.backtester._should_rebalance(current_date, previous_date, 'M')
        self.assertTrue(result)
        
        # Same month should not rebalance
        current_date = pd.Timestamp('2023-01-15')
        previous_date = pd.Timestamp('2023-01-14')
        result = self.backtester._should_rebalance(current_date, previous_date, 'M')
        self.assertFalse(result)
    
    def test_should_rebalance_quarterly(self):
        """Test quarterly rebalancing logic"""
        current_date = pd.Timestamp('2023-04-01')
        previous_date = pd.Timestamp('2023-03-31')
        
        result = self.backtester._should_rebalance(current_date, previous_date, 'Q')
        self.assertTrue(result)
    
    def test_should_rebalance_yearly(self):
        """Test yearly rebalancing logic"""
        current_date = pd.Timestamp('2024-01-01')
        previous_date = pd.Timestamp('2023-12-31')
        
        result = self.backtester._should_rebalance(current_date, previous_date, 'Y')
        self.assertTrue(result)
    
    def test_calculate_benchmark(self):
        """Test benchmark calculation"""
        benchmark = self.backtester._calculate_benchmark(self.sample_prices)
        
        self.assertIsInstance(benchmark, pd.Series)
        self.assertEqual(len(benchmark), len(self.sample_prices) - 1)
        
        # Benchmark should start at 1,000,000
        self.assertAlmostEqual(benchmark.iloc[0], 1000000, places=0)
        
        # Benchmark should be monotonically increasing (generally)
        self.assertTrue(all(benchmark.diff().dropna() >= -1e-6))  # Allow small numerical errors
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        # Create sample portfolio and benchmark
        portfolio = pd.Series([1000000, 1010000, 1020000, 1030000, 1040000])
        benchmark = pd.Series([1000000, 1005000, 1010000, 1015000, 1020000])
        
        metrics = self.backtester._calculate_performance_metrics(portfolio, benchmark)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('benchmark_return', metrics)
        self.assertIn('excess_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('information_ratio', metrics)
        self.assertIn('calmar_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        
        # Total return should be positive for this example
        self.assertGreater(metrics['total_return'], 0)
        self.assertGreater(metrics['benchmark_return'], 0)
        self.assertGreater(metrics['excess_return'], 0)
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation"""
        # Create sample portfolio with known characteristics
        portfolio = pd.Series([1000000, 1010000, 1020000, 1030000, 1040000])
        
        risk_metrics = self.backtester._calculate_risk_metrics(portfolio)
        
        self.assertIsInstance(risk_metrics, dict)
        self.assertIn('volatility', risk_metrics)
        self.assertIn('var_95', risk_metrics)
        self.assertIn('cvar_95', risk_metrics)
        self.assertIn('var_99', risk_metrics)
        self.assertIn('cvar_99', risk_metrics)
        self.assertIn('max_drawdown', risk_metrics)
        self.assertIn('skewness', risk_metrics)
        self.assertIn('kurtosis', risk_metrics)
        self.assertIn('downside_deviation', risk_metrics)
        
        # Volatility should be positive
        self.assertGreater(risk_metrics['volatility'], 0)
        
        # Max drawdown should be negative or zero
        self.assertLessEqual(risk_metrics['max_drawdown'], 0)
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation"""
        # Create a portfolio with a clear drawdown
        portfolio = pd.Series([1000000, 1100000, 1200000, 800000, 900000, 1000000])
        
        drawdown_analysis = self.backtester._calculate_drawdown(portfolio)
        
        self.assertIsInstance(drawdown_analysis, dict)
        self.assertIn('max_drawdown', drawdown_analysis)
        self.assertIn('max_drawdown_date', drawdown_analysis)
        self.assertIn('recovery_date', drawdown_analysis)
        self.assertIn('drawdown_series', drawdown_analysis)
        
        # Max drawdown should be negative
        self.assertLess(drawdown_analysis['max_drawdown'], 0)
        
        # Should be approximately -0.33 (from 1200000 to 800000)
        self.assertAlmostEqual(drawdown_analysis['max_drawdown'], -0.333, places=2)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create a portfolio with a clear drawdown
        portfolio = pd.Series([1000000, 1100000, 1200000, 800000, 900000, 1000000])
        
        max_dd = self.backtester._calculate_max_drawdown(portfolio)
        
        self.assertIsInstance(max_dd, float)
        self.assertLess(max_dd, 0)
        self.assertAlmostEqual(max_dd, -0.333, places=2)
    
    def test_run_backtest_success(self):
        """Test successful backtest execution"""
        result = self.backtester.run_backtest(
            self.sample_prices, 
            self.sample_weights, 
            rebalance_frequency='M',
            transaction_costs=0.001
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('portfolio_values', result)
        self.assertIn('weights_history', result)
        self.assertIn('benchmark', result)
        self.assertIn('performance_metrics', result)
        self.assertIn('risk_metrics', result)
        self.assertIn('drawdown_analysis', result)
        
        # Portfolio values should be a series
        self.assertIsInstance(result['portfolio_values'], pd.Series)
        self.assertGreater(len(result['portfolio_values']), 0)
        
        # Weights history should be a dataframe
        self.assertIsInstance(result['weights_history'], pd.DataFrame)
        self.assertGreater(len(result['weights_history']), 0)
        
        # Benchmark should be a series
        self.assertIsInstance(result['benchmark'], pd.Series)
        self.assertGreater(len(result['benchmark']), 0)
    
    def test_run_backtest_different_frequencies(self):
        """Test backtest with different rebalancing frequencies"""
        frequencies = ['D', 'W', 'M', 'Q', 'Y']
        
        for freq in frequencies:
            result = self.backtester.run_backtest(
                self.sample_prices, 
                self.sample_weights, 
                rebalance_frequency=freq,
                transaction_costs=0.001
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('portfolio_values', result)
            self.assertIn('performance_metrics', result)
    
    def test_run_backtest_different_transaction_costs(self):
        """Test backtest with different transaction costs"""
        costs = [0.0, 0.001, 0.01]
        
        for cost in costs:
            result = self.backtester.run_backtest(
                self.sample_prices, 
                self.sample_weights, 
                rebalance_frequency='M',
                transaction_costs=cost
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('portfolio_values', result)
            self.assertIn('performance_metrics', result)
    
    def test_generate_report(self):
        """Test report generation"""
        # Run a backtest first
        result = self.backtester.run_backtest(
            self.sample_prices, 
            self.sample_weights, 
            rebalance_frequency='M'
        )
        
        report = self.backtester.generate_report()
        
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Check that report contains expected sections
        self.assertIn('FIXED INCOME PORTFOLIO BACKTEST REPORT', report)
        self.assertIn('PERFORMANCE METRICS', report)
        self.assertIn('RISK-ADJUSTED METRICS', report)
        self.assertIn('RISK METRICS', report)
        self.assertIn('TRADING METRICS', report)
    
    def test_generate_report_no_results(self):
        """Test report generation with no backtest results"""
        # Clear results
        self.backtester.results = {}
        
        report = self.backtester.generate_report()
        
        self.assertIsInstance(report, str)
        self.assertIn('No backtest results available', report)
    
    def test_compare_strategies(self):
        """Test strategy comparison"""
        # Create sample strategies
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        strategy1 = pd.Series(np.random.normal(1000000, 50000, len(dates)), index=dates)
        strategy2 = pd.Series(np.random.normal(1000000, 30000, len(dates)), index=dates)
        
        strategies = {
            'Strategy 1': strategy1,
            'Strategy 2': strategy2
        }
        
        comparison = self.backtester.compare_strategies(strategies)
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)  # Two strategies
        
        # Check that all required metrics are present
        required_metrics = [
            'Total Return (%)', 'Annualized Return (%)', 'Volatility (%)',
            'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, comparison.columns)
    
    def test_plot_results_no_results(self):
        """Test plotting with no results"""
        # Clear results
        self.backtester.results = {}
        
        # Should not raise an exception, just print a message
        try:
            self.backtester.plot_results()
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with single asset
        single_prices = self.sample_prices[['TLT']]
        single_weights = self.sample_weights[['TLT']]
        
        result = self.backtester.run_backtest(
            single_prices, 
            single_weights, 
            rebalance_frequency='M'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('portfolio_values', result)
        
        # Test with very short data
        short_prices = self.sample_prices.iloc[:3]
        short_weights = self.sample_weights.iloc[:3]
        
        result = self.backtester.run_backtest(
            short_prices, 
            short_weights, 
            rebalance_frequency='D'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('portfolio_values', result)
    
    def test_performance_metrics_edge_cases(self):
        """Test performance metrics with edge cases"""
        # Test with identical portfolio and benchmark
        portfolio = pd.Series([1000000, 1010000, 1020000])
        benchmark = pd.Series([1000000, 1010000, 1020000])
        
        metrics = self.backtester._calculate_performance_metrics(portfolio, benchmark)
        
        # Excess return should be zero
        self.assertAlmostEqual(metrics['excess_return'], 0, places=5)
        
        # Information ratio should be zero or NaN
        self.assertTrue(np.isnan(metrics['information_ratio']) or metrics['information_ratio'] == 0)
    
    def test_risk_metrics_edge_cases(self):
        """Test risk metrics with edge cases"""
        # Test with constant portfolio (no volatility)
        constant_portfolio = pd.Series([1000000, 1000000, 1000000])
        
        risk_metrics = self.backtester._calculate_risk_metrics(constant_portfolio)
        
        # Volatility should be zero
        self.assertAlmostEqual(risk_metrics['volatility'], 0, places=5)
        
        # Max drawdown should be zero
        self.assertAlmostEqual(risk_metrics['max_drawdown'], 0, places=5)


if __name__ == '__main__':
    unittest.main() 
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import cvxpy as cp

from src.core.config import Config
from src.optimization.optimization import PortfolioOptimizer, MLPortfolioOptimizer, RLPortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.optimizer = PortfolioOptimizer(self.config)
        
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.sample_returns = pd.DataFrame({
            'TLT': np.random.normal(0.001, 0.02, len(dates)),
            'IEF': np.random.normal(0.0005, 0.015, len(dates)),
            'LQD': np.random.normal(0.002, 0.025, len(dates)),
            'HYG': np.random.normal(0.003, 0.03, len(dates)),
            'BND': np.random.normal(0.001, 0.018, len(dates))
        }, index=dates)
        
        # Remove any infinite or NaN values
        self.sample_returns = self.sample_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    def test_initialization(self):
        """Test that PortfolioOptimizer is properly initialized"""
        self.assertIsNotNone(self.optimizer.config)
        self.assertIsNotNone(self.optimizer.scaler)
    
    def test_calculate_covariance_matrix_sample(self):
        """Test sample covariance matrix calculation"""
        cov_matrix = self.optimizer.calculate_covariance_matrix(
            self.sample_returns, method='sample', lookback=252
        )
        
        self.assertIsInstance(cov_matrix, pd.DataFrame)
        self.assertEqual(cov_matrix.shape, (len(self.sample_returns.columns), len(self.sample_returns.columns)))
        
        # Check symmetry
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))
        
        # Check positive definiteness (diagonal should be positive)
        self.assertTrue(all(cov_matrix.diagonal() > 0))
    
    def test_calculate_covariance_matrix_exponential(self):
        """Test exponential covariance matrix calculation"""
        cov_matrix = self.optimizer.calculate_covariance_matrix(
            self.sample_returns, method='exponential', lookback=252
        )
        
        self.assertIsInstance(cov_matrix, pd.DataFrame)
        self.assertEqual(cov_matrix.shape, (len(self.sample_returns.columns), len(self.sample_returns.columns)))
        
        # Check symmetry
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))
    
    def test_calculate_covariance_matrix_robust(self):
        """Test robust covariance matrix calculation"""
        cov_matrix = self.optimizer.calculate_covariance_matrix(
            self.sample_returns, method='robust', lookback=252
        )
        
        self.assertIsInstance(cov_matrix, pd.DataFrame)
        self.assertEqual(cov_matrix.shape, (len(self.sample_returns.columns), len(self.sample_returns.columns)))
        
        # Check symmetry
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))
    
    def test_mean_variance_optimization(self):
        """Test mean-variance optimization"""
        result = self.optimizer.mean_variance_optimization(self.sample_returns)
        
        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn('weights', result)
            self.assertIn('expected_return', result)
            self.assertIn('volatility', result)
            self.assertIn('sharpe_ratio', result)
            
            # Check that weights sum to 1
            self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
            
            # Check that all weights are non-negative
            self.assertTrue(all(result['weights'] >= 0))
            
            # Check that volatility is positive
            self.assertGreater(result['volatility'], 0)
    
    def test_mean_variance_optimization_target_return(self):
        """Test mean-variance optimization with target return"""
        target_return = 0.05  # 5% annual return
        result = self.optimizer.mean_variance_optimization(
            self.sample_returns, target_return=target_return
        )
        
        if result is not None:
            self.assertGreaterEqual(result['expected_return'], target_return)
            self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
    
    def test_mean_variance_optimization_target_volatility(self):
        """Test mean-variance optimization with target volatility"""
        target_volatility = 0.15  # 15% annual volatility
        result = self.optimizer.mean_variance_optimization(
            self.sample_returns, target_volatility=target_volatility
        )
        
        if result is not None:
            self.assertLessEqual(result['volatility'], target_volatility)
            self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
    
    def test_sector_constrained_optimization(self):
        """Test sector-constrained optimization"""
        sector_mapping = {
            'TLT': 'government',
            'IEF': 'government',
            'LQD': 'corporate',
            'HYG': 'high_yield',
            'BND': 'government'
        }
        
        sector_constraints = {
            'government': 0.6,
            'corporate': 0.3,
            'high_yield': 0.2
        }
        
        result = self.optimizer.sector_constrained_optimization(
            self.sample_returns, sector_mapping, sector_constraints
        )
        
        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn('weights', result)
            self.assertIn('expected_return', result)
            self.assertIn('volatility', result)
            self.assertIn('sharpe_ratio', result)
            
            # Check that weights sum to 1
            self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
            
            # Check sector constraints
            gov_weight = result['weights'][['TLT', 'IEF', 'BND']].sum()
            corp_weight = result['weights'][['LQD']].sum()
            hy_weight = result['weights'][['HYG']].sum()
            
            self.assertLessEqual(gov_weight, 0.6)
            self.assertLessEqual(corp_weight, 0.3)
            self.assertLessEqual(hy_weight, 0.2)
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization"""
        result = self.optimizer.risk_parity_optimization(self.sample_returns)
        
        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn('weights', result)
            self.assertIn('risk_contributions', result)
            
            # Check that weights sum to 1
            self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
            
            # Check that all weights are non-negative
            self.assertTrue(all(result['weights'] >= 0))
            
            # Check that risk contributions are equal (approximately)
            risk_contribs = result['risk_contributions']
            if len(risk_contribs) > 1:
                mean_contrib = np.mean(risk_contribs)
                for contrib in risk_contribs:
                    self.assertAlmostEqual(contrib, mean_contrib, places=3)
    
    def test_black_litterman_optimization(self):
        """Test Black-Litterman optimization"""
        market_caps = pd.Series({
            'TLT': 1000000,
            'IEF': 800000,
            'LQD': 600000,
            'HYG': 400000,
            'BND': 1200000
        })
        
        views = {
            'TLT': 0.08,  # 8% expected return for TLT
            'HYG': 0.12   # 12% expected return for HYG
        }
        
        confidence = {
            'TLT': 0.1,
            'HYG': 0.15
        }
        
        result = self.optimizer.black_litterman_optimization(
            self.sample_returns, market_caps, views, confidence
        )
        
        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn('weights', result)
            self.assertIn('expected_return', result)
            self.assertIn('volatility', result)
            
            # Check that weights sum to 1
            self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
    
    def test_empty_returns_handling(self):
        """Test handling of empty returns data"""
        empty_returns = pd.DataFrame()
        result = self.optimizer.mean_variance_optimization(empty_returns)
        
        self.assertIsNone(result)
    
    def test_single_asset_returns(self):
        """Test optimization with single asset"""
        single_returns = self.sample_returns[['TLT']]
        result = self.optimizer.mean_variance_optimization(single_returns)
        
        if result is not None:
            self.assertEqual(len(result['weights']), 1)
            self.assertAlmostEqual(result['weights'].iloc[0], 1.0, places=5)


class TestMLPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.ml_optimizer = MLPortfolioOptimizer(self.config)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.sample_returns = pd.DataFrame({
            'TLT': np.random.normal(0.001, 0.02, len(dates)),
            'IEF': np.random.normal(0.0005, 0.015, len(dates)),
            'LQD': np.random.normal(0.002, 0.025, len(dates))
        }, index=dates).dropna()
        
        self.sample_market_factors = pd.DataFrame({
            'equity_market': np.random.normal(0.002, 0.025, len(dates)),
            'credit_spread': np.random.normal(0.001, 0.015, len(dates))
        }, index=dates).dropna()
        
        self.sample_sentiment = pd.DataFrame({
            'TLT': np.random.normal(0.1, 0.3, len(dates)),
            'IEF': np.random.normal(0.05, 0.2, len(dates)),
            'LQD': np.random.normal(0.15, 0.4, len(dates))
        }, index=dates).dropna()
    
    def test_initialization(self):
        """Test that MLPortfolioOptimizer is properly initialized"""
        self.assertIsNotNone(self.ml_optimizer.config)
        self.assertIsNotNone(self.ml_optimizer.scaler)
        self.assertIsNotNone(self.ml_optimizer.rf_model)
        self.assertIsNotNone(self.ml_optimizer.nn_model)
    
    def test_prepare_features(self):
        """Test feature preparation"""
        features = self.ml_optimizer.prepare_features(
            self.sample_returns, self.sample_market_factors, self.sample_sentiment
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        
        # Check that features are created for each ticker
        for ticker in self.sample_returns.columns:
            self.assertIn(f'{ticker}_volatility', features.columns)
            self.assertIn(f'{ticker}_momentum', features.columns)
            self.assertIn(f'{ticker}_skewness', features.columns)
            self.assertIn(f'{ticker}_kurtosis', features.columns)
    
    def test_train_return_predictor(self):
        """Test return predictor training"""
        features = self.ml_optimizer.prepare_features(
            self.sample_returns, self.sample_market_factors, self.sample_sentiment
        )
        
        # Should not raise an exception
        try:
            self.ml_optimizer.train_return_predictor(features, self.sample_returns, lookback=252)
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")
    
    def test_predict_returns(self):
        """Test return prediction"""
        features = self.ml_optimizer.prepare_features(
            self.sample_returns, self.sample_market_factors, self.sample_sentiment
        )
        
        # Train the models first
        self.ml_optimizer.train_return_predictor(features, self.sample_returns, lookback=252)
        
        # Make predictions
        predictions = self.ml_optimizer.predict_returns(features)
        
        self.assertIsInstance(predictions, pd.Series)
        self.assertIn('rf_prediction', predictions.index)
        self.assertIn('nn_prediction', predictions.index)
        
        # Predictions should be finite numbers
        self.assertTrue(np.isfinite(predictions['rf_prediction']))
        self.assertTrue(np.isfinite(predictions['nn_prediction']))


class TestRLPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.state_dim = 10
        self.action_dim = 3
        self.rl_optimizer = RLPortfolioOptimizer(self.config, self.state_dim, self.action_dim)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.sample_returns = pd.DataFrame({
            'TLT': np.random.normal(0.001, 0.02, len(dates)),
            'IEF': np.random.normal(0.0005, 0.015, len(dates)),
            'LQD': np.random.normal(0.002, 0.025, len(dates))
        }, index=dates).dropna()
        
        self.sample_market_factors = pd.DataFrame({
            'equity_market': np.random.normal(0.002, 0.025, len(dates)),
            'credit_spread': np.random.normal(0.001, 0.015, len(dates))
        }, index=dates).dropna()
        
        self.sample_sentiment = pd.DataFrame({
            'TLT': np.random.normal(0.1, 0.3, len(dates)),
            'IEF': np.random.normal(0.05, 0.2, len(dates)),
            'LQD': np.random.normal(0.15, 0.4, len(dates))
        }, index=dates).dropna()
    
    def test_initialization(self):
        """Test that RLPortfolioOptimizer is properly initialized"""
        self.assertIsNotNone(self.rl_optimizer.config)
        self.assertEqual(self.rl_optimizer.state_dim, self.state_dim)
        self.assertEqual(self.rl_optimizer.action_dim, self.action_dim)
        self.assertIsNotNone(self.rl_optimizer.actor)
        self.assertIsNotNone(self.rl_optimizer.critic)
    
    def test_get_state(self):
        """Test state representation"""
        state = self.rl_optimizer.get_state(
            self.sample_returns, self.sample_market_factors, self.sample_sentiment
        )
        
        self.assertIsInstance(state, type(self.rl_optimizer.actor.fc1.weight))
        self.assertEqual(state.shape[0], self.state_dim)
    
    def test_get_action(self):
        """Test action selection"""
        state = self.rl_optimizer.get_state(
            self.sample_returns, self.sample_market_factors, self.sample_sentiment
        )
        
        action = self.rl_optimizer.get_action(state)
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_update(self):
        """Test model update"""
        states = []
        actions = []
        rewards = []
        
        # Create sample training data
        for i in range(10):
            state = self.rl_optimizer.get_state(
                self.sample_returns.iloc[i:i+20], 
                self.sample_market_factors.iloc[i:i+20] if not self.sample_market_factors.empty else pd.DataFrame(),
                self.sample_sentiment.iloc[i:i+1] if not self.sample_sentiment.empty else pd.DataFrame()
            )
            states.append(state)
            actions.append(np.random.randint(0, self.action_dim))
            rewards.append(np.random.normal(0, 0.01))
        
        # Should not raise an exception
        try:
            self.rl_optimizer.update(states, actions, rewards)
        except Exception as e:
            self.fail(f"Update failed with exception: {e}")


if __name__ == '__main__':
    unittest.main() 
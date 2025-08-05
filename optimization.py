import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def calculate_covariance_matrix(self, returns: pd.DataFrame, method: str = 'sample', 
                                  lookback: int = 252) -> pd.DataFrame:
        if method == 'sample':
            return returns.rolling(lookback).cov().iloc[-1].unstack()
        elif method == 'exponential':
            lambda_param = 0.94
            weights = np.array([(1-lambda_param) * lambda_param**i for i in range(lookback)])
            weights = weights / weights.sum()
            
            weighted_returns = returns.rolling(lookback).apply(
                lambda x: np.average(x, weights=weights[-len(x):])
            )
            return weighted_returns.cov()
        elif method == 'robust':
            from sklearn.covariance import MinCovDet
            robust_cov = MinCovDet().fit(returns.dropna())
            cov_matrix = pd.DataFrame(robust_cov.covariance_, 
                                    index=returns.columns, 
                                    columns=returns.columns)
            return cov_matrix
        else:
            return returns.cov()
    
    def mean_variance_optimization(self, returns: pd.DataFrame, 
                                 target_return: Optional[float] = None,
                                 target_volatility: Optional[float] = None) -> Dict:
        mu = returns.mean() * 252
        sigma = self.calculate_covariance_matrix(returns)
        
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        if target_return is not None:
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                mu @ weights >= target_return
            ]
            objective = cp.Minimize(cp.quad_form(weights, sigma))
        elif target_volatility is not None:
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                cp.quad_form(weights, sigma) <= target_volatility**2
            ]
            objective = cp.Maximize(mu @ weights)
        else:
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0
            ]
            objective = cp.Maximize(mu @ weights - 0.5 * cp.quad_form(weights, sigma))
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            return {
                'weights': pd.Series(weights.value, index=returns.columns),
                'expected_return': mu @ weights.value,
                'volatility': np.sqrt(weights.value @ sigma @ weights.value),
                'sharpe_ratio': (mu @ weights.value - self.config.RISK_FREE_RATE) / 
                               np.sqrt(weights.value @ sigma @ weights.value)
            }
        else:
            return None
    
    def sector_constrained_optimization(self, returns: pd.DataFrame, 
                                      sector_mapping: Dict[str, str],
                                      sector_constraints: Dict[str, float]) -> Dict:
        mu = returns.mean() * 252
        sigma = self.calculate_covariance_matrix(returns)
        
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        constraints = [cp.sum(weights) == 1, weights >= 0]
        
        for sector, max_weight in sector_constraints.items():
            sector_assets = [i for i, asset in enumerate(returns.columns) 
                           if sector_mapping.get(asset) == sector]
            if sector_assets:
                constraints.append(cp.sum(weights[sector_assets]) <= max_weight)
        
        objective = cp.Maximize(mu @ weights - 0.5 * cp.quad_form(weights, sigma))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            return {
                'weights': pd.Series(weights.value, index=returns.columns),
                'expected_return': mu @ weights.value,
                'volatility': np.sqrt(weights.value @ sigma @ weights.value),
                'sharpe_ratio': (mu @ weights.value - self.config.RISK_FREE_RATE) / 
                               np.sqrt(weights.value @ sigma @ weights.value)
            }
        else:
            return None
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict:
        sigma = self.calculate_covariance_matrix(returns)
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        risk_contributions = []
        for i in range(n_assets):
            risk_contrib = cp.quad_form(weights, sigma[i, :])
            risk_contributions.append(risk_contrib)
        
        constraints = [cp.sum(weights) == 1, weights >= 0]
        
        for i in range(1, n_assets):
            constraints.append(risk_contributions[i] == risk_contributions[0])
        
        objective = cp.Minimize(cp.sum_squares(risk_contributions))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            return {
                'weights': pd.Series(weights.value, index=returns.columns),
                'risk_contributions': [rc.value for rc in risk_contributions]
            }
        else:
            return None
    
    def black_litterman_optimization(self, returns: pd.DataFrame, 
                                   market_caps: pd.Series,
                                   views: Dict[str, float],
                                   confidence: Dict[str, float]) -> Dict:
        sigma = self.calculate_covariance_matrix(returns)
        pi = returns.mean() * 252
        
        market_weights = market_caps / market_caps.sum()
        tau = 0.05
        
        sigma_market = market_weights @ sigma @ market_weights
        pi_bl = pi + tau * sigma @ market_weights
        
        P = np.zeros((len(views), len(returns.columns)))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))
        
        for i, (asset, view) in enumerate(views.items()):
            P[i, returns.columns.get_loc(asset)] = 1
            Q[i] = view
            Omega[i, i] = confidence.get(asset, 0.1)
        
        M1 = np.linalg.inv(tau * sigma)
        M2 = P.T @ np.linalg.inv(Omega) @ P
        M3 = np.linalg.inv(tau * sigma) @ pi_bl
        M4 = P.T @ np.linalg.inv(Omega) @ Q
        
        mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
        sigma_bl = np.linalg.inv(M1 + M2)
        
        return self.mean_variance_optimization_with_views(returns, mu_bl, sigma_bl)
    
    def mean_variance_optimization_with_views(self, returns: pd.DataFrame, 
                                            mu: np.ndarray, sigma: np.ndarray) -> Dict:
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        constraints = [cp.sum(weights) == 1, weights >= 0]
        objective = cp.Maximize(mu @ weights - 0.5 * cp.quad_form(weights, sigma))
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            return {
                'weights': pd.Series(weights.value, index=returns.columns),
                'expected_return': mu @ weights.value,
                'volatility': np.sqrt(weights.value @ sigma @ weights.value)
            }
        else:
            return None

class MLPortfolioOptimizer:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        
    def prepare_features(self, returns: pd.DataFrame, 
                        market_factors: pd.DataFrame,
                        sentiment_features: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        
        for ticker in returns.columns:
            ticker_features = {}
            
            ticker_returns = returns[ticker]
            ticker_features[f'{ticker}_volatility'] = ticker_returns.rolling(20).std()
            ticker_features[f'{ticker}_momentum'] = ticker_returns.rolling(60).mean()
            ticker_features[f'{ticker}_skewness'] = ticker_returns.rolling(60).skew()
            ticker_features[f'{ticker}_kurtosis'] = ticker_returns.rolling(60).kurt()
            
            for factor in market_factors.columns:
                ticker_features[f'{ticker}_{factor}_beta'] = (
                    ticker_returns.rolling(60).cov(market_factors[factor]) / 
                    market_factors[factor].rolling(60).var()
                )
            
            if not sentiment_features.empty and ticker in sentiment_features.columns:
                ticker_features[f'{ticker}_sentiment'] = sentiment_features[ticker]
            
            features = pd.concat([features, pd.DataFrame(ticker_features)], axis=1)
        
        return features.dropna()
    
    def train_return_predictor(self, features: pd.DataFrame, 
                              returns: pd.DataFrame, 
                              lookback: int = 252) -> None:
        X = features.iloc[:-1]
        y = returns.iloc[1:].mean(axis=1)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.rf_model.fit(X_scaled, y)
        self.nn_model.fit(X_scaled, y)
    
    def predict_returns(self, features: pd.DataFrame) -> pd.Series:
        X_scaled = self.scaler.transform(features.iloc[-1:])
        
        rf_pred = self.rf_model.predict(X_scaled)[0]
        nn_pred = self.nn_model.predict(X_scaled)[0]
        
        return pd.Series([rf_pred, nn_pred], index=['rf_prediction', 'nn_prediction'])

class RLPortfolioOptimizer:
    def __init__(self, config, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
    def get_state(self, returns: pd.DataFrame, 
                  market_factors: pd.DataFrame,
                  sentiment_features: pd.DataFrame) -> torch.Tensor:
        state_features = []
        
        returns_features = returns.iloc[-20:].values.flatten()
        state_features.extend(returns_features)
        
        if not market_factors.empty:
            market_features = market_factors.iloc[-20:].values.flatten()
            state_features.extend(market_features)
        
        if not sentiment_features.empty:
            sentiment_features_flat = sentiment_features.iloc[-1].values
            state_features.extend(sentiment_features_flat)
        
        return torch.FloatTensor(state_features).to(self.device)
    
    def get_action(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action_probs = self.actor(state)
            action = torch.multinomial(action_probs, 1).item()
            return action
    
    def update(self, states: List[torch.Tensor], 
               actions: List[int], 
               rewards: List[float]) -> None:
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        action_probs = self.actor(states)
        values = self.critic(states)
        
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        advantages = rewards - values.detach()
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((rewards - values) ** 2).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
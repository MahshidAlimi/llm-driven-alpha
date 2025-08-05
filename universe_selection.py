import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FixedIncomeUniverse:
    def __init__(self, config):
        self.config = config
        self.universe = config.FIXED_INCOME_UNIVERSE
        self.sector_mapping = {
            'TLT': 'government', 'IEF': 'government', 'SHY': 'government',
            'LQD': 'corporate', 'VCIT': 'corporate', 'VCSH': 'corporate',
            'HYG': 'high_yield', 'JNK': 'high_yield', 'SJNK': 'high_yield',
            'BND': 'government', 'AGG': 'government', 'BNDX': 'government',
            'EMB': 'emerging_markets', 'PCY': 'emerging_markets', 'VWOB': 'emerging_markets',
            'IGOV': 'government', 'BWX': 'emerging_markets', 'EMLC': 'emerging_markets'
        }
    
    def get_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        data = {}
        for ticker in self.universe:
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not ticker_data.empty:
                    data[ticker] = ticker_data['Adj Close']
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        
        return pd.DataFrame(data).dropna()
    
    def calculate_metrics(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().dropna()
        
        metrics = {}
        for ticker in returns.columns:
            ticker_returns = returns[ticker].dropna()
            
            metrics[ticker] = {
                'volatility': ticker_returns.std() * np.sqrt(252),
                'sharpe_ratio': (ticker_returns.mean() * 252 - self.config.RISK_FREE_RATE) / (ticker_returns.std() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(prices[ticker]),
                'avg_volume': self._get_avg_volume(ticker),
                'sector': self.sector_mapping.get(ticker, 'unknown')
            }
        
        return pd.DataFrame(metrics).T
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _get_avg_volume(self, ticker: str) -> float:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return info.get('averageVolume', 0)
        except:
            return 0
    
    def filter_universe(self, prices: pd.DataFrame, min_volatility: float = 0.05, 
                       max_volatility: float = 0.25, min_sharpe: float = -0.5) -> List[str]:
        metrics = self.calculate_metrics(prices)
        
        filtered = metrics[
            (metrics['volatility'] >= min_volatility) &
            (metrics['volatility'] <= max_volatility) &
            (metrics['sharpe_ratio'] >= min_sharpe) &
            (metrics['avg_volume'] > 0)
        ]
        
        return filtered.index.tolist()
    
    def get_sector_weights(self, selected_tickers: List[str]) -> Dict[str, float]:
        sector_counts = {}
        for ticker in selected_tickers:
            sector = self.sector_mapping.get(ticker, 'unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        total = len(selected_tickers)
        return {sector: count/total for sector, count in sector_counts.items()} 
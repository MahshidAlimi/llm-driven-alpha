import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def run_backtest(self, prices: pd.DataFrame, weights_history: pd.DataFrame,
                    rebalance_frequency: str = 'M', transaction_costs: float = 0.001) -> Dict:
        
        portfolio_values = []
        weights_list = []
        dates = []
        
        current_weights = weights_history.iloc[0]
        portfolio_value = 1000000
        
        for i in range(1, len(prices)):
            current_date = prices.index[i]
            previous_date = prices.index[i-1]
            
            if i == 1 or self._should_rebalance(current_date, previous_date, rebalance_frequency):
                current_weights = weights_history.loc[current_date] if current_date in weights_history.index else current_weights
            
            returns = prices.loc[current_date] / prices.loc[previous_date] - 1
            portfolio_return = (current_weights * returns).sum()
            
            if i > 1 and self._should_rebalance(current_date, previous_date, rebalance_frequency):
                portfolio_return -= transaction_costs
            
            portfolio_value *= (1 + portfolio_return)
            
            portfolio_values.append(portfolio_value)
            weights_list.append(current_weights.copy())
            dates.append(current_date)
        
        portfolio_series = pd.Series(portfolio_values, index=dates)
        weights_df = pd.DataFrame(weights_list, index=dates)
        
        benchmark = self._calculate_benchmark(prices)
        
        self.results = {
            'portfolio_values': portfolio_series,
            'weights_history': weights_df,
            'benchmark': benchmark,
            'performance_metrics': self._calculate_performance_metrics(portfolio_series, benchmark),
            'risk_metrics': self._calculate_risk_metrics(portfolio_series),
            'drawdown_analysis': self._calculate_drawdown(portfolio_series)
        }
        
        return self.results
    
    def _should_rebalance(self, current_date, previous_date, frequency):
        if frequency == 'D':
            return True
        elif frequency == 'W':
            return current_date.week != previous_date.week
        elif frequency == 'M':
            return current_date.month != previous_date.month
        elif frequency == 'Q':
            return current_date.quarter != previous_date.quarter
        elif frequency == 'Y':
            return current_date.year != previous_date.year
        return False
    
    def _calculate_benchmark(self, prices: pd.DataFrame) -> pd.Series:
        benchmark_weights = pd.Series(1/len(prices.columns), index=prices.columns)
        benchmark_values = []
        
        for i in range(1, len(prices)):
            current_date = prices.index[i]
            previous_date = prices.index[i-1]
            
            returns = prices.loc[current_date] / prices.loc[previous_date] - 1
            benchmark_return = (benchmark_weights * returns).sum()
            
            if i == 1:
                benchmark_value = 1000000
            else:
                benchmark_value = benchmark_values[-1] * (1 + benchmark_return)
            
            benchmark_values.append(benchmark_value)
        
        return pd.Series(benchmark_values, index=prices.index[1:])
    
    def _calculate_performance_metrics(self, portfolio: pd.Series, benchmark: pd.Series) -> Dict:
        portfolio_returns = portfolio.pct_change().dropna()
        benchmark_returns = benchmark.pct_change().dropna()
        
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        excess_returns = portfolio_returns - benchmark_returns
        
        metrics = {
            'total_return': (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100,
            'annualized_return': (portfolio.iloc[-1] / portfolio.iloc[0]) ** (252/len(portfolio)) - 1,
            'benchmark_return': (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100,
            'excess_return': (portfolio.iloc[-1] / portfolio.iloc[0] - benchmark.iloc[-1] / benchmark.iloc[0]) * 100,
            'sharpe_ratio': (portfolio_returns.mean() * 252 - self.config.RISK_FREE_RATE) / (portfolio_returns.std() * np.sqrt(252)),
            'information_ratio': excess_returns.mean() * 252 / (excess_returns.std() * np.sqrt(252)),
            'calmar_ratio': (portfolio_returns.mean() * 252) / abs(self._calculate_max_drawdown(portfolio)),
            'sortino_ratio': (portfolio_returns.mean() * 252 - self.config.RISK_FREE_RATE) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)),
            'win_rate': (portfolio_returns > 0).mean(),
            'profit_factor': abs(portfolio_returns[portfolio_returns > 0].sum() / portfolio_returns[portfolio_returns < 0].sum()) if portfolio_returns[portfolio_returns < 0].sum() != 0 else float('inf')
        }
        
        return metrics
    
    def _calculate_risk_metrics(self, portfolio: pd.Series) -> Dict:
        returns = portfolio.pct_change().dropna()
        
        risk_metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'var_99': returns.quantile(0.01),
            'cvar_99': returns[returns <= returns.quantile(0.01)].mean(),
            'max_drawdown': self._calculate_max_drawdown(portfolio),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252)
        }
        
        return risk_metrics
    
    def _calculate_drawdown(self, portfolio: pd.Series) -> Dict:
        cumulative = portfolio / portfolio.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        recovery_date = drawdown[drawdown.index > max_drawdown_date][drawdown[drawdown.index > max_drawdown_date] >= 0].index[0] if len(drawdown[drawdown.index > max_drawdown_date][drawdown[drawdown.index > max_drawdown_date] >= 0]) > 0 else None
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'recovery_date': recovery_date,
            'drawdown_series': drawdown
        }
    
    def _calculate_max_drawdown(self, portfolio: pd.Series) -> float:
        cumulative = portfolio / portfolio.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        if not self.results:
            print("No backtest results available. Run backtest first.")
            return
        
        portfolio = self.results['portfolio_values']
        benchmark = self.results['benchmark']
        weights = self.results['weights_history']
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Performance', 'Weight Allocation', 
                          'Drawdown Analysis', 'Rolling Sharpe Ratio',
                          'Monthly Returns Heatmap', 'Risk-Return Scatter'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=portfolio.index, y=portfolio.values, name='Portfolio', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=benchmark.index, y=benchmark.values, name='Benchmark', line=dict(color='red')),
            row=1, col=1
        )
        
        for asset in weights.columns:
            fig.add_trace(
                go.Scatter(x=weights.index, y=weights[asset], name=asset, stackgroup='weights'),
                row=1, col=2
            )
        
        drawdown = self.results['drawdown_analysis']['drawdown_series']
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, fill='tonexty', name='Drawdown', line=dict(color='red')),
            row=2, col=1
        )
        
        rolling_sharpe = portfolio.pct_change().rolling(252).apply(
            lambda x: (x.mean() * 252 - self.config.RISK_FREE_RATE) / (x.std() * np.sqrt(252))
        )
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Rolling Sharpe', line=dict(color='green')),
            row=2, col=2
        )
        
        monthly_returns = portfolio.pct_change().resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_matrix = monthly_returns.values.reshape(-1, 12)
        
        fig.add_trace(
            go.Heatmap(z=monthly_returns_matrix, colorscale='RdYlGn', name='Monthly Returns'),
            row=3, col=1
        )
        
        fig.update_layout(height=1200, title_text="Backtest Results")
        fig.show()
        
        if save_path:
            fig.write_html(save_path)
    
    def generate_report(self) -> str:
        if not self.results:
            return "No backtest results available."
        
        performance = self.results['performance_metrics']
        risk = self.results['risk_metrics']
        
        report = f"""
        FIXED INCOME PORTFOLIO BACKTEST REPORT
        ======================================
        
        PERFORMANCE METRICS:
        Total Return: {performance['total_return']:.2f}%
        Annualized Return: {performance['annualized_return']*100:.2f}%
        Benchmark Return: {performance['benchmark_return']:.2f}%
        Excess Return: {performance['excess_return']:.2f}%
        
        RISK-ADJUSTED METRICS:
        Sharpe Ratio: {performance['sharpe_ratio']:.3f}
        Information Ratio: {performance['information_ratio']:.3f}
        Sortino Ratio: {performance['sortino_ratio']:.3f}
        Calmar Ratio: {performance['calmar_ratio']:.3f}
        
        RISK METRICS:
        Volatility: {risk['volatility']*100:.2f}%
        VaR (95%): {risk['var_95']*100:.2f}%
        CVaR (95%): {risk['cvar_95']*100:.2f}%
        Maximum Drawdown: {risk['max_drawdown']*100:.2f}%
        
        TRADING METRICS:
        Win Rate: {performance['win_rate']*100:.1f}%
        Profit Factor: {performance['profit_factor']:.2f}
        Skewness: {risk['skewness']:.3f}
        Kurtosis: {risk['kurtosis']:.3f}
        """
        
        return report
    
    def compare_strategies(self, strategies: Dict[str, pd.Series]) -> pd.DataFrame:
        comparison = {}
        
        for strategy_name, portfolio_values in strategies.items():
            returns = portfolio_values.pct_change().dropna()
            
            comparison[strategy_name] = {
                'Total Return (%)': (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100,
                'Annualized Return (%)': ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252/len(portfolio_values)) - 1) * 100,
                'Volatility (%)': returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (returns.mean() * 252 - self.config.RISK_FREE_RATE) / (returns.std() * np.sqrt(252)),
                'Max Drawdown (%)': self._calculate_max_drawdown(portfolio_values) * 100,
                'Win Rate (%)': (returns > 0).mean() * 100
            }
        
        return pd.DataFrame(comparison).T 
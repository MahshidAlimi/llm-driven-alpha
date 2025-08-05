import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import Config
from universe_selection import FixedIncomeUniverse
from factor_analysis import FactorAnalysis
from optimization import PortfolioOptimizer, MLPortfolioOptimizer, RLPortfolioOptimizer
from backtest import Backtester

class FixedIncomeTradingSystem:
    def __init__(self):
        self.config = Config()
        self.universe = FixedIncomeUniverse(self.config)
        self.factor_analysis = FactorAnalysis(self.config)
        self.optimizer = PortfolioOptimizer(self.config)
        self.ml_optimizer = MLPortfolioOptimizer(self.config)
        self.backtester = Backtester(self.config)
        
    def run_complete_analysis(self):
        print("Starting Fixed Income Trading System Analysis...")
        
        start_date = self.config.BACKTEST_START_DATE
        end_date = self.config.BACKTEST_END_DATE
        
        print(f"1. Selecting Universe ({start_date} to {end_date})")
        prices = self.universe.get_historical_data(start_date, end_date)
        selected_tickers = self.universe.filter_universe(prices)
        print(f"Selected {len(selected_tickers)} securities: {selected_tickers}")
        
        filtered_prices = prices[selected_tickers]
        
        print("2. Analyzing Market Factors and Sentiment")
        market_factors = self.factor_analysis.get_market_factors(start_date, end_date)
        rolling_features = self.factor_analysis.calculate_rolling_returns(filtered_prices)
        
        sentiment_features = pd.DataFrame()
        for ticker in selected_tickers[:5]:
            sentiment = self.factor_analysis.aggregate_sentiment_features(ticker)
            sentiment_features[ticker] = pd.Series(sentiment)
        
        print("3. Running Portfolio Optimization")
        returns = filtered_prices.pct_change().dropna()
        
        optimization_results = {}
        
        mvo_result = self.optimizer.mean_variance_optimization(returns)
        if mvo_result:
            optimization_results['Mean Variance'] = mvo_result
        
        sector_result = self.optimizer.sector_constrained_optimization(
            returns, 
            self.universe.sector_mapping, 
            self.config.SECTOR_CONSTRAINTS
        )
        if sector_result:
            optimization_results['Sector Constrained'] = sector_result
        
        risk_parity_result = self.optimizer.risk_parity_optimization(returns)
        if risk_parity_result:
            optimization_results['Risk Parity'] = risk_parity_result
        
        print("4. Running Backtests")
        backtest_results = {}
        
        for strategy_name, result in optimization_results.items():
            weights = result['weights']
            weights_history = pd.DataFrame([weights] * len(filtered_prices), 
                                         index=filtered_prices.index)
            
            backtest_result = self.backtester.run_backtest(
                filtered_prices, 
                weights_history, 
                self.config.REBALANCE_FREQUENCY
            )
            backtest_results[strategy_name] = backtest_result
        
        print("5. Generating Reports and Visualizations")
        self._print_summary(optimization_results, backtest_results)
        self._plot_comparison(backtest_results)
        
        return optimization_results, backtest_results
    
    def run_ml_enhanced_strategy(self):
        print("Running ML-Enhanced Strategy...")
        
        start_date = self.config.BACKTEST_START_DATE
        end_date = self.config.BACKTEST_END_DATE
        
        prices = self.universe.get_historical_data(start_date, end_date)
        selected_tickers = self.universe.filter_universe(prices)
        filtered_prices = prices[selected_tickers]
        
        market_factors = self.factor_analysis.get_market_factors(start_date, end_date)
        sentiment_features = pd.DataFrame()
        
        for ticker in selected_tickers[:5]:
            sentiment = self.factor_analysis.aggregate_sentiment_features(ticker)
            sentiment_features[ticker] = pd.Series(sentiment)
        
        features = self.ml_optimizer.prepare_features(
            filtered_prices.pct_change().dropna(),
            market_factors,
            sentiment_features
        )
        
        returns = filtered_prices.pct_change().dropna()
        
        train_end = len(features) // 2
        self.ml_optimizer.train_return_predictor(
            features.iloc[:train_end],
            returns.iloc[:train_end]
        )
        
        predictions = self.ml_optimizer.predict_returns(features.iloc[train_end:])
        
        print(f"ML Predictions: {predictions}")
        
        return predictions
    
    def run_rl_strategy(self):
        print("Running RL Strategy...")
        
        start_date = self.config.BACKTEST_START_DATE
        end_date = self.config.BACKTEST_END_DATE
        
        prices = self.universe.get_historical_data(start_date, end_date)
        selected_tickers = self.universe.filter_universe(prices)
        filtered_prices = prices[selected_tickers]
        
        market_factors = self.factor_analysis.get_market_factors(start_date, end_date)
        sentiment_features = pd.DataFrame()
        
        state_dim = 20 * len(selected_tickers) + 20 * len(market_factors.columns) + len(sentiment_features.columns)
        action_dim = len(selected_tickers)
        
        rl_optimizer = RLPortfolioOptimizer(self.config, state_dim, action_dim)
        
        returns = filtered_prices.pct_change().dropna()
        
        states = []
        actions = []
        rewards = []
        
        for i in range(20, len(returns) - 1):
            state = rl_optimizer.get_state(
                returns.iloc[i-20:i],
                market_factors.iloc[i-20:i] if not market_factors.empty else pd.DataFrame(),
                sentiment_features.iloc[i:i+1] if not sentiment_features.empty else pd.DataFrame()
            )
            
            action = rl_optimizer.get_action(state)
            
            portfolio_return = returns.iloc[i+1].iloc[action]
            reward = portfolio_return
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if len(states) >= 100:
                rl_optimizer.update(states, actions, rewards)
                states, actions, rewards = [], [], []
        
        print("RL Strategy training completed")
        return rl_optimizer
    
    def _print_summary(self, optimization_results, backtest_results):
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        for strategy_name, result in optimization_results.items():
            print(f"\n{strategy_name}:")
            print(f"  Expected Return: {result['expected_return']*100:.2f}%")
            print(f"  Volatility: {result['volatility']*100:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"  Top Holdings:")
            top_holdings = result['weights'].nlargest(5)
            for asset, weight in top_holdings.items():
                print(f"    {asset}: {weight*100:.1f}%")
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        for strategy_name, result in backtest_results.items():
            performance = result['performance_metrics']
            risk = result['risk_metrics']
            
            print(f"\n{strategy_name}:")
            print(f"  Total Return: {performance['total_return']:.2f}%")
            print(f"  Annualized Return: {performance['annualized_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {risk['max_drawdown']*100:.2f}%")
            print(f"  Volatility: {risk['volatility']*100:.2f}%")
    
    def _plot_comparison(self, backtest_results):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Performance Comparison', 'Risk-Return Scatter',
                          'Drawdown Comparison', 'Rolling Sharpe Comparison')
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (strategy_name, result) in enumerate(backtest_results.items()):
            portfolio = result['portfolio_values']
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(x=portfolio.index, y=portfolio.values, 
                          name=strategy_name, line=dict(color=color)),
                row=1, col=1
            )
            
            performance = result['performance_metrics']
            risk = result['risk_metrics']
            
            fig.add_trace(
                go.Scatter(x=[risk['volatility']*100], y=[performance['annualized_return']*100],
                          mode='markers', name=f"{strategy_name} Risk-Return",
                          marker=dict(color=color, size=10)),
                row=1, col=2
            )
            
            drawdown = result['drawdown_analysis']['drawdown_series']
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values*100,
                          name=f"{strategy_name} Drawdown", line=dict(color=color)),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Strategy Comparison")
        fig.show()

def main():
    system = FixedIncomeTradingSystem()
    
    print("Fixed Income Trading System")
    print("="*50)
    
    optimization_results, backtest_results = system.run_complete_analysis()
    
    print("\n" + "="*50)
    print("Running ML-Enhanced Strategy...")
    ml_predictions = system.run_ml_enhanced_strategy()
    
    print("\n" + "="*50)
    print("Running RL Strategy...")
    rl_optimizer = system.run_rl_strategy()
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    
    return optimization_results, backtest_results, ml_predictions, rl_optimizer

if __name__ == "__main__":
    main() 
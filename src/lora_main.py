#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import torch
warnings.filterwarnings('ignore')

from src.core.lora_config import LoRAConfig
from src.core.universe_selection import FixedIncomeUniverse
from src.analysis.lora_factor_analysis import LoRAFactorAnalysis
from src.optimization.optimization import PortfolioOptimizer, MLPortfolioOptimizer
from src.backtesting.backtest import Backtester
from src.utils.training_data_generator import FinancialTrainingDataGenerator


class LoRAFixedIncomeTradingSystem:
    def __init__(self, use_120b_model: bool = False):
        self.config = LoRAConfig()
        self.config.USE_120B_MODEL = use_120b_model
        
        self.config.validate_config()
        
        self.universe = FixedIncomeUniverse(self.config)
        self.factor_analysis = LoRAFactorAnalysis(self.config, self.config.get_model_name())
        self.optimizer = PortfolioOptimizer(self.config)
        self.ml_optimizer = MLPortfolioOptimizer(self.config)
        self.backtester = Backtester(self.config)
        self.data_generator = FinancialTrainingDataGenerator(self.config)
        
        print(f"LoRA Fixed Income Trading System initialized")
        print(f"Model: {'GPT-OSS-120B' if use_120b_model else 'GPT-OSS-20B'}")
        print(f"Device: {self.factor_analysis.device}")
        print(f"4-bit Quantization: {self.config.USE_4BIT_QUANTIZATION}")
    
    def prepare_training_data(self, generate_new: bool = True):
        print("Preparing training data...")
        
        if generate_new:
            texts, labels = self.data_generator.generate_comprehensive_training_data()
            
            if self.data_generator.validate_training_data(texts, labels):
                self.data_generator.save_training_data(texts, labels, self.config.TRAINING_DATA_PATH)
                print(f"Generated {len(texts)} training samples")
            else:
                print("Training data validation failed")
                return False
        else:
            try:
                texts, labels = self.data_generator.load_training_data(self.config.TRAINING_DATA_PATH)
                print(f"Loaded {len(texts)} training samples")
            except FileNotFoundError:
                print("No existing training data found, generating new data...")
                return self.prepare_training_data(generate_new=True)
        
        return texts, labels
    
    def fine_tune_lora_model(self, texts: List[str], labels: List[str]):
        print("Starting LoRA fine-tuning...")
        
        try:
            self.factor_analysis.fine_tune_lora(
                texts, 
                labels,
                epochs=self.config.NUM_EPOCHS,
                batch_size=self.config.BATCH_SIZE
            )
            
            self.factor_analysis.save_lora_model(self.config.LORA_MODEL_PATH)
            print("LoRA fine-tuning completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during LoRA fine-tuning: {e}")
            return False
    
    def run_complete_analysis(self):
        print("Starting LoRA-based Fixed Income Trading System Analysis...")
        
        start_date = self.config.BACKTEST_START_DATE
        end_date = self.config.BACKTEST_END_DATE
        
        print(f"1. Selecting Universe ({start_date} to {end_date})")
        prices = self.universe.get_historical_data(start_date, end_date)
        selected_tickers = self.universe.filter_universe(prices)
        print(f"Selected {len(selected_tickers)} securities: {selected_tickers}")
        
        if len(selected_tickers) == 0:
            print("No securities selected, using sample data")
            selected_tickers = ['TLT', 'IEF', 'LQD', 'HYG']
            dates = pd.date_range(start_date, end_date, freq='D')
            prices = pd.DataFrame({
                ticker: np.random.normal(100, 5, len(dates)) for ticker in selected_tickers
            }, index=dates)
            prices = prices.abs()
        
        filtered_prices = prices[selected_tickers]
        
        print("2. Analyzing Market Factors and Sentiment with LoRA")
        market_factors = self.factor_analysis.get_market_factors(start_date, end_date)
        rolling_features = self.factor_analysis.calculate_rolling_returns(filtered_prices)
        
        sentiment_features = pd.DataFrame()
        for ticker in selected_tickers[:5]:
            print(f"Analyzing sentiment for {ticker}...")
            sentiment = self.factor_analysis.aggregate_sentiment_features(ticker)
            sentiment_features[ticker] = pd.Series(sentiment)
        
        print("3. Running Portfolio Optimization")
        returns = filtered_prices.pct_change().dropna()
        
        optimization_results = {}
        
        mvo_result = self.optimizer.mean_variance_optimization(returns)
        if mvo_result:
            optimization_results['Mean Variance'] = mvo_result
        
        rp_result = self.optimizer.risk_parity_optimization(returns)
        if rp_result:
            optimization_results['Risk Parity'] = rp_result
        
        sector_result = self.optimizer.sector_constrained_optimization(
            returns, 
            self.universe.sector_mapping, 
            self.config.SECTOR_CONSTRAINTS
        )
        if sector_result:
            optimization_results['Sector Constrained'] = sector_result
        
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
    
    def run_lora_sentiment_analysis(self, text: str):
        print(f"Analyzing sentiment for: {text[:100]}...")
        
        sentiment = self.factor_analysis.analyze_financial_sentiment_lora(text)
        
        print("LoRA Sentiment Analysis Results:")
        print(f"Sentiment: {sentiment['sentiment']:.3f}")
        print(f"Confidence: {sentiment['confidence']:.3f}")
        print(f"Relevance: {sentiment['relevance']:.3f}")
        print(f"Risk Level: {sentiment['risk_level']:.3f}")
        
        return sentiment
    
    def run_lora_triplet_extraction(self, text: str):
        print(f"Extracting triplets from: {text[:100]}...")
        
        triplets = self.factor_analysis.extract_triplets(text)
        
        print("LoRA Triplet Extraction Results:")
        for i, (subject, verb, obj) in enumerate(triplets):
            print(f"Triplet {i+1}: ({subject}, {verb}, {obj})")
        
        return triplets
    
    def compare_lora_vs_baseline(self, text: str):
        print("Comparing LoRA vs Baseline Analysis...")
        
        lora_sentiment = self.factor_analysis.analyze_financial_sentiment_lora(text)
        baseline_sentiment = self.factor_analysis._fallback_sentiment_analysis(text)
        
        print("\nComparison Results:")
        print("LoRA Analysis:")
        print(f"  Sentiment: {lora_sentiment['sentiment']:.3f}")
        print(f"  Confidence: {lora_sentiment['confidence']:.3f}")
        print(f"  Relevance: {lora_sentiment['relevance']:.3f}")
        print(f"  Risk Level: {lora_sentiment['risk_level']:.3f}")
        
        print("\nBaseline Analysis:")
        print(f"  Sentiment: {baseline_sentiment['sentiment']:.3f}")
        print(f"  Confidence: {baseline_sentiment['confidence']:.3f}")
        print(f"  Relevance: {baseline_sentiment['relevance']:.3f}")
        print(f"  Risk Level: {baseline_sentiment['risk_level']:.3f}")
        
        return {
            'lora': lora_sentiment,
            'baseline': baseline_sentiment
        }
    
    def _print_summary(self, optimization_results, backtest_results):
        print("\n" + "="*60)
        print("LoRA-BASED OPTIMIZATION RESULTS SUMMARY")
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
        print("LoRA-BASED BACKTEST RESULTS SUMMARY")
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
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('LoRA Portfolio Performance', 'Risk-Return Scatter',
                              'Drawdown Comparison', 'Rolling Sharpe Comparison')
            )
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (strategy_name, result) in enumerate(backtest_results.items()):
                portfolio = result['portfolio_values']
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(x=portfolio.index, y=portfolio.values, 
                              name=f"LoRA {strategy_name}", line=dict(color=color)),
                    row=1, col=1
                )
                
                performance = result['performance_metrics']
                risk = result['risk_metrics']
                
                fig.add_trace(
                    go.Scatter(x=[risk['volatility']*100], y=[performance['annualized_return']*100],
                              mode='markers', name=f"LoRA {strategy_name} Risk-Return",
                              marker=dict(color=color, size=10)),
                    row=1, col=2
                )
                
                drawdown = result['drawdown_analysis']['drawdown_series']
                fig.add_trace(
                    go.Scatter(x=drawdown.index, y=drawdown.values*100,
                              name=f"LoRA {strategy_name} Drawdown", line=dict(color=color)),
                    row=2, col=1
                )
            
            fig.update_layout(height=800, title_text="LoRA-Based Strategy Comparison")
            fig.show()
            
        except ImportError:
            print("Plotly not available, skipping visualization")


def main():
    print("LoRA-based Fixed Income Trading System")
    print("="*50)
    
    system = LoRAFixedIncomeTradingSystem(use_120b_model=False)
    
    if os.path.exists(system.config.LORA_MODEL_PATH):
        print("Loading existing LoRA model...")
        system.factor_analysis.load_lora_model(system.config.LORA_MODEL_PATH)
    else:
        print("No existing LoRA model found. Preparing training data...")
        texts, labels = system.prepare_training_data(generate_new=True)
        
        if texts and labels:
            print("Fine-tuning LoRA model...")
            success = system.fine_tune_lora_model(texts, labels)
            if not success:
                print("LoRA fine-tuning failed, using baseline methods")
    
    optimization_results, backtest_results = system.run_complete_analysis()
    
    demo_text = "Federal Reserve signals potential rate cuts, bond yields decline across the curve. Treasury prices rally as investors anticipate accommodative monetary policy."
    system.run_lora_sentiment_analysis(demo_text)
    
    demo_text2 = "The Federal Reserve raised interest rates by 25 basis points, causing bond yields to increase."
    system.run_lora_triplet_extraction(demo_text2)
    
    system.compare_lora_vs_baseline(demo_text)
    
    print("\n" + "="*50)
    print("LoRA-based Analysis Complete!")
    
    return optimization_results, backtest_results


if __name__ == "__main__":
    main() 
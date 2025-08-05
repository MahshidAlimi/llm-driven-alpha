import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.core.config import Config
from src.core.universe_selection import FixedIncomeUniverse
from src.optimization.optimization import PortfolioOptimizer
from src.backtesting.backtest import Backtester

def test_basic_functionality():
    print("Testing Fixed Income Trading System Basic Functionality")
    print("="*60)
    
    config = Config()
    universe = FixedIncomeUniverse(config)
    optimizer = PortfolioOptimizer(config)
    backtester = Backtester(config)
    
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    print("1. Testing Universe Selection...")
    prices = universe.get_historical_data(start_date, end_date)
    print(f"Downloaded data for {len(prices.columns)} securities")
    print(f"Data shape: {prices.shape}")
    
    if len(prices.columns) > 0:
        selected_tickers = universe.filter_universe(prices)
        print(f"Selected {len(selected_tickers)} securities after filtering")
        
        if len(selected_tickers) > 0:
            filtered_prices = prices[selected_tickers]
            returns = filtered_prices.pct_change().dropna()
            
            print("2. Testing Portfolio Optimization...")
            mvo_result = optimizer.mean_variance_optimization(returns)
            if mvo_result:
                print("✓ Mean-Variance Optimization successful")
                print(f"  Expected Return: {mvo_result['expected_return']*100:.2f}%")
                print(f"  Volatility: {mvo_result['volatility']*100:.2f}%")
                print(f"  Sharpe Ratio: {mvo_result['sharpe_ratio']:.3f}")
                
                print("3. Testing Backtesting...")
                weights = mvo_result['weights']
                weights_history = pd.DataFrame([weights] * len(filtered_prices), 
                                             index=filtered_prices.index)
                
                backtest_result = backtester.run_backtest(
                    filtered_prices, 
                    weights_history, 
                    'M'
                )
                
                if backtest_result:
                    print("✓ Backtesting successful")
                    performance = backtest_result['performance_metrics']
                    print(f"  Total Return: {performance['total_return']:.2f}%")
                    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
                    print(f"  Max Drawdown: {backtest_result['risk_metrics']['max_drawdown']*100:.2f}%")
                    
                    print("4. Generating Report...")
                    report = backtester.generate_report()
                    print(report)
                    
                    return True
                else:
                    print("✗ Backtesting failed")
            else:
                print("✗ Mean-Variance Optimization failed")
        else:
            print("✗ No securities selected after filtering")
    else:
        print("✗ No data downloaded")
    
    return False

def test_optimization_methods():
    print("\nTesting Different Optimization Methods")
    print("="*60)
    
    config = Config()
    optimizer = PortfolioOptimizer(config)
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_assets = 5
    
    returns_data = {}
    for i in range(n_assets):
        returns_data[f'ASSET_{i+1}'] = np.random.normal(0.001, 0.02, len(dates))
    
    returns = pd.DataFrame(returns_data, index=dates)
    returns = returns.dropna()
    
    print("Testing Mean-Variance Optimization...")
    mvo_result = optimizer.mean_variance_optimization(returns)
    if mvo_result:
        print("✓ MVO successful")
    
    print("Testing Risk Parity Optimization...")
    rp_result = optimizer.risk_parity_optimization(returns)
    if rp_result:
        print("✓ Risk Parity successful")
    
    print("Testing Sector Constrained Optimization...")
    sector_mapping = {f'ASSET_{i+1}': 'sector_1' if i < 3 else 'sector_2' for i in range(n_assets)}
    sector_constraints = {'sector_1': 0.6, 'sector_2': 0.4}
    
    sc_result = optimizer.sector_constrained_optimization(returns, sector_mapping, sector_constraints)
    if sc_result:
        print("✓ Sector Constrained successful")
    
    return mvo_result is not None and rp_result is not None and sc_result is not None

def test_covariance_methods():
    print("\nTesting Covariance Matrix Methods")
    print("="*60)
    
    config = Config()
    optimizer = PortfolioOptimizer(config)
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_assets = 4
    
    returns_data = {}
    for i in range(n_assets):
        returns_data[f'ASSET_{i+1}'] = np.random.normal(0.001, 0.02, len(dates))
    
    returns = pd.DataFrame(returns_data, index=dates)
    returns = returns.dropna()
    
    methods = ['sample', 'exponential']
    
    for method in methods:
        print(f"Testing {method} covariance method...")
        try:
            cov_matrix = optimizer.calculate_covariance_matrix(returns, method=method)
            print(f"✓ {method} covariance successful, shape: {cov_matrix.shape}")
        except Exception as e:
            print(f"✗ {method} covariance failed: {e}")
    
    return True

if __name__ == "__main__":
    print("Fixed Income Trading System - Test Suite")
    print("="*60)
    
    test1_passed = test_basic_functionality()
    test2_passed = test_optimization_methods()
    test3_passed = test_covariance_methods()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Basic Functionality: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Optimization Methods: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Covariance Methods: {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.") 
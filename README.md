# Fixed Income Trading System

A comprehensive fixed income trading system that combines traditional portfolio optimization with advanced machine learning and natural language processing techniques.

## Features

### 1. Universe Selection
- Fixed income ETF universe covering government, corporate, emerging markets, and high-yield bonds
- Dynamic filtering based on volatility, Sharpe ratio, and liquidity metrics
- Sector-based classification and constraints

### 2. Factor Analysis
- **Market Factors**: Equity markets, credit spreads, Treasury yields, dollar index, commodities
- **Rolling Returns**: Momentum and mean reversion signals across multiple timeframes
- **NLP Sentiment Analysis**: 
  - News sentiment from financial news sources
  - Twitter sentiment analysis
  - Reddit sentiment from investment communities
  - Central bank reports and speeches analysis
- **LLM Integration**: OpenAI GPT-4 and Claude for advanced text analysis
- **Triplet Extraction**: Subject-verb-object relationships from financial text

### 3. Portfolio Optimization
- **Mean-Variance Optimization**: Traditional Markowitz approach
- **Sector-Constrained Optimization**: Risk management through sector limits
- **Risk Parity**: Equal risk contribution across assets
- **Black-Litterman**: Incorporating market views and confidence levels
- **Machine Learning**: Random Forest and Neural Network return prediction
- **Reinforcement Learning**: Actor-Critic networks for dynamic portfolio allocation

### 4. Backtesting
- Comprehensive performance metrics (Sharpe, Sortino, Calmar ratios)
- Risk metrics (VaR, CVaR, maximum drawdown)
- Transaction cost modeling
- Multiple rebalancing frequencies
- Interactive visualizations with Plotly

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-driven-alpha
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Set up environment variables:
```bash
cp env_example.txt .env
# Edit .env with your API keys
```

## Configuration

Edit `config.py` to customize:
- Fixed income universe
- Sector constraints
- Backtest parameters
- Risk management settings
- LLM provider preferences

## Usage

### Basic Usage
```python
# Option 1: Import and use
from src.main import FixedIncomeTradingSystem

system = FixedIncomeTradingSystem()
optimization_results, backtest_results = system.run_complete_analysis()

# Option 2: Run from command line
python run.py

# Option 3: Install and run as command
pip install -e .
fixed-income-trading
```

### Testing

The project includes comprehensive unit tests for all components:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python tests/run_tests.py --module test_universe_selection

# Run specific test class
python tests/run_tests.py --module test_optimization --class TestPortfolioOptimizer

# Run individual test files
python -m pytest tests/
python -m unittest tests.test_universe_selection
```

#### Test Coverage

- **Unit Tests**: Individual component testing with mocked dependencies
- **Integration Tests**: End-to-end workflow testing
- **Edge Case Testing**: Error handling and boundary conditions
- **Performance Testing**: Metrics validation and consistency checks

#### Test Structure

```
tests/
├── test_config.py              # Configuration tests
├── test_universe_selection.py  # Universe selection tests
├── test_factor_analysis.py     # Factor analysis tests
├── test_optimization.py        # Optimization tests
├── test_backtest.py           # Backtesting tests
├── test_system.py             # Integration tests
└── run_tests.py               # Test runner
```

### ML-Enhanced Strategy
```python
ml_predictions = system.run_ml_enhanced_strategy()
```

### RL Strategy
```python
rl_optimizer = system.run_rl_strategy()
```

## API Keys Required

- **OpenAI API Key**: For GPT-4 text analysis
- **Anthropic API Key**: For Claude text analysis
- **Twitter Bearer Token**: For social media sentiment
- **Reddit API**: For community sentiment analysis

## Project Structure

```
llm-driven-alpha/
├── src/
│   ├── core/
│   │   ├── config.py              # Configuration settings
│   │   └── universe_selection.py  # Universe selection and filtering
│   ├── analysis/
│   │   └── factor_analysis.py     # Factor analysis and NLP
│   ├── optimization/
│   │   └── optimization.py        # Portfolio optimization methods
│   ├── backtesting/
│   │   └── backtest.py           # Backtesting engine
│   └── main.py                   # Main execution script
├── tests/
│   └── test_system.py            # System tests
├── configs/
│   └── env_example.txt           # Environment variables template
├── data/                         # Data storage directory
├── docs/                         # Documentation
├── run.py                       # Main entry point
├── setup.py                     # Package setup
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Key Components

### Universe Selection (`universe_selection.py`)
- Downloads historical data for fixed income ETFs
- Calculates risk and return metrics
- Filters universe based on criteria
- Maps securities to sectors

### Factor Analysis (`factor_analysis.py`)
- Extracts market factors and rolling returns
- Performs sentiment analysis on multiple data sources
- Uses LLMs for advanced text analysis
- Extracts triplets for relationship analysis

### Optimization (`optimization.py`)
- Implements multiple optimization strategies
- Includes ML and RL approaches
- Handles sector constraints and risk management
- Provides Black-Litterman framework

### Backtesting (`backtest.py`)
- Comprehensive performance analysis
- Risk metrics calculation
- Interactive visualizations
- Strategy comparison tools

## Performance Metrics

The system calculates:
- **Return Metrics**: Total return, annualized return, excess return
- **Risk Metrics**: Volatility, VaR, CVaR, maximum drawdown
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, information ratio
- **Trading Metrics**: Win rate, profit factor, skewness, kurtosis

## Visualization

The system generates interactive plots showing:
- Portfolio performance vs benchmark
- Weight allocation over time
- Drawdown analysis
- Rolling Sharpe ratios
- Monthly returns heatmap
- Risk-return scatter plots

## Advanced Features

### LLM Integration
- Uses LiteLLM for flexible LLM provider switching
- Supports both OpenAI and Anthropic models
- Implements aspect-based sentiment analysis
- Extracts financial relationships from text

### Machine Learning
- Random Forest for return prediction
- Neural Networks for non-linear patterns
- Feature engineering from market data and sentiment
- Cross-validation and model selection

### Reinforcement Learning
- Actor-Critic networks for portfolio optimization
- State representation including market factors and sentiment
- Reward function based on portfolio returns
- Continuous learning and adaptation


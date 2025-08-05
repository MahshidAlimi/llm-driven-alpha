import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    
    MONGODB_URI = os.getenv('MONGODB_URI', None)
    REDIS_URL = os.getenv('REDIS_URL', None)
    
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4')
    
    FIXED_INCOME_UNIVERSE = [
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'VCIT', 'VCSH', 'BND', 'AGG',
        'BNDX', 'EMB', 'PCY', 'JNK', 'SJNK', 'VWOB', 'IGOV', 'BWX', 'EMLC'
    ]
    
    SECTOR_CONSTRAINTS = {
        'government': 0.4,
        'corporate': 0.3,
        'emerging_markets': 0.2,
        'high_yield': 0.1
    }
    
    BACKTEST_START_DATE = '2020-01-01'
    BACKTEST_END_DATE = '2024-01-01'
    
    REBALANCE_FREQUENCY = 'M'
    LOOKBACK_PERIOD = 252
    ROLLING_WINDOW = 60
    
    RISK_FREE_RATE = 0.02
    TARGET_VOLATILITY = 0.08
    MAX_WEIGHT = 0.15
    MIN_WEIGHT = 0.02 
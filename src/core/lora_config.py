import os
from dotenv import load_dotenv

load_dotenv()

class LoRAConfig:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    
    MODEL_NAME_120B = "microsoft/DialoGPT-large"
    MODEL_NAME_20B = "microsoft/DialoGPT-medium"
    
    USE_120B_MODEL = os.getenv('USE_120B_MODEL', 'false').lower() == 'true'
    USE_4BIT_QUANTIZATION = os.getenv('USE_4BIT_QUANTIZATION', 'true').lower() == 'true'
    
    LORA_R = int(os.getenv('LORA_R', '16'))
    LORA_ALPHA = int(os.getenv('LORA_ALPHA', '32'))
    LORA_DROPOUT = float(os.getenv('LORA_DROPOUT', '0.05'))
    
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', '512'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '2e-4'))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '3'))
    WARMUP_STEPS = int(os.getenv('WARMUP_STEPS', '100'))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4'))
    
    LORA_MODEL_PATH = os.getenv('LORA_MODEL_PATH', './lora_financial_model')
    TRAINING_DATA_PATH = os.getenv('TRAINING_DATA_PATH', './training_data')
    
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
    
    FINANCIAL_KEYWORDS = [
        'bond', 'yield', 'interest', 'rate', 'treasury', 'credit', 'spread',
        'inflation', 'fed', 'federal reserve', 'monetary', 'policy', 'market',
        'investment', 'portfolio', 'risk', 'volatility', 'return', 'price',
        'duration', 'convexity', 'maturity', 'coupon', 'par', 'premium',
        'discount', 'liquidity', 'default', 'rating', 'sector', 'issuer'
    ]
    
    RISK_KEYWORDS = [
        'crash', 'fall', 'decline', 'drop', 'volatile', 'uncertainty', 'risk',
        'default', 'bankruptcy', 'crisis', 'recession', 'depression', 'panic',
        'sell-off', 'correction', 'bear market', 'downturn', 'loss', 'negative'
    ]
    
    SENTIMENT_THRESHOLDS = {
        'very_negative': -0.5,
        'negative': -0.1,
        'neutral': 0.1,
        'positive': 0.5,
        'very_positive': 1.0
    }
    
    LORA_TARGET_MODULES = {
        'gpt_oss_120b': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'gpt_oss_20b': ["q_proj", "v_proj", "k_proj", "o_proj"],
        'default': ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
    
    @classmethod
    def get_model_name(cls):
        if cls.USE_120B_MODEL:
            return cls.MODEL_NAME_120B
        else:
            return cls.MODEL_NAME_20B
    
    @classmethod
    def get_target_modules(cls):
        if cls.USE_120B_MODEL:
            return cls.LORA_TARGET_MODULES['gpt_oss_120b']
        else:
            return cls.LORA_TARGET_MODULES['gpt_oss_20b']
    
    @classmethod
    def validate_config(cls):
        errors = []
        
        if cls.LORA_R <= 0:
            errors.append("LORA_R must be positive")
        if cls.LORA_ALPHA <= 0:
            errors.append("LORA_ALPHA must be positive")
        if not 0 <= cls.LORA_DROPOUT <= 1:
            errors.append("LORA_DROPOUT must be between 0 and 1")
        
        if cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")
        if cls.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be positive")
        if cls.NUM_EPOCHS <= 0:
            errors.append("NUM_EPOCHS must be positive")
        
        if cls.RISK_FREE_RATE < 0:
            errors.append("RISK_FREE_RATE must be non-negative")
        if cls.TARGET_VOLATILITY <= 0:
            errors.append("TARGET_VOLATILITY must be positive")
        if cls.MAX_WEIGHT <= 0 or cls.MAX_WEIGHT > 1:
            errors.append("MAX_WEIGHT must be between 0 and 1")
        if cls.MIN_WEIGHT < 0 or cls.MIN_WEIGHT >= cls.MAX_WEIGHT:
            errors.append("MIN_WEIGHT must be non-negative and less than MAX_WEIGHT")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True 
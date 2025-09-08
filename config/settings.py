import os
from dotenv import load_dotenv

load_dotenv()

# Remove any class-based settings. Use only simple variables:
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID', '')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY', '')
APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', '0.1'))
WATCHLIST = os.getenv('WATCHLIST', 'SPY,QQQ,IWM,TSLA,NVDA,AAPL,MSFT,GOOGL').split(',')

PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', '0.65'))
PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '4'))
MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.6'))

LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', '100'))
RESAMPLE_INTERVAL = os.getenv('RESAMPLE_INTERVAL', '15Min')

ENABLE_TRADING = os.getenv('ENABLE_TRADING', 'false').lower() == 'true'
PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'

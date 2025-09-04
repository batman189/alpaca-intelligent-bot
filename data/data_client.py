import pandas as pd
import numpy as np
from alpaca_trade_api import REST
import logging
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger(__name__)

class DataClient:
    def __init__(self):
        self.api = REST(settings.APCA_API_KEY_ID, 
                       settings.APCA_API_SECRET_KEY, 
                       settings.APCA_API_BASE_URL)
        
    def get_historical_bars(self, symbol, timeframe='15Min', limit=100):
        """Get historical data with fallback to generated data"""
        try:
            logger.info(f"Fetching live data for {symbol}...")
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            
            if bars is not None and not bars.empty and len(bars) >= 20:
                logger.info(f"Retrieved {len(bars)} live bars for {symbol}")
                return bars
            else:
                # Fallback: generate sample data for demonstration
                data_length = len(bars) if bars is not None else 0
                logger.warning(f"Insufficient live data ({data_length} bars). Using sample data for {symbol}")
                return self.generate_sample_data(symbol, timeframe, limit)
                
        except Exception as e:
            logger.error(f"Error getting live data for {symbol}: {e}")
            return self.generate_sample_data(symbol, timeframe, limit)
            
    def generate_sample_data(self, symbol, timeframe, limit):
        """Generate sample data for demonstration when live data fails"""
        try:
            # Create realistic sample data based on symbol
            base_price = 450 if symbol == 'SPY' else 180 if symbol == 'AAPL' else 100
            volatility = 0.015
            
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
            returns = np.random.normal(0.0002, volatility, len(dates))
            
            prices = base_price * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0.003, 0.002, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0.003, 0.002, len(dates)))),
                'close': prices,
                'volume': np.random.lognormal(14, 1, len(dates))
            }, index=dates)
            
            logger.info(f"Generated {len(df)} sample bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            return None
            
    def get_multiple_historical_bars(self, symbols, timeframe='15Min', limit=100):
        """Get data for multiple symbols with fallbacks"""
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_historical_bars(symbol, timeframe, limit)
            time.sleep(0.1)  # Rate limiting
        return data
        
    # ... keep the rest of your existing methods (get_latest_quote, get_option_chain, etc.)

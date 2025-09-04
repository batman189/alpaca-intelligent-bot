import pandas as pd
import numpy as np
from alpaca_trade_api import REST
import logging
from datetime import datetime, timedelta
from config.settings import settings

logger = logging.getLogger(__name__)

class DataClient:
    def __init__(self):
        self.api = REST(settings.APCA_API_KEY_ID, 
                       settings.APCA_API_SECRET_KEY, 
                       settings.APCA_API_BASE_URL)
        self.symbol_cache = {}
        
    def get_historical_bars(self, symbol, timeframe='15Min', limit=100):
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            if not bars.empty:
                logger.info(f"Retrieved {len(bars)} bars for {symbol}")
                return bars
            return None
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
            
    def get_multiple_historical_bars(self, symbols, timeframe='15Min', limit=100):
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_historical_bars(symbol, timeframe, limit)
            import time
            time.sleep(0.1)
        return data
        
    def get_latest_quote(self, symbol):
        try:
            quote = self.api.get_latest_quote(symbol)
            return {
                'ask_price': float(quote.askprice),
                'bid_price': float(quote.bidprice),
                'ask_size': int(quote.asksize),
                'bid_size': int(quote.bidsize)
            }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
            
    def get_option_chain(self, symbol, expiration_date=None):
        try:
            latest_trade = self.api.get_latest_trade(symbol)
            current_price = float(latest_trade.price)
            
            strikes = [round(current_price * (1 + i * 0.05)) for i in range(-3, 4)]
            
            option_chain = []
            for strike in strikes:
                option_chain.append({
                    'symbol': f"{symbol}{expiration_date or '250117'}C{strike:08d}",
                    'strike': strike,
                    'type': 'call',
                    'expiration': expiration_date or '2025-01-17'
                })
                
            return option_chain
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return None
            
    def get_account_info(self):
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

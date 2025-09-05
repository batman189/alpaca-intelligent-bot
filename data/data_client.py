import pandas as pd
import numpy as np
from alpaca_trade_api import REST
import logging
from datetime import datetime, timedelta
import os
import time

logger = logging.getLogger(__name__)

# Direct environment variable access - NO SETTINGS IMPORT
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID', '')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY', '')
APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

class DataClient:
    def __init__(self):
        try:
            self.api = REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)
            logger.info("Alpaca API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
            self.api = None
        
    def get_historical_bars(self, symbol, timeframe='15Min', limit=100):
        """Get historical data with fallback to generated data"""
        try:
            if self.api is None:
                logger.warning(f"API not available. Generating sample data for {symbol}")
                return self.generate_sample_data(symbol, timeframe, limit)
                
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
            if symbol == 'SPY':
                base_price = 450
            elif symbol == 'AAPL':
                base_price = 180
            elif symbol == 'NVDA':
                base_price = 125
            elif symbol == 'TSLA':
                base_price = 250
            elif symbol == 'GOOGL':
                base_price = 140
            elif symbol == 'MSFT':
                base_price = 300
            elif symbol == 'QQQ':
                base_price = 380
            elif symbol == 'IWM':
                base_price = 190
            else:
                base_price = 100
                
            volatility = 0.015
            
            # Create date range based on timeframe
            end_date = datetime.now()
            if timeframe == '15Min':
                start_date = end_date - timedelta(days=7)
                freq = '15min'
            elif timeframe == '1D':
                start_date = end_date - timedelta(days=limit)
                freq = 'D'
            else:
                start_date = end_date - timedelta(days=30)
                freq = 'H'
            
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            if len(dates) > limit:
                dates = dates[-limit:]
            
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
        
    def get_latest_quote(self, symbol):
        """Get the latest quote for a symbol with fallback"""
        try:
            if self.api is None:
                logger.warning(f"API not available. Returning sample quote for {symbol}")
                return self.generate_sample_quote(symbol)
                
            quote = self.api.get_latest_quote(symbol)
            return {
                'ask_price': float(quote.askprice),
                'bid_price': float(quote.bidprice),
                'ask_size': int(quote.asksize),
                'bid_size': int(quote.bidsize)
            }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return self.generate_sample_quote(symbol)
            
    def generate_sample_quote(self, symbol):
        """Generate sample quote data"""
        base_price = 100
        if symbol == 'SPY': base_price = 450
        elif symbol == 'AAPL': base_price = 180
        elif symbol == 'NVDA': base_price = 125
        
        return {
            'ask_price': base_price * 1.001,
            'bid_price': base_price * 0.999,
            'ask_size': 100,
            'bid_size': 150
        }
            
    def get_option_chain(self, symbol, expiration_date=None):
        """Get option chain for a symbol (simplified)"""
        try:
            current_price = 100.0  # Default price
            
            # Try to get real price first
            try:
                if self.api:
                    latest_trade = self.api.get_latest_trade(symbol)
                    current_price = float(latest_trade.price)
            except:
                pass
                
            # Adjust base price based on symbol
            if symbol == 'SPY': current_price = 450
            elif symbol == 'AAPL': current_price = 180
            elif symbol == 'NVDA': current_price = 125
            elif symbol == 'TSLA': current_price = 250
                
            strikes = [round(current_price * (1 + i * 0.05)) for i in range(-3, 4)]
            
            option_chain = []
            for strike in strikes:
                option_chain.append({
                    'symbol': f"{symbol}{expiration_date or '260116'}C{strike:08d}",
                    'strike': strike,
                    'type': 'call',
                    'expiration': expiration_date or '2026-01-16'
                })
                
            return option_chain
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return None

    def get_detailed_option_chain(self, symbol, expiration_date=None):
        """Get detailed options chain with Greeks and volume"""
        try:
            chain = self.get_option_chain(symbol, expiration_date)
            
            if not chain:
                return None
                
            # Get current price for calculations
            current_price = 100
            try:
                if self.api:
                    latest_trade = self.api.get_latest_trade(symbol)
                    current_price = float(latest_trade.price)
            except:
                # Use symbol-based default prices
                if symbol == 'SPY': current_price = 450
                elif symbol == 'AAPL': current_price = 180
                elif symbol == 'NVDA': current_price = 125
                elif symbol == 'TSLA': current_price = 250
                elif symbol == 'MSFT': current_price = 300
            
            for option in chain:
                # Add Greeks and market data (simplified for demo)
                strike = option['strike']
                option_type = option['type']
                
                # Simplified Greeks calculation
                option['delta'] = self.calculate_delta(strike, current_price, option_type)
                option['gamma'] = 0.05
                option['theta'] = -0.02
                option['vega'] = 0.15
                option['implied_vol'] = 0.25
                option['volume'] = np.random.randint(100, 5000)
                option['open_interest'] = np.random.randint(500, 10000)
                option['price'] = current_price * 0.01 * (1 + abs(strike - current_price) / current_price)
                
            return chain
            
        except Exception as e:
            logger.error(f"Error getting detailed option chain: {e}")
            return None

    def calculate_delta(self, strike, current_price, option_type):
        """Calculate option delta"""
        # Simplified delta calculation based on moneyness
        moneyness = strike / current_price
        
        if option_type == 'call':
            if moneyness < 0.95: return 0.8  # Deep ITM
            elif moneyness < 1.05: return 0.5  # ATM
            else: return 0.2  # OTM
        else:  # put
            if moneyness > 1.05: return -0.8  # Deep ITM
            elif moneyness > 0.95: return -0.5  # ATM
            else: return -0.2  # OTM
            
    def get_account_info(self):
        """Get current account information with fallback"""
        try:
            if self.api is None:
                logger.warning("API not available. Returning sample account info")
                return self.generate_sample_account_info()
                
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return self.generate_sample_account_info()
            
    def generate_sample_account_info(self):
        """Generate sample account data for demonstration"""
        return {
            'equity': 10000.0,
            'cash': 5000.0,
            'buying_power': 10000.0,
            'portfolio_value': 10000.0
        }
        
    def get_positions(self):
        """Get current positions with fallback"""
        try:
            if self.api is None:
                return {}
                
            positions = self.api.list_positions()
            return {pos.symbol: {
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'current_price': float(pos.current_price)
            } for pos in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

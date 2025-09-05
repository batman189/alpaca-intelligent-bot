import logging
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import requests
import json

logger = logging.getLogger(__name__)

class DataClient:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            base_url=os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.secret_key = os.getenv('APCA_API_SECRET_KEY')
        self.base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
    def get_multiple_historical_bars(self, symbols, timeframe='15Min', limit=100):
        """Get historical bars for multiple symbols"""
        try:
            market_data = {}
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        limit=limit
                    ).df
                    market_data[symbol] = bars
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    market_data[symbol] = None
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {symbol: None for symbol in symbols}
            
    def get_account_info(self):
        """Get account information"""
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
            return {'equity': 10000, 'cash': 5000, 'buying_power': 10000, 'portfolio_value': 10000}
            
    def get_detailed_option_chain(self, symbol: str, expiration_date: str = None) -> List[Dict]:
        """
        Get option chain - Simplified version that uses mock data
        since Alpaca's options API seems to have issues
        """
        try:
            # If no expiration date provided, use the default from environment
            if not expiration_date:
                expiration_date = os.getenv('OPTIONS_EXPIRATION', '2026-01-16')
            
            # For now, use mock data since Alpaca's options API is not working
            return self._get_mock_option_chain(symbol, expiration_date)
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return self._get_mock_option_chain(symbol, expiration_date)
    
    def _get_mock_option_chain(self, symbol: str, expiration_date: str) -> List[Dict]:
        """Create realistic mock option chain for testing"""
        logger.info(f"Using mock option chain for {symbol} expiring {expiration_date}")
        
        mock_chain = []
        current_price = 150  # Reasonable default price
        
        # Get actual current price for the symbol if possible
        try:
            bars = self.api.get_bars(symbol, '1Min', limit=1).df
            if not bars.empty:
                current_price = bars['close'].iloc[-1]
        except:
            pass  # Use default price if unable to get real price
        
        # Create call options (5 strikes above and below current price)
        for i in range(-5, 6):
            strike = current_price + (i * 5)  # $5 increments
            if strike <= 0:
                continue
                
            # Calculate reasonable option price based on distance from current price
            price_diff_pct = abs(strike - current_price) / current_price
            option_price = max(0.01, current_price * 0.05 * (1 - price_diff_pct))
            
            # Calculate reasonable delta
            if strike <= current_price:
                delta = 0.6 + (0.4 * (1 - price_diff_pct))
            else:
                delta = 0.4 * (1 - price_diff_pct)
            
            delta = max(0.1, min(0.9, delta))  # Keep delta between 0.1-0.9
            
            mock_chain.append({
                'symbol': f"{symbol}{expiration_date.replace('-', '')}C{int(strike*1000):08d}",
                'type': 'call',
                'strike': strike,
                'expiration': expiration_date,
                'price': option_price,
                'delta': delta,
                'volume': 100,
                'open_interest': 50
            })
        
        # Create put options (5 strikes above and below current price)
        for i in range(-5, 6):
            strike = current_price + (i * 5)  # $5 increments
            if strike <= 0:
                continue
                
            # Calculate reasonable option price based on distance from current price
            price_diff_pct = abs(strike - current_price) / current_price
            option_price = max(0.01, current_price * 0.05 * (1 - price_diff_pct))
            
            # Calculate reasonable delta
            if strike >= current_price:
                delta = 0.6 + (0.4 * (1 - price_diff_pct))
            else:
                delta = 0.4 * (1 - price_diff_pct)
            
            delta = max(0.1, min(0.9, delta))  # Keep delta between 0.1-0.9
            
            mock_chain.append({
                'symbol': f"{symbol}{expiration_date.replace('-', '')}P{int(strike*1000):08d}",
                'type': 'put',
                'strike': strike,
                'expiration': expiration_date,
                'price': option_price,
                'delta': delta,
                'volume': 100,
                'open_interest': 50
            })
        
        logger.info(f"Created {len(mock_chain)} mock options for {symbol}")
        return mock_chain

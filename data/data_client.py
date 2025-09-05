import logging
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DataClient:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            base_url=os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
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
        Get option chain directly from Alpaca to ensure symbol compatibility
        This returns options in the exact format that Alpaca expects for trading
        """
        try:
            # If no expiration date provided, use the default from environment
            if not expiration_date:
                expiration_date = os.getenv('OPTIONS_EXPIRATION', '2026-01-16')
            
            # Get options chain from Alpaca
            options_chain = self.api.get_options_chain(
                symbol,
                expiration=expiration_date
            )
            
            # Convert to list of dictionaries in the format our bot expects
            formatted_chain = []
            for option in options_chain:
                formatted_chain.append({
                    'symbol': option.symbol,  # This is the EXACT symbol Alpaca expects
                    'type': 'call' if option.right == 'C' else 'put',
                    'strike': float(option.strike),
                    'expiration': expiration_date,
                    'price': float(option.close) if option.close else float(option.last) if option.last else 0.01,
                    'delta': float(option.greeks.delta) if option.greeks and option.greeks.delta else 0.5,
                    'volume': int(option.volume) if option.volume else 0,
                    'open_interest': int(option.open_interest) if option.open_interest else 0
                })
            
            logger.info(f"Retrieved {len(formatted_chain)} options for {symbol} expiring {expiration_date}")
            return formatted_chain
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return None

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
        Get option chain using Alpaca's options API
        Uses direct HTTP requests to the options endpoint
        """
        try:
            # If no expiration date provided, use the default from environment
            if not expiration_date:
                expiration_date = os.getenv('OPTIONS_EXPIRATION', '2026-01-16')
            
            # Format expiration date for Alpaca API (YYYY-MM-DD)
            expiration_formatted = expiration_date
            
            # Make HTTP request to Alpaca's options endpoint
            url = f"https://data.alpaca.markets/v1beta1/options/contracts"
            params = {
                'underlying_symbol': symbol,
                'expiration_date': expiration_formatted,
                'status': 'active'
            }
            
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.secret_key
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            options_data = response.json()
            
            formatted_chain = []
            for contract in options_data.get('contracts', []):
                # Get the latest quote for this option contract
                quote_url = f"https://data.alpaca.markets/v1beta1/options/{contract['symbol']}/quotes/latest"
                quote_response = requests.get(quote_url, headers=headers)
                
                price = 0.01
                if quote_response.status_code == 200:
                    quote_data = quote_response.json()
                    if quote_data.get('quote'):
                        price = float(quote_data['quote'].get('ap', quote_data['quote'].get('bp', 0.01)))
                
                formatted_chain.append({
                    'symbol': contract['symbol'],  # The exact symbol Alpaca expects
                    'type': 'call' if contract['right'] == 'C' else 'put',
                    'strike': float(contract['strike_price']),
                    'expiration': contract['expiration_date'],
                    'price': price,
                    'delta': 0.5,  # Default value since we don't have Greeks from this endpoint
                    'volume': 0,   # Default values
                    'open_interest': 0
                })
            
            logger.info(f"Retrieved {len(formatted_chain)} options for {symbol} expiring {expiration_date}")
            return formatted_chain
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            # Fallback: return a simple mock option chain
            return self._get_mock_option_chain(symbol, expiration_date)
    
    def _get_mock_option_chain(self, symbol: str, expiration_date: str) -> List[Dict]:
        """Fallback mock option chain when API fails"""
        logger.warning(f"Using mock option chain for {symbol}")
        
        mock_chain = []
        current_price = 100  # Default price for mock data
        
        # Create some call options
        for strike in [current_price * 0.9, current_price, current_price * 1.1]:
            mock_chain.append({
                'symbol': f"{symbol}{expiration_date.replace('-', '')}C{int(strike*1000):08d}",
                'type': 'call',
                'strike': strike,
                'expiration': expiration_date,
                'price': max(0.01, strike * 0.05),
                'delta': 0.6 if strike <= current_price else 0.4,
                'volume': 100,
                'open_interest': 50
            })
        
        # Create some put options
        for strike in [current_price * 0.9, current_price, current_price * 1.1]:
            mock_chain.append({
                'symbol': f"{symbol}{expiration_date.replace('-', '')}P{int(strike*1000):08d}",
                'type': 'put',
                'strike': strike,
                'expiration': expiration_date,
                'price': max(0.01, strike * 0.05),
                'delta': 0.4 if strike <= current_price else 0.6,
                'volume': 100,
                'open_interest': 50
            })
        
        return mock_chain

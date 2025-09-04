import logging
import os  # ADD THIS IMPORT
from alpaca_trade_api import REST

logger = logging.getLogger(__name__)

# Direct environment variable access
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID', '')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY', '')
APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
ENABLE_TRADING = os.getenv('ENABLE_TRADING', 'false').lower() == 'true'
PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'

class ExecutionClient:
    def __init__(self):
        try:
            self.api = REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
            self.api = None
        
    def place_option_order(self, symbol: str, quantity: int, order_type: str = 'call',
                         strike: float = None, expiration: str = None) -> bool:
        if not ENABLE_TRADING:
            logger.info(f"Would place {order_type} order for {quantity} {symbol}")
            return True
            
        try:
            if self.api is None:
                logger.info(f"API not available. Would place {order_type} order for {quantity} {symbol}")
                return True
                
            option_symbol = self._resolve_option_symbol(symbol, strike, expiration, order_type)
            
            if not option_symbol:
                logger.error(f"Could not resolve option symbol for {symbol}")
                return False
                
            order = self.api.submit_order(
                symbol=option_symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"Option order placed: {order_type} {quantity} {option_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing option order: {e}")
            return False
            
    def _resolve_option_symbol(self, symbol: str, strike: float, 
                             expiration: str, order_type: str) -> str:
        strike_str = f"{int(strike * 1000):08d}"
        return f"{symbol}{expiration.replace('-', '')}{'C' if order_type == 'call' else 'P'}{strike_str}"
        
    def close_position(self, symbol: str) -> bool:
        if not ENABLE_TRADING:
            logger.info(f"Would close position for {symbol}")
            return True
            
        try:
            if self.api is None:
                logger.info(f"API not available. Would close position for {symbol}")
                return True
                
            self.api.close_position(symbol)
            logger.info(f"Position closed for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
            
    def get_current_positions(self) -> dict:
        try:
            if self.api is None:
                return {}
                
            positions = self.api.list_positions()
            return {pos.symbol: {
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'current_price': float(pos.current_price),
                'stop_loss': float(pos.current_price) * 0.95  # Default stop loss
            } for pos in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

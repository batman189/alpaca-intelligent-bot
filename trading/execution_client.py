import logging
from alpaca_trade_api import REST
from config.settings import settings

logger = logging.getLogger(__name__)

class ExecutionClient:
    def __init__(self):
        self.api = REST(settings.APCA_API_KEY_ID, 
                       settings.APCA_API_SECRET_KEY, 
                       settings.APCA_API_BASE_URL)
        
    def place_option_order(self, symbol: str, quantity: int, order_type: str = 'call',
                         strike: float = None, expiration: str = None) -> bool:
        if not settings.ENABLE_TRADING:
            logger.info(f"Would place {order_type} order for {quantity} {symbol}")
            return True
            
        try:
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
        if not settings.ENABLE_TRADING:
            logger.info(f"Would close position for {symbol}")
            return True
            
        try:
            self.api.close_position(symbol)
            logger.info(f"Position closed for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
            
    def get_current_positions(self) -> dict:
        try:
            positions = self.api.list_positions()
            return {pos.symbol: float(pos.market_value) for pos in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

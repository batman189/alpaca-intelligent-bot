import logging
import alpaca_trade_api as tradeapi
from typing import Dict
import os

logger = logging.getLogger(__name__)

class ExecutionClient:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            base_url=os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
    def get_current_positions(self) -> Dict:
        """Get current positions from Alpaca"""
        try:
            positions = self.api.list_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
            
    def place_option_order(self, symbol: str, quantity: int, order_type: str, 
                          strike: float, expiration: str) -> bool:
        """
        Place an options order with Alpaca
        NOW: Uses the exact option symbol from Alpaca's data (no formatting needed)
        """
        try:
            # The 'symbol' parameter now contains the EXACT option symbol from Alpaca
            # No need to format it - it's already in the correct format
            
            # Convert order type to side
            side = 'buy' if order_type.lower() == 'call' else 'sell'
            
            logger.info(f"Placing options order: {side.upper()} {quantity} contracts of {symbol}")
            logger.info(f"Option details: {order_type} ${strike} expiring {expiration}")
            
            # Place the order using the exact symbol from Alpaca
            order = self.api.submit_order(
                symbol=symbol,  # This is the exact symbol from Alpaca's options chain
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day',
                order_class='simple'
            )
            
            logger.info(f"Options order placed successfully: {order.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing option order: {e}")
            return False
            
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            logger.info(f"Closing position: {symbol}")
            self.api.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position: {symbol}: {e}")
            return False

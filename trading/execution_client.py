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
        Uses the exact option symbol from Alpaca's options chain
        """
        try:
            # The 'symbol' parameter contains the EXACT option symbol from Alpaca
            # Convert order type to side
            side = 'buy' if order_type.lower() == 'call' else 'sell'
            
            logger.info(f"Placing options order: {side.upper()} {quantity} contracts of {symbol}")
            logger.info(f"Option details: {order_type} ${strike} expiring {expiration}")
            
            # Verify the option exists before placing order
            try:
                asset = self.api.get_asset(symbol)
                logger.info(f"Option asset verified: {asset.symbol}")
            except Exception as e:
                logger.error(f"Option asset verification failed: {e}")
                return False
            
            # Place the order using the exact symbol from Alpaca
            order = self.api.submit_order(
                symbol=symbol,
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

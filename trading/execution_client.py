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
        self.day_trade_count = 0  # Track day trades to prevent PDT violations
        
    def get_current_positions(self) -> Dict:
        """Get current positions from Alpaca - FRESH DATA EVERY TIME"""
        try:
            positions = self.api.list_positions()
            # Return simple quantity mapping - no cached data
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}  # Return empty dict on error, not cached data
            
    def place_option_order(self, symbol: str, quantity: int, order_type: str, 
                          strike: float, expiration: str) -> bool:
        """
        Place an options order with Alpaca with safety checks
        """
        try:
            # Convert order type to side
            side = 'buy' if order_type.lower() == 'call' else 'sell'
            
            logger.info(f"Attempting options order: {side.upper()} {quantity} contracts of {symbol}")
            
            # SAFETY CHECK: Verify we have buying power
            try:
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                option_cost = quantity * 100 * strike  # Rough estimate
                
                if option_cost > buying_power * 0.5:  # Don't use more than 50% BP
                    logger.error(f"Insufficient buying power: ${option_cost:.2f} > 50% of ${buying_power:.2f}")
                    return False
            except Exception as e:
                logger.error(f"Failed to check buying power: {e}")
                return False
            
            # Place the order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day',
                order_class='simple'
            )
            
            logger.info(f"Options order placed successfully: {order.id}")
            
            # Track day trades
            if order.side == 'sell' and quantity > 0:
                self.day_trade_count += 1
                logger.info(f"Day trade count: {self.day_trade_count}/3")
                
            return True
            
        except Exception as e:
            logger.error(f"Error placing option order: {e}")
            return False
            
    def close_position(self, symbol: str) -> bool:
        """Close a position - ONLY IF IT EXISTS"""
        try:
            # Verify position actually exists before trying to close
            positions = self.api.list_positions()
            position_exists = any(pos.symbol == symbol for pos in positions)
            
            if not position_exists:
                logger.warning(f"Cannot close {symbol} - no position exists")
                return False
                
            logger.info(f"Closing position: {symbol}")
            self.api.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            
            # Track day trade
            self.day_trade_count += 1
            logger.info(f"Day trade count: {self.day_trade_count}/3")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
            
    def get_day_trade_count(self) -> int:
        """Get current day trade count"""
        return self.day_trade_count

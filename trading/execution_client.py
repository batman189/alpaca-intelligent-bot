import logging
import alpaca_trade_api as tradeapi
from typing import Dict
import os
from datetime import datetime

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
        FIXED: Correct option symbol formatting for Alpaca
        """
        try:
            # Convert order type to side
            side = 'buy' if order_type.lower() == 'call' else 'sell'
            
            # Build the option symbol properly for Alpaca
            # Correct format: {SYMBOL}{EXPIRATION}{TYPE}{STRIKE}
            # Example: AAPL240321C00150000 (AAPL, Mar 21 2024, Call, $150 strike)
            
            # Format expiration from YYYY-MM-DD to YYMMDD
            expiration_date = datetime.strptime(expiration, '%Y-%m-%d')
            expiration_formatted = expiration_date.strftime('%y%m%d')
            
            # Format strike price correctly (convert to integer and pad to 8 digits)
            # Strike price needs to be in cents (e.g., $150.00 = 15000)
            strike_cents = int(strike * 100)
            strike_formatted = f"{strike_cents:08d}"  # Pad to 8 digits with leading zeros
            
            # Build the option symbol
            option_type_char = 'C' if order_type.lower() == 'call' else 'P'
            option_symbol = f"{symbol}{expiration_formatted}{option_type_char}{strike_formatted}"
            
            logger.info(f"Placing options order: {side.upper()} {quantity} contracts of {option_symbol}")
            logger.info(f"Option details: {symbol} {expiration} {order_type} ${strike}")
            
            # Place the order
            order = self.api.submit_order(
                symbol=option_symbol,
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

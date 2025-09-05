import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self):
        self.max_portfolio_risk = 0.1  # Maximum 10% portfolio risk
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        
    def should_enter_trade(self, symbol: str, confidence: float, current_positions: Dict) -> bool:
        """
        Determine if we should enter a new trade based on current positions and risk
        """
        try:
            # Check if we already have a position in this symbol
            if symbol in current_positions:
                logger.info(f"Skipping {symbol} - already have a position")
                return False
                
            # Check if we have too many positions already
            if len(current_positions) >= 5:  # Max 5 simultaneous positions
                logger.info(f"Skipping {symbol} - maximum positions reached ({len(current_positions)})")
                return False
                
            # Confidence check
            if confidence < 0.6:
                logger.info(f"Skipping {symbol} - confidence too low ({confidence:.2f})")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in should_enter_trade for {symbol}: {e}")
            return False
            
    def manage_risk(self, current_positions: Dict, market_data: Dict) -> List[Dict]:
        """
        Manage risk on existing positions and generate exit signals
        FIXED: Properly handle position data structure
        """
        exit_signals = []
        
        try:
            for symbol, position_data in current_positions.items():
                # FIX: Check if position_data is a dictionary or just quantity
                if isinstance(position_data, dict):
                    # Position data is a dictionary with details
                    quantity = position_data.get('qty', 0)
                    entry_price = position_data.get('entry_price', 0)
                    current_value = position_data.get('current_value', 0)
                else:
                    # Position data is just the quantity (float)
                    quantity = position_data
                    entry_price = 0  # Unknown without more data
                    current_value = 0
                
                # Skip if no position or unable to get market data
                if quantity == 0 or symbol not in market_data or market_data[symbol] is None:
                    continue
                    
                # Get current price from market data
                current_price = market_data[symbol]['close'].iloc[-1] if hasattr(market_data[symbol], 'iloc') else 100
                
                # Calculate P&L if we have entry price
                if entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Check for stop loss
                    if pnl_pct <= -self.stop_loss_pct:
                        exit_signals.append({
                            'symbol': symbol,
                            'reason': 'stop_loss',
                            'pnl_pct': pnl_pct
                        })
                    
                    # Check for take profit
                    elif pnl_pct >= self.take_profit_pct:
                        exit_signals.append({
                            'symbol': symbol,
                            'reason': 'take_profit',
                            'pnl_pct': pnl_pct
                        })
                
                # Simple exit rule: if we have any position, consider exit based on other factors
                # This is a placeholder for more sophisticated risk management
                exit_signals.append({
                    'symbol': symbol,
                    'reason': 'risk_management',
                    'pnl_pct': 0
                })
                
        except Exception as e:
            logger.error(f"Error in manage_risk: {e}")
            
        return exit_signals

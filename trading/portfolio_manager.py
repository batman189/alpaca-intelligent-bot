import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self):
        self.max_portfolio_risk = 0.1  # Maximum 10% portfolio risk
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        self.max_day_trades = 2  # Max 2 day trades to avoid PDT limit
        
    def should_enter_trade(self, symbol: str, confidence: float, current_positions: Dict, day_trade_count: int) -> bool:
        """
        Determine if we should enter a new trade with comprehensive checks
        """
        try:
            # 1. Check PDT limits
            if day_trade_count >= self.max_day_trades:
                logger.info(f"Skipping {symbol} - PDT limit reached ({day_trade_count}/{self.max_day_trades})")
                return False
                
            # 2. Check if we already have a position
            if symbol in current_positions and current_positions[symbol] > 0:
                logger.info(f"Skipping {symbol} - already have a position")
                return False
                
            # 3. Check position count limit
            if len([p for p in current_positions.values() if p > 0]) >= 3:  # Max 3 positions
                logger.info(f"Skipping {symbol} - maximum positions reached")
                return False
                
            # 4. Confidence check
            if confidence < 0.65:
                logger.info(f"Skipping {symbol} - confidence too low ({confidence:.2f})")
                return False
                
            # 5. Avoid wash trades - don't trade the same symbol repeatedly
            # This is a simple prevention - could be enhanced
            if symbol in self.get_recently_traded_symbols():
                logger.info(f"Skipping {symbol} - recently traded (wash trade prevention)")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in should_enter_trade for {symbol}: {e}")
            return False
            
    def manage_risk(self, current_positions: Dict, market_data: Dict) -> List[Dict]:
        """
        Manage risk on existing positions - SIMPLIFIED to prevent errors
        """
        exit_signals = []
        
        try:
            for symbol, quantity in current_positions.items():
                if quantity <= 0:
                    continue  # Skip if no position
                    
                # Simple exit rule: close all positions at end of day to avoid PDT
                # This is a conservative approach until we have better risk management
                exit_signals.append({
                    'symbol': symbol,
                    'reason': 'end_of_day_closeout',
                    'quantity': quantity
                })
                
        except Exception as e:
            logger.error(f"Error in manage_risk: {e}")
            
        return exit_signals
        
    def get_recently_traded_symbols(self) -> List[str]:
        """
        Get list of symbols traded recently to avoid wash trades
        """
        # This would ideally track trade history, but for now return empty
        return []
        
    def check_pdt_limits(self, day_trade_count: int) -> bool:
        """Check if we're approaching PDT limits"""
        return day_trade_count < self.max_day_trades

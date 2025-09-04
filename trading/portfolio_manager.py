import numpy as np
import logging
from typing import Dict, List
from config.settings import settings

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self):
        self.current_positions = {}
        self.pending_orders = []
        
    def calculate_position_size(self, account_equity: float, entry_price: float, 
                              stop_loss_price: float, confidence: float) -> int:
        if not all([account_equity, entry_price, stop_loss_price]):
            return 0
            
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return 0
            
        max_risk_amount = account_equity * settings.RISK_PER_TRADE * confidence
        shares = max_risk_amount / risk_per_share
        
        max_shares = (account_equity * settings.MAX_PORTFOLIO_RISK) / entry_price
        shares = min(shares, max_shares)
        
        return int(max(1, shares))
        
    def determine_stop_loss(self, entry_price: float, volatility: float, 
                          pattern_type: str = None) -> float:
        atr_based_stop = entry_price * (1 - 2 * volatility)
        
        if pattern_type == 'breakout':
            stop_loss = entry_price * 0.98
        elif pattern_type == 'trend':
            stop_loss = atr_based_stop
        else:
            stop_loss = entry_price * 0.95
            
        return max(stop_loss, entry_price * 0.90)
        
    def should_enter_trade(self, symbol: str, confidence: float, 
                         current_positions: Dict) -> bool:
        if confidence < settings.MIN_CONFIDENCE:
            return False
            
        if symbol in current_positions:
            logger.info(f"Already in position for {symbol}")
            return False
            
        if len(current_positions) >= 5:
            logger.info("Maximum positions reached")
            return False
            
        return True
        
    def manage_risk(self, current_positions: Dict, market_data: Dict) -> List:
        exit_signals = []
        
        for symbol, position in current_positions.items():
            current_price = market_data.get(symbol, {}).get('close', 0)
            if current_price <= position['stop_loss']:
                exit_signals.append({
                    'symbol': symbol,
                    'reason': 'stop_loss',
                    'price': current_price
                })
                
        return exit_signals

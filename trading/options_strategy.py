import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class OptionsStrategyEngine:
    def __init__(self):
        self.strategies = {
            'long_call': self.long_call_strategy,
            'long_put': self.long_put_strategy,
            'credit_spread': self.credit_spread_strategy,
            'iron_condor': self.iron_condor_strategy
        }
    
    def select_best_option(self, symbol, prediction, confidence, option_chain):
        """Select the best option contract based on strategy"""
        if prediction == 1:  # Bullish
            return self.select_call_option(symbol, confidence, option_chain)
        else:  # Bearish
            return self.select_put_option(symbol, confidence, option_chain)
    
    def select_call_option(self, symbol, confidence, option_chain):
        """Select optimal call option"""
        if not option_chain:
            return None
            
        # Filter for calls only
        calls = [opt for opt in option_chain if opt['type'] == 'call']
        
        # Strategy: Higher confidence -> closer to ATM, lower confidence -> OTM
        if confidence > 0.7:
            # ATM calls for high confidence
            selected = min(calls, key=lambda x: abs(x['strike'] - self.get_current_price(symbol)))
        else:
            # OTM calls for lower confidence (higher leverage)
            selected = min(calls, key=lambda x: x['strike'])
            
        return selected
    
    def select_put_option(self, symbol, confidence, option_chain):
        """Select optimal put option"""
        if not option_chain:
            return None
            
        # Filter for puts only
        puts = [opt for opt in option_chain if opt['type'] == 'put']
        
        if confidence > 0.7:
            # ATM puts for high confidence
            selected = min(puts, key=lambda x: abs(x['strike'] - self.get_current_price(symbol)))
        else:
            # OTM puts for lower confidence
            selected = max(puts, key=lambda x: x['strike'])
            
        return selected
    
    def calculate_options_position_size(self, account_equity, option_price, confidence):
        """Calculate position size for options (contracts)"""
        risk_amount = account_equity * 0.02 * confidence  # 2% risk adjusted by confidence
        contracts = risk_amount / (option_price * 100)  # Option price is per share
        return max(1, int(contracts))

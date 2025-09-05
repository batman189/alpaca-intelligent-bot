import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OptionsStrategyEngine:
    def __init__(self):
        # Minimum volume threshold to ensure liquidity
        self.min_volume = 10  
        # Minimum open interest threshold
        self.min_open_interest = 50  
        # Target delta ranges based on confidence
        self.delta_targets = {
            'high_confidence': {'min': 0.70, 'max': 0.85},    # For confidence > 0.8
            'medium_confidence': {'min': 0.60, 'max': 0.75}, # For confidence 0.65-0.8
            'low_confidence': {'min': 0.45, 'max': 0.65}     # For confidence < 0.65
        }

    def filter_liquid_options(self, option_chain: List[Dict]) -> List[Dict]:
        """Filter out illiquid options with low volume and open interest"""
        if not option_chain:  # FIXED: Use 'not option_chain' instead of '.empty'
            return []
        
        # Filter for options that meet minimum liquidity requirements
        liquid_options = [
            option for option in option_chain
            if option.get('volume', 0) >= self.min_volume 
            and option.get('open_interest', 0) >= self.min_open_interest
        ]
        
        logger.debug(f"Filtered from {len(option_chain)} to {len(liquid_options)} liquid options")
        return liquid_options

    def get_delta_target(self, confidence: float) -> dict:
        """Get target delta range based on prediction confidence"""
        if confidence > 0.8:
            return self.delta_targets['high_confidence']
        elif confidence > 0.65:
            return self.delta_targets['medium_confidence']
        else:
            return self.delta_targets['low_confidence']

    def rank_options(self, options: List[Dict], option_type: str, 
                    current_price: float, confidence: float) -> List[Dict]:
        """Rank options based on liquidity, delta fit, and moneyness"""
        if not options:  # FIXED: Use 'not options' instead of '.empty'
            return []
        
        scored_options = []
        delta_target = self.get_delta_target(confidence)
        target_min_delta = delta_target['min']
        target_max_delta = delta_target['max']
        
        for option in options:
            # Calculate absolute distance from current price
            price_distance_pct = abs(option['strike'] - current_price) / current_price
            
            # Liquidity score
            liquidity_score = np.log1p(option.get('volume', 1) * option.get('open_interest', 1))
            
            # Delta fitness score (how close to our target range)
            delta = option.get('delta', 0.5)
            delta_fitness = 0
            if target_min_delta <= delta <= target_max_delta:
                delta_fitness = 1 - (2 * abs(delta - (target_min_delta + target_max_delta)/2))
            
            # Moneyness score (prefer slightly OTM for better risk/reward)
            if option_type == 'call':
                moneyness_score = 0.7 if option['strike'] > current_price else 0.3
            else:  # put
                moneyness_score = 0.7 if option['strike'] < current_price else 0.3
            
            # Combined score (weighted factors)
            combined_score = (
                liquidity_score * 0.4 +
                delta_fitness * 0.4 +
                moneyness_score * 0.2
            )
            
            scored_options.append({
                **option,
                'liquidity_score': liquidity_score,
                'delta_fitness': delta_fitness,
                'moneyness_score': moneyness_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score descending
        return sorted(scored_options, key=lambda x: x['combined_score'], reverse=True)

    def select_best_option(self, symbol: str, prediction: int, confidence: float, 
                          option_chain: List[Dict], current_price: float) -> Optional[Dict]:
        """
        Select the best option contract based on multiple factors:
        - Liquidity (volume + open interest)
        - Delta (matches confidence level)
        - Moneyness (prefer slightly OTM for better risk/reward)
        """
        try:
            if not option_chain:  # FIXED: Use 'not option_chain' instead of '.empty'
                logger.warning(f"No option chain data for {symbol}")
                return None
            
            # Determine option type based on prediction
            option_type = 'call' if prediction == 1 else 'put'
            
            # Filter for the right type and liquid options
            type_options = [opt for opt in option_chain if opt.get('type') == option_type]
            liquid_options = self.filter_liquid_options(type_options)
            
            if not liquid_options:  # FIXED: Use 'not liquid_options' instead of '.empty'
                logger.warning(f"No liquid {option_type} options found for {symbol}")
                return None
            
            # Rank options based on our criteria
            ranked_options = self.rank_options(liquid_options, option_type, current_price, confidence)
            
            if not ranked_options:  # FIXED: Use 'not ranked_options' instead of '.empty'
                logger.warning(f"No suitable {option_type} options found for {symbol} after ranking")
                return None
            
            # Select the top-ranked option
            best_option = ranked_options[0]
            
            return best_option
            
        except Exception as e:
            logger.error(f"Error selecting best option for {symbol}: {e}")
            return None

    def calculate_options_position_size(self, account_equity: float, option_price: float, 
                                      confidence: float) -> int:
        """
        Calculate number of contracts based on:
        - Account equity
        - Option price
        - Confidence level
        """
        # Base risk per trade (1% of account equity)
        base_risk_amount = account_equity * 0.01
        
        # Adjust risk based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5-1.0 range
        adjusted_risk = base_risk_amount * confidence_multiplier
        
        # Calculate number of contracts (option price × 100 shares per contract)
        risk_per_contract = option_price * 100
        if risk_per_contract <= 0:
            return 1
        
        contracts = int(adjusted_risk / risk_per_contract)
        
        # Ensure at least 1 contract for any valid trade
        return max(1, contracts)

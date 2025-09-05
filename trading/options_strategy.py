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

    def filter_liquid_options(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """Filter out illiquid options with low volume and open interest"""
        if option_chain is None or option_chain.empty:
            return pd.DataFrame()
        
        # Filter for options that meet minimum liquidity requirements
        liquid_options = option_chain[
            (option_chain['volume'] >= self.min_volume) & 
            (option_chain['open_interest'] >= self.min_open_interest)
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

    def rank_options(self, options: pd.DataFrame, option_type: str, 
                    current_price: float, confidence: float) -> pd.DataFrame:
        """Rank options based on liquidity, delta fit, and moneyness"""
        if options.empty:
            return pd.DataFrame()
        
        # Calculate absolute distance from current price
        options['price_distance_pct'] = abs(options['strike'] - current_price) / current_price
        
        # Get target delta range for our confidence level
        delta_target = self.get_delta_target(confidence)
        target_min_delta = delta_target['min']
        target_max_delta = delta_target['max']
        
        # Score each option based on multiple factors
        options['liquidity_score'] = np.log1p(options['volume'] * options['open_interest'])
        
        # Delta fitness score (how close to our target range)
        options['delta_fitness'] = 0
        in_range_mask = (options['delta'] >= target_min_delta) & (options['delta'] <= target_max_delta)
        options.loc[in_range_mask, 'delta_fitness'] = 1 - (2 * abs(options['delta'] - 
                                                                 (target_min_delta + target_max_delta)/2))
        
        # Moneyness score (prefer slightly OTM for better risk/reward)
        if option_type == 'call':
            options['moneyness_score'] = np.where(
                options['strike'] > current_price, 
                0.7,  # Prefer OTM calls for cheaper premium
                0.3    # Discourage ITM calls as they're more expensive
            )
        else:  # put
            options['moneyness_score'] = np.where(
                options['strike'] < current_price,
                0.7,  # Prefer OTM puts for cheaper premium
                0.3    # Discourage ITM puts
            )
        
        # Combined score (weighted factors)
        options['combined_score'] = (
            options['liquidity_score'] * 0.4 +
            options['delta_fitness'] * 0.4 +
            options['moneyness_score'] * 0.2
        )
        
        return options.sort_values('combined_score', ascending=False)

    def select_best_option(self, symbol: str, prediction: int, confidence: float, 
                          option_chain: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Select the best option contract based on multiple factors:
        - Liquidity (volume + open interest)
        - Delta (matches confidence level)
        - Moneyness (prefer slightly OTM for better risk/reward)
        """
        try:
            if option_chain is None or option_chain.empty:
                logger.warning(f"No option chain data for {symbol}")
                return None
            
            # Determine option type based on prediction
            option_type = 'call' if prediction == 1 else 'put'
            
            # Filter for the right type and liquid options
            type_options = option_chain[option_chain['type'] == option_type]
            liquid_options = self.filter_liquid_options(type_options)
            
            if liquid_options.empty:
                logger.warning(f"No liquid {option_type} options found for {symbol}")
                return None
            
            # Rank options based on our criteria
            ranked_options = self.rank_options(liquid_options, option_type, current_price, confidence)
            
            if ranked_options.empty:
                logger.warning(f"No suitable {option_type} options found for {symbol} after ranking")
                return None
            
            # Select the top-ranked option
            best_option = ranked_options.iloc[0]
            best_contract = {
                'symbol': best_option['symbol'],
                'type': option_type,
                'strike': float(best_option['strike']),
                'expiration': best_option['expiration'],
                'price': float(best_option['price']),
                'delta': float(best_option['delta']),
                'volume': int(best_option['volume']),
                'open_interest': int(best_option['open_interest']),
                'score': float(best_option['combined_score'])
            }
            
            logger.info(f"Selected {option_type.upper()} for {symbol}: "
                       f"Strike ${best_contract['strike']}, "
                       f"Delta {best_contract['delta']:.2f}, "
                       f"Volume {best_contract['volume']}")
            
            return best_contract
            
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

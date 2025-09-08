"""
Professional Options Trading Engine
Advanced options strategies, Greeks calculation, and volatility analysis
Designed to compete with 15,000% gains through intelligent options trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy.stats import norm
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class StrategyType(Enum):
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    CALL_SPREAD = "call_spread"
    PUT_SPREAD = "put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    VOLATILITY_CRUSH = "volatility_crush"

@dataclass
class OptionContract:
    """Options contract representation"""
    symbol: str
    strike: float
    expiration: datetime
    option_type: OptionType
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    @property
    def days_to_expiration(self) -> int:
        return (self.expiration - datetime.now()).days
    
    @property
    def time_to_expiration(self) -> float:
        return self.days_to_expiration / 365.0
    
    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money"""
        if self.option_type == OptionType.CALL:
            return self.strike < self.price
        else:
            return self.strike > self.price

@dataclass
class OptionsPosition:
    """Options trading position"""
    contracts: List[OptionContract]
    quantities: List[int]  # Positive for long, negative for short
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_type: StrategyType = StrategyType.LONG_CALL
    
    @property
    def total_delta(self) -> float:
        return sum(contract.delta * qty for contract, qty in zip(self.contracts, self.quantities))
    
    @property
    def total_gamma(self) -> float:
        return sum(contract.gamma * qty for contract, qty in zip(self.contracts, self.quantities))
    
    @property
    def total_theta(self) -> float:
        return sum(contract.theta * qty for contract, qty in zip(self.contracts, self.quantities))
    
    @property
    def total_vega(self) -> float:
        return sum(contract.vega * qty for contract, qty in zip(self.contracts, self.quantities))

@dataclass
class MarketConditions:
    """Current market conditions for options analysis"""
    underlying_price: float
    volatility: float
    trend_strength: float  # -1 to 1
    volume_surge: bool
    earnings_approaching: bool
    technical_breakout: bool
    support_resistance_levels: List[float]
    implied_volatility_rank: float  # 0 to 1

class BlackScholes:
    """Black-Scholes options pricing model"""
    
    @staticmethod
    def calculate_option_price(
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: OptionType
    ) -> float:
        """Calculate theoretical option price"""
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)
    
    @staticmethod
    def calculate_greeks(
        S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
    ) -> Dict[str, float]:
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (per day)
        theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                     r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        theta_put = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        theta = theta_call if option_type == OptionType.CALL else theta_put
        
        # Vega (per 1% change in volatility)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho (per 1% change in interest rate)
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class ProfessionalOptionsEngine:
    """
    Professional options trading engine with advanced strategies
    Designed to identify high-probability, high-reward options opportunities
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        self.positions: List[OptionsPosition] = []
        self.strategy_performance = {strategy: {'wins': 0, 'losses': 0, 'total_pnl': 0} 
                                   for strategy in StrategyType}
        
        # Strategy parameters optimized for maximum gains
        self.strategy_config = {
            StrategyType.MOMENTUM_BREAKOUT: {
                'min_volume_surge': 2.0,  # 2x average volume
                'min_price_move': 0.03,   # 3% price move
                'max_time_to_expiration': 30,  # 30 days max
                'target_delta': 0.6,      # High delta for momentum
                'risk_reward_ratio': 3.0   # 3:1 minimum
            },
            StrategyType.VOLATILITY_CRUSH: {
                'min_iv_rank': 0.8,       # High IV rank
                'earnings_buffer': 2,     # Days after earnings
                'max_days_to_expiration': 45,
                'target_credit': 0.3      # 30% of spread width
            },
            StrategyType.STRADDLE: {
                'min_expected_move': 0.05,  # 5% expected move
                'max_days_to_expiration': 21,
                'min_iv_rank': 0.3
            },
            StrategyType.IRON_CONDOR: {
                'min_iv_rank': 0.6,
                'target_probability': 0.8,  # 80% probability of profit
                'max_days_to_expiration': 45
            }
        }
        
        self.logger.info("Professional Options Engine initialized")
    
    def analyze_market_opportunity(self, 
                                 symbol: str, 
                                 market_data: pd.DataFrame,
                                 options_chain: Dict,
                                 market_conditions: MarketConditions) -> List[Dict]:
        """
        Analyze market for high-probability options opportunities
        Returns ranked list of strategies with expected returns
        """
        opportunities = []
        current_price = market_conditions.underlying_price
        
        try:
            # 1. Momentum Breakout Strategy (for explosive moves like your classmates caught)
            momentum_ops = self._analyze_momentum_breakout(
                symbol, market_data, options_chain, market_conditions
            )
            opportunities.extend(momentum_ops)
            
            # 2. Volatility Expansion Strategy (for earnings and events)
            vol_expansion_ops = self._analyze_volatility_expansion(
                symbol, options_chain, market_conditions
            )
            opportunities.extend(vol_expansion_ops)
            
            # 3. Technical Breakout Strategy (support/resistance breaks)
            technical_ops = self._analyze_technical_breakout(
                symbol, market_data, options_chain, market_conditions
            )
            opportunities.extend(technical_ops)
            
            # 4. Volatility Crush Strategy (post-earnings)
            vol_crush_ops = self._analyze_volatility_crush(
                symbol, options_chain, market_conditions
            )
            opportunities.extend(vol_crush_ops)
            
            # 5. Gap Fill Strategy (for overnight gaps)
            gap_ops = self._analyze_gap_strategy(
                symbol, market_data, options_chain, market_conditions
            )
            opportunities.extend(gap_ops)
            
            # Rank opportunities by expected return and probability
            opportunities = self._rank_opportunities(opportunities)
            
            return opportunities[:5]  # Return top 5 opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing market opportunity for {symbol}: {e}")
            return []
    
    def _analyze_momentum_breakout(self, 
                                 symbol: str,
                                 market_data: pd.DataFrame,
                                 options_chain: Dict,
                                 market_conditions: MarketConditions) -> List[Dict]:
        """Analyze momentum breakout opportunities (like TSLA 4% swings)"""
        opportunities = []
        
        try:
            if len(market_data) < 20:
                return opportunities
            
            current_price = market_conditions.underlying_price
            latest_bar = market_data.iloc[-1]
            
            # Check for momentum conditions
            volume_surge = market_conditions.volume_surge
            price_change = abs(latest_bar['close'] - latest_bar['open']) / latest_bar['open']
            trend_strength = market_conditions.trend_strength
            
            config = self.strategy_config[StrategyType.MOMENTUM_BREAKOUT]
            
            if (volume_surge and 
                price_change >= config['min_price_move'] and
                abs(trend_strength) >= 0.6):
                
                # Determine direction
                direction = 1 if trend_strength > 0 else -1
                
                # Find optimal strikes for momentum play
                optimal_strikes = self._find_momentum_strikes(
                    current_price, direction, options_chain, config
                )
                
                for strike_info in optimal_strikes:
                    opportunity = {
                        'strategy': StrategyType.MOMENTUM_BREAKOUT,
                        'symbol': symbol,
                        'direction': direction,
                        'contracts': strike_info['contracts'],
                        'expected_return': strike_info['expected_return'],
                        'probability': strike_info['probability'],
                        'risk_amount': strike_info['risk'],
                        'max_profit': strike_info['max_profit'],
                        'breakeven': strike_info['breakeven'],
                        'confidence': 0.8 if volume_surge else 0.6,
                        'reasoning': f"Momentum breakout detected: {price_change:.1%} move with {trend_strength:.1f} trend strength"
                    }
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error in momentum breakout analysis: {e}")
            return []
    
    def _find_momentum_strikes(self, 
                             current_price: float, 
                             direction: int,
                             options_chain: Dict,
                             config: Dict) -> List[Dict]:
        """Find optimal strikes for momentum strategy"""
        strikes = []
        
        try:
            option_type = OptionType.CALL if direction > 0 else OptionType.PUT
            chain_key = 'calls' if direction > 0 else 'puts'
            
            if chain_key not in options_chain:
                return strikes
            
            for expiration, contracts in options_chain[chain_key].items():
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                
                if days_to_exp > config['max_time_to_expiration']:
                    continue
                
                for strike, contract_data in contracts.items():
                    strike_price = float(strike)
                    
                    # Calculate target delta
                    if direction > 0:  # Calls
                        moneyness = strike_price / current_price
                        if moneyness < 0.95 or moneyness > 1.15:  # Focus on ATM to slightly OTM
                            continue
                    else:  # Puts
                        moneyness = current_price / strike_price
                        if moneyness < 0.95 or moneyness > 1.15:
                            continue
                    
                    # Create option contract
                    contract = OptionContract(
                        symbol=contract_data.get('symbol', ''),
                        strike=strike_price,
                        expiration=exp_date,
                        option_type=option_type,
                        price=contract_data.get('last_price', 0),
                        bid=contract_data.get('bid', 0),
                        ask=contract_data.get('ask', 0),
                        volume=contract_data.get('volume', 0),
                        implied_volatility=contract_data.get('implied_volatility', 0.3)
                    )
                    
                    # Calculate Greeks
                    greeks = BlackScholes.calculate_greeks(
                        current_price, strike_price, days_to_exp/365, 
                        self.risk_free_rate, contract.implied_volatility, option_type
                    )
                    
                    contract.delta = greeks['delta']
                    contract.gamma = greeks['gamma']
                    contract.theta = greeks['theta']
                    contract.vega = greeks['vega']
                    
                    # Check if delta meets requirements
                    if abs(contract.delta) < config['target_delta']:
                        continue
                    
                    # Calculate expected return for momentum move
                    expected_move = 0.08 * direction  # Expect 8% move in direction
                    new_price = current_price * (1 + expected_move)
                    
                    option_value_at_target = BlackScholes.calculate_option_price(
                        new_price, strike_price, max(days_to_exp-1, 0)/365,
                        self.risk_free_rate, contract.implied_volatility * 0.9, option_type
                    )
                    
                    profit = option_value_at_target - contract.ask
                    risk = contract.ask
                    
                    if profit > 0 and profit/risk >= config['risk_reward_ratio']:
                        strike_info = {
                            'contracts': [contract],
                            'expected_return': profit / risk,
                            'probability': self._calculate_breakout_probability(
                                current_price, strike_price, expected_move, option_type
                            ),
                            'risk': risk,
                            'max_profit': profit,
                            'breakeven': self._calculate_breakeven(contract, option_type)
                        }
                        strikes.append(strike_info)
            
            # Sort by expected return
            strikes.sort(key=lambda x: x['expected_return'], reverse=True)
            return strikes[:3]  # Return top 3 strikes
            
        except Exception as e:
            self.logger.error(f"Error finding momentum strikes: {e}")
            return []
    
    def _calculate_breakout_probability(self, 
                                      current_price: float,
                                      strike_price: float,
                                      expected_move: float,
                                      option_type: OptionType) -> float:
        """Calculate probability of profitable breakout"""
        try:
            target_price = current_price * (1 + expected_move)
            
            if option_type == OptionType.CALL:
                # Probability that stock closes above strike
                distance_to_strike = (strike_price - current_price) / current_price
                move_required = distance_to_strike
            else:
                # Probability that stock closes below strike
                distance_to_strike = (current_price - strike_price) / current_price
                move_required = -distance_to_strike
            
            # Simple probability model based on expected move
            if abs(expected_move) > abs(move_required):
                base_prob = 0.7
                excess_move = abs(expected_move) - abs(move_required)
                bonus_prob = min(excess_move * 2, 0.2)
                return min(base_prob + bonus_prob, 0.9)
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return 0.5
    
    def _calculate_breakeven(self, contract: OptionContract, option_type: OptionType) -> float:
        """Calculate breakeven price for option"""
        if option_type == OptionType.CALL:
            return contract.strike + contract.ask
        else:
            return contract.strike - contract.ask
    
    def _analyze_volatility_expansion(self,
                                    symbol: str,
                                    options_chain: Dict,
                                    market_conditions: MarketConditions) -> List[Dict]:
        """Analyze volatility expansion opportunities (before earnings/events)"""
        opportunities = []
        
        try:
            current_price = market_conditions.underlying_price
            
            # Check if volatility expansion is likely
            if (market_conditions.earnings_approaching and 
                market_conditions.implied_volatility_rank < 0.5):
                
                # Look for straddles and strangles
                straddle_ops = self._find_straddle_opportunities(
                    symbol, current_price, options_chain, market_conditions
                )
                opportunities.extend(straddle_ops)
                
                strangle_ops = self._find_strangle_opportunities(
                    symbol, current_price, options_chain, market_conditions
                )
                opportunities.extend(strangle_ops)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility expansion: {e}")
            return []
    
    def _find_straddle_opportunities(self,
                                   symbol: str,
                                   current_price: float,
                                   options_chain: Dict,
                                   market_conditions: MarketConditions) -> List[Dict]:
        """Find optimal straddle opportunities"""
        opportunities = []
        
        try:
            if 'calls' not in options_chain or 'puts' not in options_chain:
                return opportunities
            
            config = self.strategy_config[StrategyType.STRADDLE]
            
            for expiration in options_chain['calls'].keys():
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                
                if days_to_exp > config['max_days_to_expiration']:
                    continue
                
                # Find ATM strikes
                atm_strikes = self._find_atm_strikes(current_price, options_chain, expiration)
                
                for strike in atm_strikes:
                    call_data = options_chain['calls'][expiration].get(str(strike))
                    put_data = options_chain['puts'][expiration].get(str(strike))
                    
                    if not call_data or not put_data:
                        continue
                    
                    # Create contracts
                    call_contract = OptionContract(
                        symbol=call_data.get('symbol', ''),
                        strike=strike,
                        expiration=exp_date,
                        option_type=OptionType.CALL,
                        ask=call_data.get('ask', 0),
                        implied_volatility=call_data.get('implied_volatility', 0.3)
                    )
                    
                    put_contract = OptionContract(
                        symbol=put_data.get('symbol', ''),
                        strike=strike,
                        expiration=exp_date,
                        option_type=OptionType.PUT,
                        ask=put_data.get('ask', 0),
                        implied_volatility=put_data.get('implied_volatility', 0.3)
                    )
                    
                    total_cost = call_contract.ask + put_contract.ask
                    expected_move = total_cost / current_price  # Implied move
                    
                    if expected_move >= config['min_expected_move']:
                        opportunity = {
                            'strategy': StrategyType.STRADDLE,
                            'symbol': symbol,
                            'contracts': [call_contract, put_contract],
                            'expected_return': 2.0,  # Potential for 200% return
                            'probability': 0.6,
                            'risk_amount': total_cost,
                            'max_profit': float('inf'),  # Unlimited upside
                            'breakeven': [strike - total_cost, strike + total_cost],
                            'confidence': 0.7,
                            'reasoning': f"Volatility expansion play before earnings, implied move: {expected_move:.1%}"
                        }
                        opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding straddle opportunities: {e}")
            return []
    
    def _find_atm_strikes(self, current_price: float, options_chain: Dict, expiration: str) -> List[float]:
        """Find at-the-money strikes"""
        strikes = []
        
        try:
            if expiration in options_chain.get('calls', {}):
                available_strikes = [float(strike) for strike in options_chain['calls'][expiration].keys()]
                available_strikes.sort()
                
                # Find closest strikes to current price
                closest_strike = min(available_strikes, key=lambda x: abs(x - current_price))
                strikes.append(closest_strike)
                
                # Add neighboring strikes
                idx = available_strikes.index(closest_strike)
                if idx > 0:
                    strikes.append(available_strikes[idx - 1])
                if idx < len(available_strikes) - 1:
                    strikes.append(available_strikes[idx + 1])
            
            return strikes
            
        except Exception as e:
            self.logger.error(f"Error finding ATM strikes: {e}")
            return []
    
    def _find_strangle_opportunities(self,
                                   symbol: str,
                                   current_price: float,
                                   options_chain: Dict,
                                   market_conditions: MarketConditions) -> List[Dict]:
        """Find optimal strangle opportunities"""
        opportunities = []
        
        try:
            # Similar to straddle but with OTM strikes
            # Implementation would be similar to straddle but selecting OTM strikes
            # for lower cost and higher leverage
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding strangle opportunities: {e}")
            return []
    
    def _analyze_technical_breakout(self,
                                  symbol: str,
                                  market_data: pd.DataFrame,
                                  options_chain: Dict,
                                  market_conditions: MarketConditions) -> List[Dict]:
        """Analyze technical breakout opportunities"""
        opportunities = []
        
        try:
            if not market_conditions.technical_breakout:
                return opportunities
            
            current_price = market_conditions.underlying_price
            support_resistance = market_conditions.support_resistance_levels
            
            # Identify breakout direction
            if support_resistance:
                nearest_level = min(support_resistance, key=lambda x: abs(x - current_price))
                
                if current_price > nearest_level:
                    # Resistance breakout - bullish
                    direction = 1
                else:
                    # Support break - bearish
                    direction = -1
                
                # Find options for breakout direction
                breakout_ops = self._find_breakout_options(
                    symbol, current_price, direction, options_chain, nearest_level
                )
                opportunities.extend(breakout_ops)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical breakout: {e}")
            return []
    
    def _find_breakout_options(self,
                             symbol: str,
                             current_price: float,
                             direction: int,
                             options_chain: Dict,
                             key_level: float) -> List[Dict]:
        """Find options for technical breakout"""
        opportunities = []
        
        try:
            option_type = OptionType.CALL if direction > 0 else OptionType.PUT
            chain_key = 'calls' if direction > 0 else 'puts'
            
            if chain_key not in options_chain:
                return opportunities
            
            # Calculate target based on key level distance
            level_distance = abs(current_price - key_level) / current_price
            target_move = max(level_distance * 2, 0.05)  # At least 5% move expected
            
            for expiration, contracts in options_chain[chain_key].items():
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                
                if days_to_exp > 21:  # Focus on near-term options for breakouts
                    continue
                
                for strike, contract_data in contracts.items():
                    strike_price = float(strike)
                    
                    # Select strikes based on breakout target
                    if direction > 0:  # Calls
                        target_price = current_price * (1 + target_move)
                        if strike_price > target_price * 1.1:  # Too far OTM
                            continue
                    else:  # Puts
                        target_price = current_price * (1 - target_move)
                        if strike_price < target_price * 0.9:  # Too far OTM
                            continue
                    
                    contract = OptionContract(
                        symbol=contract_data.get('symbol', ''),
                        strike=strike_price,
                        expiration=exp_date,
                        option_type=option_type,
                        ask=contract_data.get('ask', 0),
                        volume=contract_data.get('volume', 0),
                        implied_volatility=contract_data.get('implied_volatility', 0.3)
                    )
                    
                    # Calculate potential profit
                    if direction > 0:
                        new_price = current_price * (1 + target_move)
                    else:
                        new_price = current_price * (1 - target_move)
                    
                    option_value_at_target = BlackScholes.calculate_option_price(
                        new_price, strike_price, max(days_to_exp-2, 0)/365,
                        self.risk_free_rate, contract.implied_volatility * 0.8, option_type
                    )
                    
                    profit = option_value_at_target - contract.ask
                    risk = contract.ask
                    
                    if profit > 0 and profit/risk >= 2.0:  # At least 2:1 risk/reward
                        opportunity = {
                            'strategy': StrategyType.MOMENTUM_BREAKOUT,
                            'symbol': symbol,
                            'direction': direction,
                            'contracts': [contract],
                            'expected_return': profit / risk,
                            'probability': 0.65,  # Technical breakouts have good probability
                            'risk_amount': risk,
                            'max_profit': profit,
                            'breakeven': self._calculate_breakeven(contract, option_type),
                            'confidence': 0.75,
                            'reasoning': f"Technical breakout from {key_level:.2f} level, target move: {target_move:.1%}"
                        }
                        opportunities.append(opportunity)
            
            return opportunities[:2]  # Return best 2 opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding breakout options: {e}")
            return []
    
    def _analyze_volatility_crush(self,
                                symbol: str,
                                options_chain: Dict,
                                market_conditions: MarketConditions) -> List[Dict]:
        """Analyze volatility crush opportunities (post-earnings)"""
        opportunities = []
        
        try:
            config = self.strategy_config[StrategyType.VOLATILITY_CRUSH]
            
            # Only consider if IV rank is high and earnings passed
            if (market_conditions.implied_volatility_rank >= config['min_iv_rank'] and
                not market_conditions.earnings_approaching):
                
                # Look for iron condors and credit spreads
                condor_ops = self._find_iron_condor_opportunities(
                    symbol, market_conditions.underlying_price, options_chain, config
                )
                opportunities.extend(condor_ops)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility crush: {e}")
            return []
    
    def _find_iron_condor_opportunities(self,
                                      symbol: str,
                                      current_price: float,
                                      options_chain: Dict,
                                      config: Dict) -> List[Dict]:
        """Find iron condor opportunities for volatility crush"""
        opportunities = []
        
        try:
            if 'calls' not in options_chain or 'puts' not in options_chain:
                return opportunities
            
            for expiration in options_chain['calls'].keys():
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                
                if days_to_exp > config['max_days_to_expiration']:
                    continue
                
                # Find strikes for iron condor (sell closer, buy further)
                condor_setups = self._build_iron_condor_setups(
                    current_price, options_chain, expiration
                )
                
                for setup in condor_setups:
                    net_credit = setup['credit']
                    max_loss = setup['max_loss']
                    probability = setup['probability']
                    
                    if (net_credit / max_loss >= config['target_credit'] and
                        probability >= config['target_probability']):
                        
                        opportunity = {
                            'strategy': StrategyType.IRON_CONDOR,
                            'symbol': symbol,
                            'contracts': setup['contracts'],
                            'expected_return': net_credit / max_loss,
                            'probability': probability,
                            'risk_amount': max_loss,
                            'max_profit': net_credit,
                            'breakeven': setup['breakevens'],
                            'confidence': 0.8,
                            'reasoning': f"High IV crush opportunity, {probability:.0%} probability of profit"
                        }
                        opportunities.append(opportunity)
            
            return opportunities[:2]
            
        except Exception as e:
            self.logger.error(f"Error finding iron condor opportunities: {e}")
            return []
    
    def _build_iron_condor_setups(self,
                                current_price: float,
                                options_chain: Dict,
                                expiration: str) -> List[Dict]:
        """Build iron condor setups"""
        setups = []
        
        try:
            # Simplified iron condor setup
            # This would need more detailed implementation for real trading
            
            return setups
            
        except Exception as e:
            self.logger.error(f"Error building iron condor setups: {e}")
            return []
    
    def _analyze_gap_strategy(self,
                            symbol: str,
                            market_data: pd.DataFrame,
                            options_chain: Dict,
                            market_conditions: MarketConditions) -> List[Dict]:
        """Analyze gap fill strategy opportunities"""
        opportunities = []
        
        try:
            if len(market_data) < 2:
                return opportunities
            
            current_bar = market_data.iloc[-1]
            previous_bar = market_data.iloc[-2]
            
            # Check for significant gap
            gap_size = abs(current_bar['open'] - previous_bar['close']) / previous_bar['close']
            
            if gap_size >= 0.02:  # 2% gap
                gap_direction = 1 if current_bar['open'] > previous_bar['close'] else -1
                fill_direction = -gap_direction  # Expect gap to fill
                
                # Find options for gap fill
                gap_fill_ops = self._find_gap_fill_options(
                    symbol, market_conditions.underlying_price, fill_direction,
                    options_chain, previous_bar['close']
                )
                opportunities.extend(gap_fill_ops)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing gap strategy: {e}")
            return []
    
    def _find_gap_fill_options(self,
                             symbol: str,
                             current_price: float,
                             direction: int,
                             options_chain: Dict,
                             target_price: float) -> List[Dict]:
        """Find options for gap fill strategy"""
        opportunities = []
        
        try:
            option_type = OptionType.CALL if direction > 0 else OptionType.PUT
            chain_key = 'calls' if direction > 0 else 'puts'
            
            if chain_key not in options_chain:
                return opportunities
            
            # Look for short-term options that benefit from gap fill
            for expiration, contracts in options_chain[chain_key].items():
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                
                if days_to_exp > 7:  # Gap fills usually happen quickly
                    continue
                
                for strike, contract_data in contracts.items():
                    strike_price = float(strike)
                    
                    # Select appropriate strikes for gap fill
                    if direction > 0:  # Calls for upward gap fill
                        if strike_price > target_price * 1.05:  # Too far OTM
                            continue
                    else:  # Puts for downward gap fill
                        if strike_price < target_price * 0.95:  # Too far OTM
                            continue
                    
                    contract = OptionContract(
                        symbol=contract_data.get('symbol', ''),
                        strike=strike_price,
                        expiration=exp_date,
                        option_type=option_type,
                        ask=contract_data.get('ask', 0),
                        implied_volatility=contract_data.get('implied_volatility', 0.3)
                    )
                    
                    # Calculate profit potential if gap fills
                    option_value_at_target = BlackScholes.calculate_option_price(
                        target_price, strike_price, max(days_to_exp-1, 0)/365,
                        self.risk_free_rate, contract.implied_volatility * 0.9, option_type
                    )
                    
                    profit = option_value_at_target - contract.ask
                    risk = contract.ask
                    
                    if profit > 0 and profit/risk >= 1.5:  # At least 1.5:1 risk/reward
                        opportunity = {
                            'strategy': StrategyType.LONG_CALL if option_type == OptionType.CALL else StrategyType.LONG_PUT,
                            'symbol': symbol,
                            'direction': direction,
                            'contracts': [contract],
                            'expected_return': profit / risk,
                            'probability': 0.7,  # Gaps often fill
                            'risk_amount': risk,
                            'max_profit': profit,
                            'breakeven': self._calculate_breakeven(contract, option_type),
                            'confidence': 0.6,
                            'reasoning': f"Gap fill strategy targeting {target_price:.2f}"
                        }
                        opportunities.append(opportunity)
            
            return opportunities[:1]  # Return best opportunity
            
        except Exception as e:
            self.logger.error(f"Error finding gap fill options: {e}")
            return []
    
    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank opportunities by expected return, probability, and confidence"""
        try:
            def opportunity_score(opp):
                expected_return = opp.get('expected_return', 0)
                probability = opp.get('probability', 0)
                confidence = opp.get('confidence', 0)
                
                # Weighted score favoring high returns with good probability
                score = (expected_return * 0.4 + 
                        probability * 0.35 + 
                        confidence * 0.25)
                
                return score
            
            opportunities.sort(key=opportunity_score, reverse=True)
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error ranking opportunities: {e}")
            return opportunities
    
    def execute_strategy(self, opportunity: Dict) -> Optional[OptionsPosition]:
        """Execute an options strategy"""
        try:
            contracts = opportunity['contracts']
            strategy_type = opportunity['strategy']
            
            # Determine quantities (simplified)
            if strategy_type in [StrategyType.LONG_CALL, StrategyType.LONG_PUT]:
                quantities = [1] * len(contracts)
            elif strategy_type == StrategyType.STRADDLE:
                quantities = [1, 1]  # Long call, long put
            elif strategy_type == StrategyType.IRON_CONDOR:
                quantities = [-1, 1, 1, -1]  # Sell put spread, sell call spread
            else:
                quantities = [1] * len(contracts)
            
            # Calculate entry price
            entry_price = sum(contract.ask * qty for contract, qty in zip(contracts, quantities))
            
            # Create position
            position = OptionsPosition(
                contracts=contracts,
                quantities=quantities,
                entry_price=abs(entry_price),
                entry_time=datetime.now(),
                strategy_type=strategy_type,
                stop_loss=opportunity.get('stop_loss'),
                take_profit=opportunity.get('take_profit')
            )
            
            self.positions.append(position)
            self.logger.info(f"Executed {strategy_type.value} strategy for {contracts[0].symbol}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing strategy: {e}")
            return None
    
    def calculate_position_pnl(self, 
                              position: OptionsPosition, 
                              current_prices: Dict[str, float],
                              current_underlying: float) -> Dict:
        """Calculate current P&L for a position"""
        try:
            total_current_value = 0
            
            for contract, quantity in zip(position.contracts, position.quantities):
                # Get current option price (simplified - would need real option pricing)
                current_option_price = self._estimate_current_option_price(
                    contract, current_underlying
                )
                
                position_value = current_option_price * quantity * 100  # Options are per 100 shares
                total_current_value += position_value
            
            entry_value = position.entry_price * 100
            pnl = total_current_value - entry_value
            pnl_percentage = (pnl / entry_value) * 100 if entry_value > 0 else 0
            
            return {
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'current_value': total_current_value,
                'entry_value': entry_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position P&L: {e}")
            return {'pnl': 0, 'pnl_percentage': 0, 'current_value': 0, 'entry_value': 0}
    
    def _estimate_current_option_price(self, 
                                     contract: OptionContract, 
                                     current_underlying: float) -> float:
        """Estimate current option price using Black-Scholes"""
        try:
            time_to_expiration = max(contract.time_to_expiration, 0.001)
            
            current_price = BlackScholes.calculate_option_price(
                current_underlying,
                contract.strike,
                time_to_expiration,
                self.risk_free_rate,
                contract.implied_volatility,
                contract.option_type
            )
            
            return max(current_price, 0.01)  # Minimum price of $0.01
            
        except Exception as e:
            self.logger.error(f"Error estimating option price: {e}")
            return contract.price
    
    def should_close_position(self, 
                             position: OptionsPosition, 
                             current_pnl: Dict,
                             market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        try:
            pnl_percentage = current_pnl['pnl_percentage']
            
            # Stop loss check
            if position.stop_loss and pnl_percentage <= -position.stop_loss:
                return True, f"Stop loss triggered at {pnl_percentage:.1f}%"
            
            # Take profit check
            if position.take_profit and pnl_percentage >= position.take_profit:
                return True, f"Take profit triggered at {pnl_percentage:.1f}%"
            
            # Time decay check for long options
            days_held = (datetime.now() - position.entry_time).days
            days_to_expiration = min([contract.days_to_expiration for contract in position.contracts])
            
            if days_to_expiration <= 5 and position.strategy_type in [
                StrategyType.LONG_CALL, StrategyType.LONG_PUT, StrategyType.STRADDLE
            ]:
                if pnl_percentage < 50:  # Close if not significantly profitable
                    return True, f"Time decay concern with {days_to_expiration} days left"
            
            # Volatility change check
            if position.strategy_type == StrategyType.STRADDLE:
                if market_conditions.implied_volatility_rank < 0.2:  # IV crush
                    return True, "Implied volatility crush detected"
            
            # Momentum loss check
            if position.strategy_type == StrategyType.MOMENTUM_BREAKOUT:
                if abs(market_conditions.trend_strength) < 0.3:  # Lost momentum
                    return True, "Momentum lost"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Error checking position close conditions: {e}")
            return False, "Error in analysis"
    
    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate total portfolio Greeks"""
        try:
            total_delta = sum(pos.total_delta for pos in self.positions)
            total_gamma = sum(pos.total_gamma for pos in self.positions)
            total_theta = sum(pos.total_theta for pos in self.positions)
            total_vega = sum(pos.total_vega for pos in self.positions)
            
            return {
                'delta': total_delta,
                'gamma': total_gamma,
                'theta': total_theta,
                'vega': total_vega
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def get_strategy_performance(self) -> Dict:
        """Get performance statistics by strategy"""
        try:
            performance = {}
            
            for strategy in StrategyType:
                strategy_positions = [pos for pos in self.positions if pos.strategy_type == strategy]
                
                if strategy_positions:
                    total_pnl = sum(self.calculate_position_pnl(pos, {}, 0)['pnl'] for pos in strategy_positions)
                    win_rate = len([pos for pos in strategy_positions if self.calculate_position_pnl(pos, {}, 0)['pnl'] > 0]) / len(strategy_positions)
                    
                    performance[strategy.value] = {
                        'total_positions': len(strategy_positions),
                        'total_pnl': total_pnl,
                        'win_rate': win_rate,
                        'avg_pnl': total_pnl / len(strategy_positions)
                    }
                else:
                    performance[strategy.value] = {
                        'total_positions': 0,
                        'total_pnl': 0,
                        'win_rate': 0,
                        'avg_pnl': 0
                    }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy performance: {e}")
            return {}
    
    def optimize_position_sizing(self, 
                               opportunity: Dict, 
                               portfolio_value: float,
                               risk_tolerance: float = 0.02) -> int:
        """Optimize position sizing based on Kelly Criterion and risk management"""
        try:
            expected_return = opportunity.get('expected_return', 0)
            probability = opportunity.get('probability', 0.5)
            risk_amount = opportunity.get('risk_amount', 0)
            
            if risk_amount <= 0:
                return 0
            
            # Kelly Criterion calculation
            win_rate = probability
            avg_win = expected_return
            avg_loss = 1.0  # Assume 100% loss on bad trades
            
            if win_rate <= 0 or avg_win <= 0:
                return 0
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Risk-based position sizing
            max_risk_amount = portfolio_value * risk_tolerance
            max_contracts_by_risk = int(max_risk_amount / (risk_amount * 100))
            
            # Kelly-based position sizing
            kelly_risk_amount = portfolio_value * kelly_fraction
            kelly_contracts = int(kelly_risk_amount / (risk_amount * 100))
            
            # Take the smaller of the two
            optimal_contracts = min(max_contracts_by_risk, kelly_contracts, 10)  # Max 10 contracts
            
            return max(1, optimal_contracts)  # At least 1 contract
            
        except Exception as e:
            self.logger.error(f"Error optimizing position sizing: {e}")
            return 1
    
    def generate_options_alerts(self, 
                              opportunities: List[Dict],
                              market_conditions: MarketConditions) -> List[str]:
        """Generate trading alerts for high-probability opportunities"""
        alerts = []
        
        try:
            for opp in opportunities[:3]:  # Top 3 opportunities
                symbol = opp['symbol']
                strategy = opp['strategy'].value
                expected_return = opp['expected_return']
                probability = opp['probability']
                confidence = opp['confidence']
                reasoning = opp['reasoning']
                
                if expected_return >= 2.0 and probability >= 0.6:  # High-quality opportunity
                    alert = (f"🚀 HIGH ALERT: {symbol} {strategy.upper()} - "
                           f"Expected Return: {expected_return:.1f}x, "
                           f"Probability: {probability:.0%}, "
                           f"Confidence: {confidence:.0%} - {reasoning}")
                    alerts.append(alert)
                elif expected_return >= 1.5 and probability >= 0.7:
                    alert = (f"📈 ALERT: {symbol} {strategy.upper()} - "
                           f"Expected Return: {expected_return:.1f}x, "
                           f"Probability: {probability:.0%} - {reasoning}")
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error generating alerts: {e}")
            return []
    
    def backtest_strategy(self, 
                         strategy_type: StrategyType,
                         historical_data: pd.DataFrame,
                         options_data: Dict) -> Dict:
        """Backtest a specific options strategy"""
        try:
            results = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0
            }
            
            # Simplified backtesting implementation
            # In real implementation, this would be much more detailed
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error backtesting strategy: {e}")
            return {}
    
    def get_market_outlook(self, market_conditions: MarketConditions) -> Dict:
        """Generate market outlook for options trading"""
        try:
            outlook = {
                'overall_bias': 'neutral',
                'volatility_environment': 'normal',
                'preferred_strategies': [],
                'risk_level': 'medium'
            }
            
            # Determine bias
            if market_conditions.trend_strength > 0.6:
                outlook['overall_bias'] = 'bullish'
                outlook['preferred_strategies'].extend([
                    StrategyType.LONG_CALL.value,
                    StrategyType.CALL_SPREAD.value
                ])
            elif market_conditions.trend_strength < -0.6:
                outlook['overall_bias'] = 'bearish'
                outlook['preferred_strategies'].extend([
                    StrategyType.LONG_PUT.value,
                    StrategyType.PUT_SPREAD.value
                ])
            
            # Determine volatility environment
            if market_conditions.implied_volatility_rank > 0.7:
                outlook['volatility_environment'] = 'high'
                outlook['preferred_strategies'].append(StrategyType.IRON_CONDOR.value)
                outlook['risk_level'] = 'high'
            elif market_conditions.implied_volatility_rank < 0.3:
                outlook['volatility_environment'] = 'low'
                outlook['preferred_strategies'].extend([
                    StrategyType.STRADDLE.value,
                    StrategyType.STRANGLE.value
                ])
            
            # Check for special conditions
            if market_conditions.earnings_approaching:
                outlook['preferred_strategies'].extend([
                    StrategyType.STRADDLE.value,
                    StrategyType.VOLATILITY_CRUSH.value
                ])
            
            if market_conditions.technical_breakout:
                outlook['preferred_strategies'].append(StrategyType.MOMENTUM_BREAKOUT.value)
                outlook['risk_level'] = 'high'
            
            return outlook
            
        except Exception as e:
            self.logger.error(f"Error generating market outlook: {e}")
            return {}
            
    def analyze_options_opportunity(self, symbol: str, analysis: Dict, price_data: pd.DataFrame) -> Dict:
        """
        Analyze options trading opportunities for a given symbol
        This method is required by the main application
        """
        try:
            options_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'opportunities': [],
                'best_strategy': None,
                'risk_level': 'medium',
                'expected_return': 0.0,
                'probability_of_profit': 0.0
            }
            
            # Get current price and prediction
            current_price = analysis.get('current_price', price_data['close'].iloc[-1] if len(price_data) > 0 else 100)
            prediction = analysis.get('prediction', 0)
            confidence = analysis.get('confidence', 0.5)
            
            # Create mock market conditions based on analysis
            market_conditions = MarketConditions(
                underlying_price=current_price,
                implied_volatility=0.25,  # Default IV
                implied_volatility_rank=0.5,
                historical_volatility=0.20,
                volume=1000000,
                volume_ratio=1.0,
                trend_strength=prediction * confidence,
                support_levels=[current_price * 0.95],
                resistance_levels=[current_price * 1.05],
                earnings_approaching=False,
                technical_breakout=confidence > 0.8,
                momentum_score=confidence
            )
            
            # Mock options chain - in real implementation this would come from data feed
            mock_options_chain = self._create_mock_options_chain(symbol, current_price)
            
            # Analyze opportunities using existing methods
            opportunities = self.analyze_market_opportunity(
                symbol, price_data, mock_options_chain, market_conditions
            )
            
            if opportunities:
                # Select best opportunity
                best_opportunity = opportunities[0] if opportunities else None
                
                options_analysis.update({
                    'opportunities': opportunities[:3],  # Top 3 opportunities
                    'best_strategy': best_opportunity.get('strategy', '').value if best_opportunity else None,
                    'risk_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
                    'expected_return': best_opportunity.get('expected_return', 0) if best_opportunity else 0,
                    'probability_of_profit': best_opportunity.get('probability', 0) if best_opportunity else 0
                })
            
            return options_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing options opportunity for {symbol}: {e}")
            return {
                'symbol': symbol,
                'opportunities': [],
                'error': str(e)
            }
    
    def select_best_option_contract(self, symbol: str, opportunity: Dict, account_info: Dict) -> Optional[Dict]:
        """
        Select the best option contract from available opportunities
        This method is required by the main application
        """
        try:
            options_analysis = opportunity.get('options_analysis', {})
            opportunities = options_analysis.get('opportunities', [])
            
            if not opportunities:
                return None
            
            # Select the highest-ranked opportunity
            best_opportunity = opportunities[0]
            
            # Extract contract information
            contracts = best_opportunity.get('contracts', [])
            if not contracts:
                return None
            
            # Return the primary contract with required fields
            primary_contract = contracts[0] if contracts else {}
            
            return {
                'symbol': primary_contract.get('symbol', symbol),
                'strike': primary_contract.get('strike', 0),
                'expiration': primary_contract.get('expiration', datetime.now() + timedelta(days=30)),
                'option_type': primary_contract.get('option_type', 'call'),
                'ask': primary_contract.get('price', primary_contract.get('ask', 1.0)),
                'delta': primary_contract.get('delta', 0.5),
                'gamma': primary_contract.get('gamma', 0.1),
                'theta': primary_contract.get('theta', -0.05),
                'vega': primary_contract.get('vega', 0.1),
                'implied_volatility': primary_contract.get('implied_volatility', 0.25)
            }
            
        except Exception as e:
            self.logger.error(f"Error selecting best option contract: {e}")
            return None
    
    def _create_mock_options_chain(self, symbol: str, current_price: float) -> Dict:
        """Create a mock options chain for analysis when real data is not available"""
        try:
            options_chain = {'calls': {}, 'puts': {}}
            
            # Create options for next 3 Friday expirations
            for weeks in [1, 2, 4]:
                expiration_date = self._get_next_friday(weeks)
                exp_str = expiration_date.strftime('%Y-%m-%d')
                
                options_chain['calls'][exp_str] = {}
                options_chain['puts'][exp_str] = {}
                
                # Create strikes around current price
                strikes = []
                for pct in [-0.1, -0.05, 0, 0.05, 0.1]:  # ±10% from current price
                    strike = round(current_price * (1 + pct))
                    strikes.append(strike)
                
                for strike in strikes:
                    # Mock call data
                    options_chain['calls'][exp_str][str(strike)] = {
                        'strike': strike,
                        'last_price': max(0.01, current_price - strike + 2.0) if strike < current_price else 1.0,
                        'bid': 0.5,
                        'ask': 1.0,
                        'volume': 100,
                        'open_interest': 500,
                        'implied_volatility': 0.25
                    }
                    
                    # Mock put data
                    options_chain['puts'][exp_str][str(strike)] = {
                        'strike': strike,
                        'last_price': max(0.01, strike - current_price + 2.0) if strike > current_price else 1.0,
                        'bid': 0.5,
                        'ask': 1.0,
                        'volume': 100,
                        'open_interest': 500,
                        'implied_volatility': 0.25
                    }
            
            return options_chain
            
        except Exception as e:
            self.logger.error(f"Error creating mock options chain: {e}")
            return {'calls': {}, 'puts': {}}
    
    def _get_next_friday(self, weeks: int = 1) -> datetime:
        """Get the next Friday N weeks from now"""
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:  # If today is Friday
            days_until_friday = 7
        
        next_friday = today + timedelta(days=days_until_friday + (weeks - 1) * 7)
        return next_friday
# Additional utility functions for options analysis

def calculate_implied_volatility(option_price: float,
                               underlying_price: float,
                               strike_price: float,
                               time_to_expiration: float,
                               risk_free_rate: float,
                               option_type: OptionType) -> float:
    """Calculate implied volatility using Newton-Raphson method"""
    try:
        # Initial guess
        volatility = 0.3
        tolerance = 0.0001
        max_iterations = 100
        
        for i in range(max_iterations):
            # Calculate option price with current volatility guess
            calculated_price = BlackScholes.calculate_option_price(
                underlying_price, strike_price, time_to_expiration,
                risk_free_rate, volatility, option_type
            )
            
            # Calculate vega for Newton-Raphson
            d1 = (np.log(underlying_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiration) / (volatility * np.sqrt(time_to_expiration))
            vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_expiration) / 100
            
            # Newton-Raphson update
            price_diff = calculated_price - option_price
            
            if abs(price_diff) < tolerance:
                return volatility
            
            if vega != 0:
                volatility = volatility - price_diff / vega
                volatility = max(0.01, min(volatility, 5.0))  # Keep within reasonable bounds
            else:
                break
        
        return volatility
        
    except Exception:
        return 0.3  # Default volatility if calculation fails

def calculate_option_volume_analysis(options_chain: Dict) -> Dict:
    """Analyze option volume patterns for unusual activity"""
    try:
        analysis = {
            'total_call_volume': 0,
            'total_put_volume': 0,
            'put_call_ratio': 0,
            'unusual_activity': [],
            'high_volume_strikes': []
        }
        
        call_volume = 0
        put_volume = 0
        
        # Analyze call volume
        if 'calls' in options_chain:
            for expiration, strikes in options_chain['calls'].items():
                for strike, data in strikes.items():
                    volume = data.get('volume', 0)
                    call_volume += volume
                    
                    # Check for unusual volume
                    avg_volume = data.get('average_volume', 0)
                    if volume > avg_volume * 3 and volume > 100:  # 3x average volume
                        analysis['unusual_activity'].append({
                            'type': 'call',
                            'strike': float(strike),
                            'expiration': expiration,
                            'volume': volume,
                            'avg_volume': avg_volume
                        })
        
        # Analyze put volume
        if 'puts' in options_chain:
            for expiration, strikes in options_chain['puts'].items():
                for strike, data in strikes.items():
                    volume = data.get('volume', 0)
                    put_volume += volume
                    
                    # Check for unusual volume
                    avg_volume = data.get('average_volume', 0)
                    if volume > avg_volume * 3 and volume > 100:
                        analysis['unusual_activity'].append({
                            'type': 'put',
                            'strike': float(strike),
                            'expiration': expiration,
                            'volume': volume,
                            'avg_volume': avg_volume
                        })
        
        analysis['total_call_volume'] = call_volume
        analysis['total_put_volume'] = put_volume
        analysis['put_call_ratio'] = put_volume / call_volume if call_volume > 0 else 0
        
        return analysis
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in volume analysis: {e}")
        return {}

def find_arbitrage_opportunities(options_chain: Dict, underlying_price: float) -> List[Dict]:
    """Find potential arbitrage opportunities in options pricing"""
    arbitrage_opportunities = []
    
    try:
        # Look for violations of basic option pricing principles
        # 1. Call option should be worth more than intrinsic value
        # 2. Put-call parity violations
        # 3. Calendar spread arbitrage
        
        if 'calls' in options_chain and 'puts' in options_chain:
            for expiration in options_chain['calls'].keys():
                if expiration not in options_chain['puts']:
                    continue
                
                for strike in options_chain['calls'][expiration].keys():
                    if strike not in options_chain['puts'][expiration]:
                        continue
                    
                    strike_price = float(strike)
                    call_data = options_chain['calls'][expiration][strike]
                    put_data = options_chain['puts'][expiration][strike]
                    
                    call_price = call_data.get('mid_price', 0)
                    put_price = put_data.get('mid_price', 0)
                    
                    # Check put-call parity: C - P = S - K * e^(-r*T)
                    # Simplified: C - P ≈ S - K for near-term options
                    theoretical_diff = underlying_price - strike_price
                    actual_diff = call_price - put_price
                    
                    parity_violation = abs(actual_diff - theoretical_diff)
                    
                    if parity_violation > 0.5:  # Significant violation
                        arbitrage_opportunities.append({
                            'type': 'put_call_parity',
                            'strike': strike_price,
                            'expiration': expiration,
                            'violation_amount': parity_violation,
                            'call_price': call_price,
                            'put_price': put_price,
                            'theoretical_diff': theoretical_diff,
                            'actual_diff': actual_diff
                        })
        
        return arbitrage_opportunities
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error finding arbitrage opportunities: {e}")
        return []

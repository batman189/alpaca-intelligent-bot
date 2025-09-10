"""
OPTIONS FLOW ANALYZER
Advanced options flow detection and unusual activity monitoring

This module analyzes options flow patterns to detect:
- Unusual options activity (UOA)
- Large block trades and sweeps
- Smart money positioning
- Gamma and delta flow patterns
- Options market maker positioning
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from dataclasses import dataclass
from data.multi_source_data_manager import MultiSourceDataManager

@dataclass
class OptionsFlow:
    """Options flow data structure"""
    timestamp: datetime
    symbol: str
    option_symbol: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    volume: int
    open_interest: int
    price: float
    underlying_price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    flow_type: str  # 'sweep', 'block', 'unusual', 'normal'
    size_score: float  # Relative size compared to normal activity
    smart_money_indicator: float  # 0-1 score for smart money likelihood

@dataclass
class FlowAlert:
    """Options flow alert"""
    timestamp: datetime
    symbol: str
    alert_type: str
    description: str
    confidence: float
    options_data: List[OptionsFlow]
    potential_targets: List[float]
    time_horizon: str
    bullish_bearish: str

class OptionsFlowAnalyzer:
    """Advanced options flow analysis engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_manager = MultiSourceDataManager()
        self.flow_history = {}  # Store recent flow data per symbol
        self.unusual_activity_thresholds = {
            'volume_multiplier': 3.0,
            'size_threshold': 100,  # Minimum contract size to consider
            'iv_spike_threshold': 0.2,  # 20% IV increase
            'block_size': 500,  # Contracts for block trade
            'sweep_threshold': 0.95  # Percentage of ask/bid for sweep
        }
    
    async def analyze_options_flow(self, symbols: List[str]) -> List[FlowAlert]:
        """Analyze options flow for multiple symbols"""
        alerts = []
        
        for symbol in symbols:
            try:
                symbol_alerts = await self._analyze_symbol_flow(symbol)
                alerts.extend(symbol_alerts)
            except Exception as e:
                self.logger.error(f"Error analyzing flow for {symbol}: {e}")
        
        return self._prioritize_alerts(alerts)
    
    async def _analyze_symbol_flow(self, symbol: str) -> List[FlowAlert]:
        """Analyze options flow for a single symbol"""
        alerts = []
        
        # Get options chain data
        options_data = await self.data_manager.get_options_chain(symbol)
        if not options_data:
            return alerts
        
        # Get current stock data
        stock_data = await self.data_manager.get_market_data(symbol)
        if stock_data is None or stock_data.empty:
            return alerts
        
        current_price = stock_data['Close'].iloc[-1]
        
        # Analyze calls and puts separately
        call_flows = await self._process_options_data(
            options_data.get('calls', []), symbol, current_price, 'call'
        )
        put_flows = await self._process_options_data(
            options_data.get('puts', []), symbol, current_price, 'put'
        )
        
        all_flows = call_flows + put_flows
        
        if all_flows:
            # Detect unusual activity patterns
            unusual_volume_alerts = self._detect_unusual_volume(symbol, all_flows)
            alerts.extend(unusual_volume_alerts)
            
            # Detect block trades and sweeps
            block_sweep_alerts = self._detect_blocks_and_sweeps(symbol, all_flows)
            alerts.extend(block_sweep_alerts)
            
            # Detect smart money positioning
            smart_money_alerts = self._detect_smart_money_flow(symbol, all_flows, current_price)
            alerts.extend(smart_money_alerts)
            
            # Detect gamma squeeze potential
            gamma_alerts = self._detect_gamma_squeeze_setup(symbol, all_flows, current_price)
            alerts.extend(gamma_alerts)
            
            # Update flow history
            self.flow_history[symbol] = {
                'timestamp': datetime.now(),
                'flows': all_flows,
                'current_price': current_price
            }
        
        return alerts
    
    async def _process_options_data(self, options_list: List[Dict], symbol: str, 
                                  underlying_price: float, option_type: str) -> List[OptionsFlow]:
        """Process raw options data into OptionsFlow objects"""
        flows = []
        
        for option in options_list:
            try:
                # Calculate Greeks (simplified - in practice would use more sophisticated models)
                strike = option.get('strike', 0)
                expiration = option.get('expiration', '')
                volume = option.get('volume', 0)
                open_interest = option.get('openInterest', 0)
                price = option.get('lastPrice', option.get('mark', 0))
                iv = option.get('impliedVolatility', 0)
                
                if volume == 0 and price == 0:
                    continue
                
                # Calculate basic Greeks
                time_to_expiry = self._calculate_time_to_expiry(expiration)
                delta = self._estimate_delta(underlying_price, strike, time_to_expiry, option_type, iv)
                gamma = self._estimate_gamma(underlying_price, strike, time_to_expiry, iv)
                
                # Determine flow characteristics
                flow_type = self._classify_flow_type(volume, open_interest, price)
                size_score = self._calculate_size_score(volume, symbol)
                smart_money_score = self._calculate_smart_money_score(
                    volume, open_interest, strike, underlying_price, option_type, time_to_expiry
                )
                
                flow = OptionsFlow(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    option_symbol=option.get('contractSymbol', ''),
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    volume=volume,
                    open_interest=open_interest,
                    price=price,
                    underlying_price=underlying_price,
                    implied_volatility=iv,
                    delta=delta,
                    gamma=gamma,
                    theta=self._estimate_theta(time_to_expiry, price),
                    vega=self._estimate_vega(time_to_expiry, iv),
                    flow_type=flow_type,
                    size_score=size_score,
                    smart_money_indicator=smart_money_score
                )
                
                flows.append(flow)
                
            except Exception as e:
                self.logger.error(f"Error processing option data: {e}")
                continue
        
        return flows
    
    def _calculate_time_to_expiry(self, expiration_str: str) -> float:
        """Calculate time to expiry in years"""
        try:
            expiry_date = datetime.strptime(expiration_str, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.now()).days
            return max(days_to_expiry / 365.0, 0.001)  # Minimum 0.1% of year
        except:
            return 0.1  # Default to ~36 days
    
    def _estimate_delta(self, spot: float, strike: float, time_to_expiry: float, 
                       option_type: str, iv: float) -> float:
        """Simplified delta estimation"""
        try:
            from scipy.stats import norm
            import math
            
            d1 = (math.log(spot / strike) + (0.05 + 0.5 * iv ** 2) * time_to_expiry) / (iv * math.sqrt(time_to_expiry))
            
            if option_type == 'call':
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1
        except:
            # Simplified approximation
            moneyness = spot / strike
            if option_type == 'call':
                return min(max((moneyness - 0.9) * 2, 0), 1)
            else:
                return min(max((1.1 - moneyness) * 2, 0), 1)
    
    def _estimate_gamma(self, spot: float, strike: float, time_to_expiry: float, iv: float) -> float:
        """Simplified gamma estimation"""
        try:
            from scipy.stats import norm
            import math
            
            d1 = (math.log(spot / strike) + (0.05 + 0.5 * iv ** 2) * time_to_expiry) / (iv * math.sqrt(time_to_expiry))
            return norm.pdf(d1) / (spot * iv * math.sqrt(time_to_expiry))
        except:
            # Peak gamma near ATM
            moneyness = abs(spot - strike) / spot
            return max(0, 0.1 - moneyness * 2)  # Simplified approximation
    
    def _estimate_theta(self, time_to_expiry: float, price: float) -> float:
        """Simplified theta estimation"""
        return -price / (time_to_expiry * 365)  # Simplified time decay
    
    def _estimate_vega(self, time_to_expiry: float, iv: float) -> float:
        """Simplified vega estimation"""
        return time_to_expiry * 0.1  # Simplified vol sensitivity
    
    def _classify_flow_type(self, volume: int, open_interest: int, price: float) -> str:
        """Classify the type of options flow"""
        if volume == 0:
            return 'normal'
        
        # High volume relative to open interest suggests new positioning
        volume_oi_ratio = volume / max(open_interest, 1)
        
        if volume >= self.unusual_activity_thresholds['block_size']:
            return 'block'
        elif volume >= self.unusual_activity_thresholds['size_threshold'] and volume_oi_ratio > 0.5:
            return 'unusual'
        elif volume_oi_ratio > 2.0:
            return 'sweep'
        else:
            return 'normal'
    
    def _calculate_size_score(self, volume: int, symbol: str) -> float:
        """Calculate size score relative to normal activity"""
        # Get historical average volume for this symbol (simplified)
        historical_avg = self._get_average_option_volume(symbol)
        
        if historical_avg == 0:
            return 1.0
        
        return min(volume / historical_avg, 10.0)  # Cap at 10x
    
    def _get_average_option_volume(self, symbol: str) -> float:
        """Get average option volume for symbol (simplified)"""
        # This would typically query historical data
        # For now, use symbol-specific estimates
        volume_estimates = {
            'SPY': 1000000,
            'QQQ': 500000,
            'TSLA': 200000,
            'AAPL': 300000,
            'MSFT': 150000,
            'NVDA': 250000,
            'UNH': 50000
        }
        return volume_estimates.get(symbol, 10000)
    
    def _calculate_smart_money_score(self, volume: int, open_interest: int, strike: float,
                                   underlying_price: float, option_type: str, 
                                   time_to_expiry: float) -> float:
        """Calculate likelihood this is smart money flow"""
        score = 0.0
        
        # Size factor - larger trades more likely to be smart money
        if volume >= 1000:
            score += 0.3
        elif volume >= 500:
            score += 0.2
        elif volume >= 100:
            score += 0.1
        
        # Strike selection - near ATM for hedging, far OTM for speculation
        moneyness = underlying_price / strike if option_type == 'call' else strike / underlying_price
        if 0.95 <= moneyness <= 1.05:  # Near ATM
            score += 0.2
        elif moneyness > 1.1 or moneyness < 0.9:  # Far from ATM
            score += 0.1
        
        # Time to expiry - longer dated more likely to be positioning
        if time_to_expiry > 0.25:  # > 3 months
            score += 0.2
        elif time_to_expiry > 0.08:  # > 1 month
            score += 0.1
        
        # Volume vs Open Interest
        volume_oi_ratio = volume / max(open_interest, 1)
        if volume_oi_ratio > 0.5:  # New positioning
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_unusual_volume(self, symbol: str, flows: List[OptionsFlow]) -> List[FlowAlert]:
        """Detect unusual volume patterns"""
        alerts = []
        
        # Group flows by expiration and type
        flow_groups = {}
        for flow in flows:
            key = f"{flow.expiration}_{flow.option_type}"
            if key not in flow_groups:
                flow_groups[key] = []
            flow_groups[key].append(flow)
        
        for group_key, group_flows in flow_groups.items():
            total_volume = sum(f.volume for f in group_flows)
            weighted_strikes = sum(f.strike * f.volume for f in group_flows) / max(total_volume, 1)
            
            # Check if volume is unusual
            avg_volume = self._get_average_option_volume(symbol) / 10  # Per expiration estimate
            
            if total_volume > avg_volume * self.unusual_activity_thresholds['volume_multiplier']:
                expiration, option_type = group_key.split('_')
                
                alert = FlowAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type='unusual_volume',
                    description=f"Unusual {option_type} volume in {expiration} expiry: {total_volume:,} contracts",
                    confidence=min(total_volume / (avg_volume * 5), 1.0),
                    options_data=group_flows,
                    potential_targets=[weighted_strikes],
                    time_horizon=self._calculate_time_horizon(expiration),
                    bullish_bearish='bullish' if option_type == 'call' else 'bearish'
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_blocks_and_sweeps(self, symbol: str, flows: List[OptionsFlow]) -> List[FlowAlert]:
        """Detect block trades and option sweeps"""
        alerts = []
        
        for flow in flows:
            if flow.flow_type in ['block', 'sweep']:
                direction = 'bullish' if flow.option_type == 'call' else 'bearish'
                
                # Adjust direction based on moneyness and premium paid
                moneyness = flow.underlying_price / flow.strike if flow.option_type == 'call' else flow.strike / flow.underlying_price
                
                if flow.option_type == 'call' and moneyness < 0.9:  # Deep OTM calls
                    direction = 'very_bullish'
                elif flow.option_type == 'put' and moneyness < 0.9:  # Deep OTM puts
                    direction = 'very_bearish'
                
                alert = FlowAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type=flow.flow_type,
                    description=f"{flow.flow_type.upper()}: {flow.volume:,} {flow.option_type}s at ${flow.strike} exp {flow.expiration}",
                    confidence=flow.smart_money_indicator,
                    options_data=[flow],
                    potential_targets=[flow.strike],
                    time_horizon=self._calculate_time_horizon(flow.expiration),
                    bullish_bearish=direction
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_smart_money_flow(self, symbol: str, flows: List[OptionsFlow], 
                               current_price: float) -> List[FlowAlert]:
        """Detect likely smart money positioning"""
        alerts = []
        
        # Find flows with high smart money scores
        smart_flows = [f for f in flows if f.smart_money_indicator > 0.6]
        
        if not smart_flows:
            return alerts
        
        # Group by similar characteristics
        call_flows = [f for f in smart_flows if f.option_type == 'call']
        put_flows = [f for f in smart_flows if f.option_type == 'put']
        
        # Analyze call positioning
        if call_flows:
            total_call_volume = sum(f.volume for f in call_flows)
            avg_call_strike = sum(f.strike * f.volume for f in call_flows) / total_call_volume
            
            alert = FlowAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='smart_money_calls',
                description=f"Smart money call positioning: {total_call_volume:,} contracts around ${avg_call_strike:.2f}",
                confidence=sum(f.smart_money_indicator * f.volume for f in call_flows) / total_call_volume,
                options_data=call_flows,
                potential_targets=[avg_call_strike],
                time_horizon=self._calculate_dominant_time_horizon(call_flows),
                bullish_bearish='bullish'
            )
            alerts.append(alert)
        
        # Analyze put positioning
        if put_flows:
            total_put_volume = sum(f.volume for f in put_flows)
            avg_put_strike = sum(f.strike * f.volume for f in put_flows) / total_put_volume
            
            alert = FlowAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='smart_money_puts',
                description=f"Smart money put positioning: {total_put_volume:,} contracts around ${avg_put_strike:.2f}",
                confidence=sum(f.smart_money_indicator * f.volume for f in put_flows) / total_put_volume,
                options_data=put_flows,
                potential_targets=[avg_put_strike],
                time_horizon=self._calculate_dominant_time_horizon(put_flows),
                bullish_bearish='bearish'
            )
            alerts.append(alert)
        
        return alerts
    
    def _detect_gamma_squeeze_setup(self, symbol: str, flows: List[OptionsFlow], 
                                  current_price: float) -> List[FlowAlert]:
        """Detect potential gamma squeeze setups"""
        alerts = []
        
        # Find calls near current price with high gamma
        near_money_calls = [
            f for f in flows 
            if (f.option_type == 'call' and 
                abs(f.strike - current_price) / current_price < 0.05 and  # Within 5%
                f.gamma > 0.05 and  # High gamma
                f.volume > 100)  # Decent volume
        ]
        
        if len(near_money_calls) >= 2:  # Multiple strikes with activity
            total_gamma_exposure = sum(f.volume * f.gamma * 100 for f in near_money_calls)  # 100 shares per contract
            
            if total_gamma_exposure > 50000:  # Significant gamma exposure
                weighted_strike = sum(f.strike * f.volume for f in near_money_calls) / sum(f.volume for f in near_money_calls)
                
                alert = FlowAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type='gamma_squeeze_setup',
                    description=f"Gamma squeeze potential: {total_gamma_exposure:,.0f} gamma exposure near ${weighted_strike:.2f}",
                    confidence=min(total_gamma_exposure / 100000, 1.0),
                    options_data=near_money_calls,
                    potential_targets=[weighted_strike],
                    time_horizon='short',
                    bullish_bearish='bullish'
                )
                alerts.append(alert)
        
        return alerts
    
    def _calculate_time_horizon(self, expiration: str) -> str:
        """Calculate time horizon from expiration"""
        try:
            expiry_date = datetime.strptime(expiration, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.now()).days
            
            if days_to_expiry <= 7:
                return 'very_short'
            elif days_to_expiry <= 30:
                return 'short'
            elif days_to_expiry <= 90:
                return 'medium'
            else:
                return 'long'
        except:
            return 'short'
    
    def _calculate_dominant_time_horizon(self, flows: List[OptionsFlow]) -> str:
        """Calculate dominant time horizon from multiple flows"""
        if not flows:
            return 'short'
        
        horizons = [self._calculate_time_horizon(f.expiration) for f in flows]
        horizon_weights = {'very_short': 1, 'short': 2, 'medium': 3, 'long': 4}
        
        weighted_horizon = sum(horizon_weights[h] for h in horizons) / len(horizons)
        
        if weighted_horizon <= 1.5:
            return 'very_short'
        elif weighted_horizon <= 2.5:
            return 'short'
        elif weighted_horizon <= 3.5:
            return 'medium'
        else:
            return 'long'
    
    def _prioritize_alerts(self, alerts: List[FlowAlert]) -> List[FlowAlert]:
        """Prioritize alerts by confidence and importance"""
        # Sort by confidence score and alert type importance
        alert_priorities = {
            'gamma_squeeze_setup': 5,
            'smart_money_calls': 4,
            'smart_money_puts': 4,
            'block': 3,
            'sweep': 3,
            'unusual_volume': 2
        }
        
        def priority_score(alert):
            type_priority = alert_priorities.get(alert.alert_type, 1)
            return alert.confidence * type_priority
        
        return sorted(alerts, key=priority_score, reverse=True)
    
    async def get_flow_summary(self, symbol: str) -> Dict:
        """Get summary of options flow for a symbol"""
        if symbol not in self.flow_history:
            return {'status': 'no_data'}
        
        flows = self.flow_history[symbol]['flows']
        current_price = self.flow_history[symbol]['current_price']
        
        call_flows = [f for f in flows if f.option_type == 'call']
        put_flows = [f for f in flows if f.option_type == 'put']
        
        call_volume = sum(f.volume for f in call_flows)
        put_volume = sum(f.volume for f in put_flows)
        
        # Calculate put/call ratio
        pc_ratio = put_volume / max(call_volume, 1)
        
        # Find dominant strikes
        strike_volumes = {}
        for flow in flows:
            if flow.strike not in strike_volumes:
                strike_volumes[flow.strike] = 0
            strike_volumes[flow.strike] += flow.volume
        
        top_strikes = sorted(strike_volumes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'total_call_volume': call_volume,
            'total_put_volume': put_volume,
            'put_call_ratio': pc_ratio,
            'sentiment': 'bearish' if pc_ratio > 1.2 else 'bullish' if pc_ratio < 0.8 else 'neutral',
            'top_strikes': top_strikes,
            'smart_money_flows': len([f for f in flows if f.smart_money_indicator > 0.6]),
            'unusual_activity_count': len([f for f in flows if f.flow_type in ['block', 'sweep', 'unusual']])
        }
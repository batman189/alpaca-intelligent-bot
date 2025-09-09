"""
Intelligent Risk Manager - Professional Dynamic Risk Management
Advanced risk assessment with ML-based position sizing and exposure management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IntelligentRiskManager:
    def __init__(self):
        """Initialize the Intelligent Risk Manager with professional risk parameters"""
        # Risk parameters
        self.max_portfolio_risk = 0.05  # 5% max risk per trade
        self.max_position_size = 0.10   # 10% max position size
        self.max_correlation_exposure = 0.25  # 25% max to correlated assets
        self.volatility_lookback = 20   # Days for volatility calculation
        self.var_confidence = 0.05      # 95% VaR confidence level
        
        # Dynamic risk adjustment parameters
        self.market_regime_factor = 1.0
        self.portfolio_heat = 0.0
        self.recent_performance_multiplier = 1.0
        
        # Risk metrics storage
        self.symbol_risk_metrics = {}
        self.portfolio_correlations = {}
        self.historical_volatilities = {}
        
        logger.info("Intelligent Risk Manager initialized")
    
    def assess_symbol_risk(self, symbol: str, price_data: pd.DataFrame, 
                          current_positions: Dict, account_info: Dict) -> Dict:
        """
        Comprehensive risk assessment for a specific symbol
        Returns detailed risk metrics and recommendations
        """
        try:
            risk_assessment = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'risk_score': 0.0,
                'volatility_risk': 0.0,
                'liquidity_risk': 0.0,
                'correlation_risk': 0.0,
                'position_size_risk': 0.0,
                'market_regime_risk': 0.0,
                'recommendations': []
            }
            
            # Calculate various risk components
            volatility_risk = self._calculate_volatility_risk(symbol, price_data)
            liquidity_risk = self._calculate_liquidity_risk(symbol, price_data)
            correlation_risk = self._calculate_correlation_risk(symbol, current_positions)
            position_size_risk = self._calculate_position_size_risk(symbol, account_info)
            market_regime_risk = self._calculate_market_regime_risk(price_data)
            
            # Composite risk score (weighted average)
            weights = {
                'volatility': 0.30,
                'liquidity': 0.20,
                'correlation': 0.25,
                'position_size': 0.15,
                'market_regime': 0.10
            }
            
            composite_risk = (
                volatility_risk * weights['volatility'] +
                liquidity_risk * weights['liquidity'] +
                correlation_risk * weights['correlation'] +
                position_size_risk * weights['position_size'] +
                market_regime_risk * weights['market_regime']
            )
            
            risk_assessment.update({
                'risk_score': min(1.0, max(0.0, composite_risk)),
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'correlation_risk': correlation_risk,
                'position_size_risk': position_size_risk,
                'market_regime_risk': market_regime_risk,
                'recommendations': self._generate_risk_recommendations(risk_assessment)
            })
            
            # Store for future reference
            self.symbol_risk_metrics[symbol] = risk_assessment
            
            logger.debug(f"Risk assessment for {symbol}: {composite_risk:.3f}")
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            return {
                'symbol': symbol,
                'risk_score': 0.5,  # Neutral risk if calculation fails
                'error': str(e)
            }
    
    def calculate_options_position_size(self, account_info: Dict, option_contract: Dict, 
                                      opportunity: Dict) -> int:
        """
        Calculate optimal position size for options trading
        Considers premium cost, Greeks, and risk parameters
        """
        try:
            equity = float(account_info.get('equity', 0))
            buying_power = float(account_info.get('buying_power', equity * 0.5))
            
            # Option contract details
            premium = float(option_contract.get('ask', 0))
            delta = float(option_contract.get('delta', 0.5))
            gamma = float(option_contract.get('gamma', 0))
            theta = float(option_contract.get('theta', 0))
            
            if premium <= 0:
                logger.warning("Invalid option premium, cannot calculate position size")
                return 0
            
            # Risk-based position sizing
            max_risk_amount = equity * self.max_portfolio_risk
            confidence = opportunity.get('confidence', 0.5)
            
            # Adjust risk based on confidence and Greeks
            risk_adjustment = confidence * (1 - abs(theta) * 0.1)  # Reduce risk for high theta decay
            adjusted_max_risk = max_risk_amount * risk_adjustment
            
            # Calculate contracts based on premium cost
            max_contracts_by_cost = int(adjusted_max_risk / (premium * 100))  # Options are 100 shares
            
            # Calculate contracts based on delta-adjusted exposure
            underlying_price = float(opportunity.get('current_price', 100))
            delta_adjusted_exposure = abs(delta) * underlying_price * 100
            max_contracts_by_exposure = int(buying_power * self.max_position_size / delta_adjusted_exposure)
            
            # Take the minimum to ensure we don't exceed any limits
            position_size = min(max_contracts_by_cost, max_contracts_by_exposure)
            
            # Apply additional risk controls
            position_size = self._apply_risk_controls(position_size, opportunity)
            
            logger.info(f"Options position size calculated: {position_size} contracts "
                       f"(Premium: ${premium}, Delta: {delta:.3f})")
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating options position size: {e}")
            return 0
    
    def calculate_stock_position_size(self, account_info: Dict, opportunity: Dict) -> int:
        """
        Calculate optimal position size for stock trading
        Uses Kelly Criterion with risk adjustments
        """
        try:
            equity = float(account_info.get('equity', 0))
            buying_power = float(account_info.get('buying_power', equity * 0.5))
            
            symbol = opportunity.get('symbol', '')
            current_price = float(opportunity.get('current_price', 100))
            confidence = opportunity.get('confidence', 0.5)
            
            # Get historical volatility for the symbol
            volatility = self.historical_volatilities.get(symbol, 0.25)  # Default 25% annual volatility
            
            # Kelly Criterion calculation
            win_probability = confidence
            avg_win = 0.03  # Assume 3% average win
            avg_loss = 0.02  # Assume 2% average loss
            
            if avg_loss == 0:
                kelly_fraction = 0
            else:
                kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
            
            # Apply conservative Kelly (usually 25% of full Kelly)
            conservative_kelly = kelly_fraction * 0.25
            
            # Risk-adjusted position size
            max_position_value = min(
                equity * self.max_position_size,
                buying_power * 0.95,  # Leave 5% buffer
                equity * conservative_kelly
            )
            
            # Calculate number of shares
            position_size = int(max_position_value / current_price)
            
            # Apply additional risk controls
            position_size = self._apply_risk_controls(position_size, opportunity)
            
            logger.info(f"Stock position size calculated: {position_size} shares "
                       f"(Price: ${current_price}, Kelly: {conservative_kelly:.3f})")
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating stock position size: {e}")
            return 0
    
    def _calculate_volatility_risk(self, symbol: str, price_data: pd.DataFrame) -> float:
        """Calculate volatility-based risk (0.0 = low risk, 1.0 = high risk)"""
        try:
            if len(price_data) < self.volatility_lookback:
                return 0.5  # Neutral if insufficient data
            
            # Calculate returns and volatility
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.rolling(self.volatility_lookback).std().iloc[-1]
            annualized_vol = volatility * np.sqrt(252)
            
            # Store for future use
            self.historical_volatilities[symbol] = annualized_vol
            
            # Risk scoring: higher volatility = higher risk
            # Normalize: 10% vol = 0.0 risk, 50%+ vol = 1.0 risk
            risk_score = min(1.0, max(0.0, (annualized_vol - 0.10) / 0.40))
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {e}")
            return 0.5
    
    def _calculate_liquidity_risk(self, symbol: str, price_data: pd.DataFrame) -> float:
        """Calculate liquidity risk based on volume patterns"""
        try:
            if 'volume' not in price_data.columns or len(price_data) < 10:
                return 0.3  # Moderate risk if no volume data
            
            recent_volume = price_data['volume'].tail(10).mean()
            avg_volume = price_data['volume'].mean()
            
            # Volume ratio: recent vs average
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Lower volume ratio = higher liquidity risk
            if volume_ratio > 1.5:
                return 0.1  # High volume = low risk
            elif volume_ratio > 0.8:
                return 0.3  # Normal volume = moderate risk
            else:
                return 0.7  # Low volume = high risk
                
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.5
    
    def _calculate_correlation_risk(self, symbol: str, current_positions: Dict) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            if not current_positions:
                return 0.0  # No correlation risk if no positions
            
            # Simplified correlation risk - would need historical data for full implementation
            # For now, check sector exposure
            high_correlation_symbols = {
                'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN'],
                'MSFT': ['AAPL', 'GOOGL', 'META', 'AMZN'],
                'TSLA': ['F', 'GM', 'RIVN'],
                'SPY': ['QQQ', 'IWM'],
                'QQQ': ['SPY', 'ARKK', 'XLK']
            }
            
            correlated_symbols = high_correlation_symbols.get(symbol, [])
            
            # Check exposure to correlated assets
            total_correlated_exposure = 0
            for pos_symbol in current_positions:
                if pos_symbol in correlated_symbols:
                    total_correlated_exposure += 1
            
            # Risk increases with number of correlated positions
            correlation_risk = min(1.0, total_correlated_exposure * 0.25)
            
            return correlation_risk
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.2
    
    def _calculate_position_size_risk(self, symbol: str, account_info: Dict) -> float:
        """Calculate risk based on intended position size relative to portfolio"""
        try:
            # This is a placeholder - actual implementation would calculate
            # the intended position size and assess concentration risk
            return 0.2  # Moderate position size risk
            
        except Exception as e:
            logger.error(f"Error calculating position size risk: {e}")
            return 0.3
    
    def _calculate_market_regime_risk(self, price_data: pd.DataFrame) -> float:
        """Calculate market regime risk (trending vs choppy markets)"""
        try:
            if len(price_data) < 20:
                return 0.5
            
            # Simple trend strength calculation
            close_prices = price_data['close'].tail(20)
            returns = close_prices.pct_change().dropna()
            
            # Calculate trend consistency
            positive_days = (returns > 0).sum()
            trend_consistency = abs(positive_days - 10) / 10  # Deviation from 50/50
            
            # Higher consistency = lower regime risk
            regime_risk = 1.0 - trend_consistency
            
            return min(1.0, max(0.0, regime_risk))
            
        except Exception as e:
            logger.error(f"Error calculating market regime risk: {e}")
            return 0.5
    
    def _generate_risk_recommendations(self, risk_assessment: Dict) -> List[str]:
        """Generate actionable risk recommendations"""
        recommendations = []
        
        risk_score = risk_assessment.get('risk_score', 0)
        
        if risk_score > 0.8:
            recommendations.append("HIGH RISK: Consider avoiding this trade")
        elif risk_score > 0.6:
            recommendations.append("ELEVATED RISK: Reduce position size by 50%")
        elif risk_score > 0.4:
            recommendations.append("MODERATE RISK: Use standard position sizing")
        else:
            recommendations.append("LOW RISK: Normal position sizing acceptable")
        
        # Specific component recommendations
        if risk_assessment.get('volatility_risk', 0) > 0.7:
            recommendations.append("High volatility detected - use wider stops")
        
        if risk_assessment.get('liquidity_risk', 0) > 0.6:
            recommendations.append("Low liquidity - use limit orders only")
        
        if risk_assessment.get('correlation_risk', 0) > 0.5:
            recommendations.append("High correlation with existing positions")
        
        return recommendations
    
    def _apply_risk_controls(self, position_size: int, opportunity: Dict) -> int:
        """Apply final risk controls to position size"""
        try:
            # Apply portfolio heat adjustment
            if self.portfolio_heat > 0.7:
                position_size = int(position_size * 0.5)  # Reduce by 50% if portfolio is "hot"
            
            # Apply recent performance multiplier
            position_size = int(position_size * self.recent_performance_multiplier)
            
            # Ensure minimum viable position
            if position_size < 1:
                return 0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error applying risk controls: {e}")
            return position_size
    
    def update_portfolio_metrics(self, current_positions: Dict, account_info: Dict):
        """Update portfolio-level risk metrics"""
        try:
            # Calculate portfolio heat (how much risk is currently deployed)
            total_risk_deployed = len(current_positions) * self.max_portfolio_risk
            self.portfolio_heat = min(1.0, total_risk_deployed)
            
            # Update market regime factor based on overall market conditions
            self._update_market_regime_factor()
            
            logger.debug(f"Portfolio metrics updated - Heat: {self.portfolio_heat:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def _update_market_regime_factor(self):
        """Update market regime factor based on overall market conditions"""
        try:
            # This would typically analyze broad market indicators (VIX, SPY, etc.)
            # For now, using a simplified approach
            self.market_regime_factor = 1.0  # Neutral market
            
        except Exception as e:
            logger.error(f"Error updating market regime factor: {e}")
    
    def get_portfolio_risk_summary(self) -> Dict:
        """Get comprehensive portfolio risk summary"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_heat': self.portfolio_heat,
                'market_regime_factor': self.market_regime_factor,
                'recent_performance_multiplier': self.recent_performance_multiplier,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_size': self.max_position_size,
                'symbols_tracked': len(self.symbol_risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio risk summary: {e}")
            return {'error': str(e)}

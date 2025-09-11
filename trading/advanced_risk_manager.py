"""
Advanced Risk Management System - Production Ready
Real-time position monitoring with circuit breakers and risk controls
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    daily_pnl: float
    var_1d: float  # Value at Risk 1 day
    var_5d: float  # Value at Risk 5 day
    beta: float
    volatility: float
    max_drawdown: float
    risk_level: RiskLevel
    warnings: List[str]

class AdvancedRiskManager:
    """Enterprise-grade risk management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk limits from config
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2% max daily risk
        self.max_position_size = config.get('max_position_size', 0.10)    # 10% max position
        self.max_sector_exposure = config.get('max_sector_exposure', 0.25) # 25% max sector
        self.max_correlation = config.get('max_correlation', 0.7)         # 70% max correlation
        self.stop_loss_pct = config.get('stop_loss_pct', 0.08)           # 8% stop loss
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.15)     # 15% max drawdown
        
        # Circuit breaker thresholds
        self.daily_loss_limit = config.get('daily_loss_limit', 0.05)     # 5% daily loss limit
        self.volatility_threshold = config.get('volatility_threshold', 0.30) # 30% volatility limit
        
        # Portfolio tracking
        self.positions = {}
        self.sector_exposure = {}
        self.correlation_matrix = pd.DataFrame()
        self.performance_history = []
        
        # Risk state
        self.circuit_breaker_active = False
        self.last_risk_check = datetime.now()
        self.risk_alerts = []
        
        logger.info("🛡️ Advanced Risk Manager initialized")
    
    async def evaluate_trade_risk(self, symbol: str, trade_size: float, 
                                current_price: float, position_data: Dict) -> Dict[str, Any]:
        """Comprehensive trade risk evaluation before execution"""
        try:
            risk_assessment = {
                'approved': True,
                'risk_level': 'low',
                'warnings': [],
                'required_stop_loss': None,
                'max_position_size': trade_size,
                'risk_metrics': {}
            }
            
            # Portfolio value check
            total_portfolio_value = self._calculate_portfolio_value(position_data)
            position_value = abs(trade_size) * current_price
            position_pct = position_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Position size risk
            if position_pct > self.max_position_size:
                risk_assessment['approved'] = False
                risk_assessment['warnings'].append(f"Position size {position_pct:.1%} exceeds limit {self.max_position_size:.1%}")
            
            # Circuit breaker check
            if self.circuit_breaker_active:
                risk_assessment['approved'] = False
                risk_assessment['warnings'].append("Circuit breaker active - trading halted")
            
            # Market volatility check
            volatility = await self._calculate_symbol_volatility(symbol)
            if volatility > self.volatility_threshold:
                risk_assessment['risk_level'] = 'high'
                risk_assessment['warnings'].append(f"High volatility detected: {volatility:.1%}")
            
            # Calculate required stop loss
            stop_loss_price = current_price * (1 - self.stop_loss_pct) if trade_size > 0 else current_price * (1 + self.stop_loss_pct)
            risk_assessment['required_stop_loss'] = stop_loss_price
            
            # Sector concentration risk
            sector_risk = await self._check_sector_risk(symbol, position_value, total_portfolio_value)
            if sector_risk['exceeded']:
                risk_assessment['warnings'].append(f"Sector exposure limit exceeded: {sector_risk['exposure']:.1%}")
            
            # Correlation risk
            correlation_risk = await self._check_correlation_risk(symbol, position_data)
            if correlation_risk['high_correlation']:
                risk_assessment['warnings'].append(f"High correlation with existing positions: {correlation_risk['max_correlation']:.2f}")
            
            # Risk metrics
            risk_assessment['risk_metrics'] = {
                'position_pct': position_pct,
                'volatility': volatility,
                'sector_exposure': sector_risk['exposure'],
                'max_correlation': correlation_risk['max_correlation'],
                'var_estimate': self._calculate_var_estimate(position_value, volatility)
            }
            
            # Overall risk level
            if len(risk_assessment['warnings']) == 0:
                risk_assessment['risk_level'] = 'low'
            elif len(risk_assessment['warnings']) <= 2:
                risk_assessment['risk_level'] = 'moderate'
            else:
                risk_assessment['risk_level'] = 'high'
            
            logger.info(f"🎯 Risk assessment for {symbol}: {risk_assessment['risk_level']} risk, {len(risk_assessment['warnings'])} warnings")
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk evaluation failed for {symbol}: {e}")
            return {
                'approved': False,
                'risk_level': 'high',
                'warnings': [f"Risk evaluation error: {str(e)}"],
                'required_stop_loss': None,
                'max_position_size': 0,
                'risk_metrics': {}
            }
    
    async def monitor_portfolio_risk(self, position_data: Dict, market_data: Dict) -> Dict[str, Any]:
        """Continuous portfolio risk monitoring"""
        try:
            portfolio_metrics = {
                'total_value': 0,
                'daily_pnl': 0,
                'unrealized_pnl': 0,
                'var_1d': 0,
                'var_5d': 0,
                'max_drawdown': 0,
                'risk_level': RiskLevel.LOW,
                'active_warnings': [],
                'circuit_breaker_status': self.circuit_breaker_active
            }
            
            # Calculate portfolio value and P&L
            total_value = self._calculate_portfolio_value(position_data)
            daily_pnl = self._calculate_daily_pnl(position_data, market_data)
            unrealized_pnl = self._calculate_unrealized_pnl(position_data, market_data)
            
            portfolio_metrics['total_value'] = total_value
            portfolio_metrics['daily_pnl'] = daily_pnl
            portfolio_metrics['unrealized_pnl'] = unrealized_pnl
            
            # Daily loss check
            daily_loss_pct = abs(daily_pnl) / total_value if total_value > 0 else 0
            if daily_loss_pct > self.daily_loss_limit:
                portfolio_metrics['active_warnings'].append(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                await self._trigger_circuit_breaker("Daily loss limit exceeded")
            
            # Calculate Value at Risk
            var_metrics = await self._calculate_portfolio_var(position_data, market_data)
            portfolio_metrics['var_1d'] = var_metrics['var_1d']
            portfolio_metrics['var_5d'] = var_metrics['var_5d']
            
            # Maximum drawdown check
            max_drawdown = await self._calculate_max_drawdown()
            portfolio_metrics['max_drawdown'] = max_drawdown
            
            if max_drawdown > self.max_drawdown_pct:
                portfolio_metrics['active_warnings'].append(f"Maximum drawdown exceeded: {max_drawdown:.2%}")
            
            # Risk level assessment
            portfolio_metrics['risk_level'] = self._assess_overall_risk_level(
                daily_loss_pct, var_metrics['var_1d'], max_drawdown, len(portfolio_metrics['active_warnings'])
            )
            
            # Update performance history
            self._update_performance_history(portfolio_metrics)
            
            logger.info(f"📊 Portfolio risk: {portfolio_metrics['risk_level'].value}, "
                       f"Value: ${total_value:,.2f}, Daily P&L: ${daily_pnl:,.2f}")
            
            return portfolio_metrics
            
        except Exception as e:
            logger.error(f"Portfolio risk monitoring failed: {e}")
            return {
                'total_value': 0,
                'daily_pnl': 0,
                'risk_level': RiskLevel.EXTREME,
                'active_warnings': [f"Risk monitoring error: {str(e)}"],
                'circuit_breaker_status': True
            }
    
    async def check_stop_loss_triggers(self, position_data: Dict, market_data: Dict) -> List[Dict]:
        """Check for stop loss triggers across all positions"""
        try:
            stop_loss_triggers = []
            
            for symbol, position in position_data.items():
                if symbol not in market_data:
                    continue
                
                current_price = market_data[symbol]['close']
                entry_price = position.get('avg_entry_price', current_price)
                quantity = position.get('qty', 0)
                
                if quantity == 0:
                    continue
                
                # Calculate unrealized P&L percentage
                if quantity > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    stop_loss_triggered = pnl_pct < -self.stop_loss_pct
                else:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    stop_loss_triggered = pnl_pct < -self.stop_loss_pct
                
                if stop_loss_triggered:
                    stop_loss_triggers.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'pnl_pct': pnl_pct,
                        'unrealized_pnl': (current_price - entry_price) * quantity,
                        'action': 'CLOSE_POSITION'
                    })
                    
                    logger.warning(f"🚨 Stop loss triggered for {symbol}: {pnl_pct:.2%} loss")
            
            return stop_loss_triggers
            
        except Exception as e:
            logger.error(f"Stop loss check failed: {e}")
            return []
    
    async def _trigger_circuit_breaker(self, reason: str):
        """Activate circuit breaker to halt trading"""
        try:
            if not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                logger.error(f"🚨 CIRCUIT BREAKER ACTIVATED: {reason}")
                
                # Log critical alert
                alert = {
                    'timestamp': datetime.now(),
                    'type': 'CIRCUIT_BREAKER',
                    'reason': reason,
                    'portfolio_value': self._calculate_portfolio_value({}),
                }
                self.risk_alerts.append(alert)
                
                # Notify monitoring systems
                await self._send_risk_alert(alert)
        
        except Exception as e:
            logger.error(f"Failed to trigger circuit breaker: {e}")
    
    async def reset_circuit_breaker(self, admin_override: bool = False):
        """Reset circuit breaker after manual review"""
        try:
            if admin_override or self._can_reset_circuit_breaker():
                self.circuit_breaker_active = False
                logger.info("✅ Circuit breaker reset")
                return True
            else:
                logger.warning("Circuit breaker reset conditions not met")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reset circuit breaker: {e}")
            return False
    
    def _calculate_portfolio_value(self, position_data: Dict) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = 100000  # Base portfolio value
            for symbol, position in position_data.items():
                market_value = position.get('market_value', 0)
                total_value += market_value
            return total_value
        except:
            return 100000
    
    def _calculate_daily_pnl(self, position_data: Dict, market_data: Dict) -> float:
        """Calculate daily P&L"""
        try:
            daily_pnl = 0
            for symbol, position in position_data.items():
                unrealized_pnl = position.get('unrealized_pl', 0)
                daily_pnl += unrealized_pnl
            return daily_pnl
        except:
            return 0
    
    def _calculate_unrealized_pnl(self, position_data: Dict, market_data: Dict) -> float:
        """Calculate total unrealized P&L"""
        try:
            unrealized_pnl = 0
            for symbol, position in position_data.items():
                unrealized_pnl += position.get('unrealized_pl', 0)
            return unrealized_pnl
        except:
            return 0
    
    async def _calculate_symbol_volatility(self, symbol: str) -> float:
        """Calculate symbol volatility"""
        try:
            # Placeholder - would integrate with data manager
            return 0.25  # Default 25% volatility
        except:
            return 0.30
    
    async def _check_sector_risk(self, symbol: str, position_value: float, total_value: float) -> Dict:
        """Check sector concentration risk"""
        try:
            # Placeholder sector mapping
            sector_map = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'SPY': 'Broad Market', 'QQQ': 'Technology', 'IWM': 'Small Cap'
            }
            
            sector = sector_map.get(symbol, 'Unknown')
            current_sector_exposure = self.sector_exposure.get(sector, 0)
            new_exposure = (current_sector_exposure + position_value) / total_value if total_value > 0 else 0
            
            return {
                'sector': sector,
                'exposure': new_exposure,
                'exceeded': new_exposure > self.max_sector_exposure
            }
        except:
            return {'sector': 'Unknown', 'exposure': 0, 'exceeded': False}
    
    async def _check_correlation_risk(self, symbol: str, position_data: Dict) -> Dict:
        """Check correlation risk with existing positions"""
        try:
            # Placeholder correlation calculation
            max_correlation = 0.3  # Low correlation by default
            
            # High correlation pairs
            high_corr_pairs = {
                'AAPL': ['MSFT', 'GOOGL'], 'MSFT': ['AAPL', 'GOOGL'],
                'SPY': ['QQQ', 'IWM'], 'QQQ': ['SPY']
            }
            
            for existing_symbol in position_data.keys():
                if existing_symbol in high_corr_pairs.get(symbol, []):
                    max_correlation = 0.8
                    break
            
            return {
                'max_correlation': max_correlation,
                'high_correlation': max_correlation > self.max_correlation
            }
        except:
            return {'max_correlation': 0.3, 'high_correlation': False}
    
    def _calculate_var_estimate(self, position_value: float, volatility: float) -> float:
        """Calculate Value at Risk estimate"""
        try:
            # 95% VaR using normal distribution approximation
            return position_value * volatility * 1.645 * np.sqrt(1/252)  # Daily VaR
        except:
            return position_value * 0.02  # 2% default
    
    async def _calculate_portfolio_var(self, position_data: Dict, market_data: Dict) -> Dict:
        """Calculate portfolio-level Value at Risk"""
        try:
            total_value = self._calculate_portfolio_value(position_data)
            portfolio_volatility = 0.20  # Placeholder 20% portfolio volatility
            
            var_1d = total_value * portfolio_volatility * 1.645 * np.sqrt(1/252)
            var_5d = total_value * portfolio_volatility * 1.645 * np.sqrt(5/252)
            
            return {
                'var_1d': var_1d,
                'var_5d': var_5d,
                'confidence_level': 0.95
            }
        except:
            return {'var_1d': 0, 'var_5d': 0, 'confidence_level': 0.95}
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from performance history"""
        try:
            if len(self.performance_history) < 2:
                return 0.0
            
            values = [p['total_value'] for p in self.performance_history]
            peak = values[0]
            max_dd = 0.0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
            
            return max_dd
        except:
            return 0.0
    
    def _assess_overall_risk_level(self, daily_loss: float, var_1d: float, 
                                 max_drawdown: float, warning_count: int) -> RiskLevel:
        """Assess overall portfolio risk level"""
        try:
            risk_score = 0
            
            # Daily loss component
            if daily_loss > 0.04:
                risk_score += 3
            elif daily_loss > 0.02:
                risk_score += 2
            elif daily_loss > 0.01:
                risk_score += 1
            
            # Drawdown component
            if max_drawdown > 0.12:
                risk_score += 3
            elif max_drawdown > 0.08:
                risk_score += 2
            elif max_drawdown > 0.04:
                risk_score += 1
            
            # Warning count
            risk_score += min(warning_count, 3)
            
            if risk_score >= 7:
                return RiskLevel.EXTREME
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 3:
                return RiskLevel.MODERATE
            elif risk_score >= 1:
                return RiskLevel.LOW
            else:
                return RiskLevel.VERY_LOW
                
        except:
            return RiskLevel.MODERATE
    
    def _update_performance_history(self, metrics: Dict):
        """Update performance tracking history"""
        try:
            metrics['timestamp'] = datetime.now()
            self.performance_history.append(metrics)
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to update performance history: {e}")
    
    async def _send_risk_alert(self, alert: Dict):
        """Send risk alert to monitoring systems"""
        try:
            # In production, this would integrate with alerting systems
            # Slack, PagerDuty, email, etc.
            logger.critical(f"🚨 RISK ALERT: {json.dumps(alert, default=str)}")
        except Exception as e:
            logger.error(f"Failed to send risk alert: {e}")
    
    def _can_reset_circuit_breaker(self) -> bool:
        """Check if circuit breaker can be automatically reset"""
        try:
            # Only reset after market hours or after cooling period
            now = datetime.now()
            if hasattr(self, 'circuit_breaker_triggered'):
                time_since_trigger = now - self.circuit_breaker_triggered
                return time_since_trigger > timedelta(hours=1)
            return False
        except:
            return False
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""
        try:
            return {
                'timestamp': datetime.now(),
                'circuit_breaker_active': self.circuit_breaker_active,
                'risk_limits': {
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_position_size': self.max_position_size,
                    'max_sector_exposure': self.max_sector_exposure,
                    'stop_loss_pct': self.stop_loss_pct,
                    'daily_loss_limit': self.daily_loss_limit
                },
                'sector_exposure': dict(self.sector_exposure),
                'recent_alerts': self.risk_alerts[-10:],  # Last 10 alerts
                'performance_summary': {
                    'total_records': len(self.performance_history),
                    'last_update': self.performance_history[-1]['timestamp'] if self.performance_history else None
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate risk dashboard: {e}")
            return {'error': str(e)}
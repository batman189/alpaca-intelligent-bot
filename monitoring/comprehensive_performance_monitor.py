"""
Comprehensive Performance Monitoring System
Real-time metrics, alerts, and performance tracking for production trading bot
"""

import logging
import asyncio
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    trades_executed: int
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    average_trade_duration: float
    system_latency_ms: float
    memory_usage_mb: float
    cpu_usage_pct: float
    active_positions: int
    daily_volume: float

@dataclass
class TradingAlert:
    timestamp: datetime
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    category: str  # PERFORMANCE, RISK, SYSTEM, MARKET
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False

class ComprehensivePerformanceMonitor:
    """Enterprise-grade performance monitoring and alerting system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Performance tracking
        self.metrics_history = deque(maxlen=10000)  # Last 10k metrics points
        self.trade_history = deque(maxlen=5000)     # Last 5k trades
        self.alerts = deque(maxlen=1000)            # Last 1k alerts
        
        # Real-time counters
        self.trades_today = 0
        self.total_pnl_today = 0.0
        self.start_time = datetime.now()
        
        # Performance thresholds
        self.latency_threshold_ms = config.get('latency_threshold_ms', 100)
        self.memory_threshold_mb = config.get('memory_threshold_mb', 1024)
        self.cpu_threshold_pct = config.get('cpu_threshold_pct', 80)
        self.drawdown_threshold_pct = config.get('drawdown_threshold_pct', 0.10)
        self.min_win_rate = config.get('min_win_rate', 0.45)
        
        # Monitoring state
        self.monitoring_active = True
        self.last_health_check = datetime.now()
        
        logger.info("📊 Comprehensive Performance Monitor initialized")
    
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        try:
            logger.info("🚀 Starting performance monitoring...")
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._monitor_system_metrics()),
                asyncio.create_task(self._monitor_trading_performance()),
                asyncio.create_task(self._monitor_risk_metrics()),
                asyncio.create_task(self._generate_periodic_reports()),
                asyncio.create_task(self._cleanup_old_data())
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            await self._create_alert("CRITICAL", "SYSTEM", f"Performance monitoring failed: {e}")
    
    async def _monitor_system_metrics(self):
        """Monitor system resource usage"""
        while self.monitoring_active:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                disk_usage = psutil.disk_usage('/')
                
                # Network metrics
                network_stats = psutil.net_io_counters()
                
                # Trading bot specific metrics
                latency = await self._measure_system_latency()
                
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_usage_pct': cpu_percent,
                    'memory_usage_mb': memory_info.used / 1024 / 1024,
                    'memory_available_mb': memory_info.available / 1024 / 1024,
                    'disk_usage_pct': disk_usage.percent,
                    'network_bytes_sent': network_stats.bytes_sent,
                    'network_bytes_recv': network_stats.bytes_recv,
                    'system_latency_ms': latency
                }
                
                # Check thresholds
                if cpu_percent > self.cpu_threshold_pct:
                    await self._create_alert("WARNING", "SYSTEM", f"High CPU usage: {cpu_percent:.1f}%", metrics)
                
                if memory_info.used / 1024 / 1024 > self.memory_threshold_mb:
                    await self._create_alert("WARNING", "SYSTEM", f"High memory usage: {memory_info.used / 1024 / 1024:.1f}MB", metrics)
                
                if latency > self.latency_threshold_ms:
                    await self._create_alert("WARNING", "PERFORMANCE", f"High system latency: {latency:.1f}ms", metrics)
                
                # Store metrics
                await self._store_metrics(metrics)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"System metrics monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_trading_performance(self):
        """Monitor trading performance metrics"""
        while self.monitoring_active:
            try:
                # Calculate performance metrics
                performance = await self._calculate_trading_performance()
                
                # Check performance thresholds
                if performance['win_rate'] < self.min_win_rate and len(self.trade_history) > 50:
                    await self._create_alert("WARNING", "PERFORMANCE", 
                                           f"Low win rate: {performance['win_rate']:.2%}", performance)
                
                if performance['max_drawdown'] > self.drawdown_threshold_pct:
                    await self._create_alert("ERROR", "RISK", 
                                           f"High drawdown: {performance['max_drawdown']:.2%}", performance)
                
                if performance['sharpe_ratio'] < 0.5 and len(self.trade_history) > 100:
                    await self._create_alert("WARNING", "PERFORMANCE", 
                                           f"Low Sharpe ratio: {performance['sharpe_ratio']:.2f}", performance)
                
                logger.debug(f"📈 Trading performance - Win rate: {performance['win_rate']:.1%}, "
                           f"Sharpe: {performance['sharpe_ratio']:.2f}, Drawdown: {performance['max_drawdown']:.1%}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Trading performance monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_risk_metrics(self):
        """Monitor risk-related metrics"""
        while self.monitoring_active:
            try:
                risk_metrics = await self._calculate_risk_metrics()
                
                # Portfolio concentration risk
                if risk_metrics['max_position_concentration'] > 0.20:  # 20% max
                    await self._create_alert("WARNING", "RISK", 
                                           f"High position concentration: {risk_metrics['max_position_concentration']:.1%}")
                
                # Correlation risk
                if risk_metrics['avg_correlation'] > 0.80:  # 80% max correlation
                    await self._create_alert("WARNING", "RISK", 
                                           f"High portfolio correlation: {risk_metrics['avg_correlation']:.2f}")
                
                # Volatility risk
                if risk_metrics['portfolio_volatility'] > 0.30:  # 30% max volatility
                    await self._create_alert("WARNING", "RISK", 
                                           f"High portfolio volatility: {risk_metrics['portfolio_volatility']:.1%}")
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Risk metrics monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _generate_periodic_reports(self):
        """Generate periodic performance reports"""
        while self.monitoring_active:
            try:
                # Generate hourly report
                if datetime.now().minute == 0:
                    await self._generate_hourly_report()
                
                # Generate daily report
                if datetime.now().hour == 16 and datetime.now().minute == 0:  # 4 PM market close
                    await self._generate_daily_report()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        while self.monitoring_active:
            try:
                # Clean up data older than 30 days
                cutoff_date = datetime.now() - timedelta(days=30)
                
                # Clean alerts
                self.alerts = deque([alert for alert in self.alerts if alert.timestamp > cutoff_date], 
                                  maxlen=1000)
                
                # Clean metrics (keep last 7 days of detailed metrics)
                week_cutoff = datetime.now() - timedelta(days=7)
                self.metrics_history = deque([m for m in self.metrics_history if m['timestamp'] > week_cutoff], 
                                           maxlen=10000)
                
                logger.debug("🧹 Cleaned up old monitoring data")
                
                await asyncio.sleep(86400)  # Clean daily
                
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(86400)
    
    async def record_trade_execution(self, trade_data: Dict[str, Any]):
        """Record trade execution for performance tracking"""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'quantity': trade_data.get('quantity', 0),
                'price': trade_data.get('price', 0),
                'pnl': trade_data.get('pnl', 0),
                'duration_minutes': trade_data.get('duration_minutes', 0),
                'strategy': trade_data.get('strategy'),
                'execution_latency_ms': trade_data.get('execution_latency_ms', 0)
            }
            
            self.trade_history.append(trade_record)
            
            # Update daily counters
            self.trades_today += 1
            self.total_pnl_today += trade_record['pnl']
            
            logger.debug(f"📝 Recorded trade: {trade_record['symbol']} {trade_record['side']} "
                        f"{trade_record['quantity']} @ ${trade_record['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to record trade execution: {e}")
    
    async def _measure_system_latency(self) -> float:
        """Measure system response latency"""
        try:
            start_time = time.time()
            
            # Simulate a typical system operation
            await asyncio.sleep(0.001)  # 1ms simulated operation
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            return latency_ms
            
        except Exception as e:
            logger.error(f"Latency measurement failed: {e}")
            return 0.0
    
    async def _calculate_trading_performance(self) -> Dict[str, float]:
        """Calculate comprehensive trading performance metrics"""
        try:
            if not self.trade_history:
                return {
                    'trades_count': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl_per_trade': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'avg_trade_duration': 0.0
                }
            
            trades_df = pd.DataFrame(list(self.trade_history))
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
            
            # Sharpe ratio calculation
            if len(trades_df) > 10:
                returns = trades_df['pnl']
                sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Maximum drawdown
            cumulative_pnl = trades_df['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max)
            max_drawdown = abs(drawdown.min()) / running_max.max() if running_max.max() > 0 else 0.0
            
            # Average trade duration
            avg_duration = trades_df['duration_minutes'].mean()
            
            return {
                'trades_count': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': avg_pnl,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade_duration': avg_duration
            }
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {'trades_count': 0, 'win_rate': 0.0, 'total_pnl': 0.0}
    
    async def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        try:
            # Placeholder risk calculations
            # In production, would integrate with portfolio manager
            
            return {
                'max_position_concentration': 0.10,  # 10% max position
                'avg_correlation': 0.40,             # 40% average correlation
                'portfolio_volatility': 0.18,       # 18% portfolio volatility
                'beta_to_market': 0.85,             # 0.85 beta to SPY
                'var_95_1d': 2500.0,                # $2,500 daily VaR
                'sharpe_ratio': 1.2                 # Portfolio Sharpe
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    async def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics for historical analysis"""
        try:
            self.metrics_history.append(metrics)
            
            # In production, would also store to database
            # await self._store_to_database(metrics)
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def _create_alert(self, severity: str, category: str, message: str, details: Dict = None):
        """Create and store performance alert"""
        try:
            alert = TradingAlert(
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                message=message,
                details=details or {}
            )
            
            self.alerts.append(alert)
            
            # Log based on severity
            if severity == "CRITICAL":
                logger.critical(f"🚨 {category}: {message}")
            elif severity == "ERROR":
                logger.error(f"❌ {category}: {message}")
            elif severity == "WARNING":
                logger.warning(f"⚠️ {category}: {message}")
            else:
                logger.info(f"ℹ️ {category}: {message}")
            
            # In production, would send to alerting systems
            # await self._send_alert_notification(alert)
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def _generate_hourly_report(self):
        """Generate hourly performance report"""
        try:
            performance = await self._calculate_trading_performance()
            
            report = {
                'timestamp': datetime.now(),
                'period': 'hourly',
                'trades_executed': performance['trades_count'],
                'total_pnl': performance['total_pnl'],
                'win_rate': performance['win_rate'],
                'active_alerts': len([a for a in self.alerts if not a.acknowledged])
            }
            
            logger.info(f"📊 Hourly Report - Trades: {report['trades_executed']}, "
                       f"P&L: ${report['total_pnl']:.2f}, Win Rate: {report['win_rate']:.1%}")
            
            # Store report
            # await self._store_report(report)
            
        except Exception as e:
            logger.error(f"Hourly report generation failed: {e}")
    
    async def _generate_daily_report(self):
        """Generate comprehensive daily report"""
        try:
            performance = await self._calculate_trading_performance()
            risk_metrics = await self._calculate_risk_metrics()
            
            # Get system metrics summary
            if self.metrics_history:
                recent_metrics = list(self.metrics_history)[-100:]  # Last 100 points
                avg_cpu = np.mean([m['cpu_usage_pct'] for m in recent_metrics])
                avg_memory = np.mean([m['memory_usage_mb'] for m in recent_metrics])
                avg_latency = np.mean([m['system_latency_ms'] for m in recent_metrics])
            else:
                avg_cpu = avg_memory = avg_latency = 0
            
            daily_report = {
                'timestamp': datetime.now(),
                'period': 'daily',
                'trading_performance': performance,
                'risk_metrics': risk_metrics,
                'system_performance': {
                    'avg_cpu_usage': avg_cpu,
                    'avg_memory_usage': avg_memory,
                    'avg_latency': avg_latency
                },
                'alerts_summary': {
                    'total_alerts': len(self.alerts),
                    'critical_alerts': len([a for a in self.alerts if a.severity == "CRITICAL"]),
                    'unacknowledged_alerts': len([a for a in self.alerts if not a.acknowledged])
                }
            }
            
            logger.info(f"📈 Daily Report Generated - Performance: {performance['win_rate']:.1%} win rate, "
                       f"${performance['total_pnl']:.2f} P&L, {performance['sharpe_ratio']:.2f} Sharpe")
            
            # Store comprehensive daily report
            # await self._store_daily_report(daily_report)
            
            # Send daily summary alert
            await self._create_alert("INFO", "PERFORMANCE", "Daily performance report generated", daily_report)
            
        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get real-time performance dashboard data"""
        try:
            current_performance = asyncio.run(self._calculate_trading_performance())
            
            # Recent alerts
            recent_alerts = [asdict(alert) for alert in list(self.alerts)[-10:]]
            
            # System status
            system_status = "HEALTHY"
            critical_alerts = [a for a in self.alerts if a.severity == "CRITICAL" and not a.acknowledged]
            if critical_alerts:
                system_status = "CRITICAL"
            elif len([a for a in self.alerts if a.severity == "ERROR" and not a.acknowledged]) > 0:
                system_status = "WARNING"
            
            return {
                'timestamp': datetime.now(),
                'system_status': system_status,
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'trades_today': self.trades_today,
                'pnl_today': self.total_pnl_today,
                'performance_metrics': current_performance,
                'recent_alerts': recent_alerts,
                'monitoring_active': self.monitoring_active,
                'data_points_collected': len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {
                'timestamp': datetime.now(),
                'system_status': 'ERROR',
                'error': str(e)
            }
    
    async def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert"""
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].acknowledged = True
                logger.info(f"✅ Alert acknowledged: {self.alerts[alert_index].message}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            self.monitoring_active = False
            logger.info("🛑 Performance monitoring stopped")
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")

# Global performance monitor instance
performance_monitor = None

async def initialize_performance_monitor(config: Dict[str, Any]):
    """Initialize global performance monitor"""
    global performance_monitor
    try:
        performance_monitor = ComprehensivePerformanceMonitor(config)
        logger.info("✅ Performance monitor initialized")
        return performance_monitor
    except Exception as e:
        logger.error(f"Failed to initialize performance monitor: {e}")
        return None
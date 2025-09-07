"""
Comprehensive Logger - Complete Opportunity Tracking System
Tracks everything: opportunities, trades, misses, system health
FIXED VERSION - Resolved logger initialization issues
"""

import logging
import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import threading
import warnings
warnings.filterwarnings('ignore')

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class OpportunityRecord:
    timestamp: datetime
    symbol: str
    signal_type: str
    confidence: float
    direction: str
    executed: bool
    metadata: Dict[str, Any]

@dataclass
class SystemError:
    timestamp: datetime
    component: str
    error_type: str
    error_message: str
    metadata: Dict[str, Any]

class ComprehensiveLogger:
    def __init__(self):
        """Initialize comprehensive logging system"""
        
        # Initialize logger FIRST
        self.logger = logging.getLogger(__name__)
        
        # Initialize other attributes
        self.db_path = "logs/trading_data.db"
        self.session_start = datetime.now()
        self.opportunity_cache = defaultdict(list)
        self.performance_cache = {}
        
        # Thread safety
        self.db_lock = threading.Lock()
        
        # Statistics tracking
        self.session_stats = {
            'opportunities_detected': 0,
            'opportunities_executed': 0,
            'opportunities_missed': 0,
            'system_errors': 0,
            'trades_executed': 0
        }
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Initialize database AFTER logger is set up
        self._init_database()
        
        self.logger.info("📊 Comprehensive Logger initialized")

    def _init_database(self):
        """Initialize SQLite database for logging"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create opportunities table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS opportunities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_type TEXT,
                        confidence REAL,
                        direction TEXT,
                        executed BOOLEAN DEFAULT FALSE,
                        price_target REAL,
                        stop_loss REAL,
                        metadata TEXT,
                        session_id TEXT
                    )
                ''')
                
                # Create system errors table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        component TEXT NOT NULL,
                        error_type TEXT,
                        error_message TEXT,
                        severity TEXT DEFAULT 'ERROR',
                        metadata TEXT,
                        session_id TEXT
                    )
                ''')
                
                # Create trade history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity REAL,
                        price REAL,
                        strategy TEXT,
                        pnl REAL,
                        success BOOLEAN,
                        metadata TEXT,
                        session_id TEXT
                    )
                ''')
                
                # Create system health table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        component TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        status TEXT,
                        metadata TEXT,
                        session_id TEXT
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunities_symbol ON opportunities(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunities_timestamp ON opportunities(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_history(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trade_history(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_component ON system_errors(component)')
                
                conn.commit()
                conn.close()
                
                # Safe logger access
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            # Safe logger access with fallback
            logger = getattr(self, 'logger', logging.getLogger(__name__))
            logger.error(f"❌ Database initialization failed: {e}")
            # Continue with degraded functionality rather than crashing

    def log_opportunity_detected(self, symbol: str, signal_type: str, confidence: float, 
                               metadata: Dict[str, Any] = None):
        """Log when an opportunity is detected"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO opportunities 
                    (timestamp, symbol, signal_type, confidence, executed, metadata, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    symbol,
                    signal_type,
                    confidence,
                    False,
                    json.dumps(metadata or {}),
                    str(self.session_start)
                ))
                
                conn.commit()
                conn.close()
                
                self.session_stats['opportunities_detected'] += 1
                self.logger.info(f"🎯 Opportunity detected: {symbol} {signal_type} ({confidence:.1f}%)")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log opportunity: {e}")

    def log_opportunity_executed(self, symbol: str, signal_type: str, confidence: float, 
                               price: float, quantity: float, metadata: Dict[str, Any] = None):
        """Log when an opportunity is executed"""
        try:
            with self.db_lock:
                # Update opportunity as executed
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE opportunities 
                    SET executed = TRUE 
                    WHERE symbol = ? AND signal_type = ? AND executed = FALSE
                    ORDER BY timestamp DESC LIMIT 1
                ''', (symbol, signal_type))
                
                # Log trade execution
                cursor.execute('''
                    INSERT INTO trade_history 
                    (timestamp, symbol, action, quantity, price, strategy, metadata, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    symbol,
                    'BUY',  # Simplified
                    quantity,
                    price,
                    signal_type,
                    json.dumps(metadata or {}),
                    str(self.session_start)
                ))
                
                conn.commit()
                conn.close()
                
                self.session_stats['opportunities_executed'] += 1
                self.session_stats['trades_executed'] += 1
                self.logger.info(f"✅ Opportunity executed: {symbol} at ${price:.2f}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log execution: {e}")

    def log_opportunity_missed(self, symbol: str, signal_type: str, reason: str, 
                             confidence: float = 0, metadata: Dict[str, Any] = None):
        """Log when an opportunity is missed"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                miss_metadata = metadata or {}
                miss_metadata['miss_reason'] = reason
                
                cursor.execute('''
                    INSERT INTO opportunities 
                    (timestamp, symbol, signal_type, confidence, executed, metadata, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    symbol,
                    signal_type,
                    confidence,
                    False,
                    json.dumps(miss_metadata),
                    str(self.session_start)
                ))
                
                conn.commit()
                conn.close()
                
                self.session_stats['opportunities_missed'] += 1
                self.logger.warning(f"⚠️ Opportunity missed: {symbol} {signal_type} - {reason}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log missed opportunity: {e}")

    def log_system_error(self, component: str, error_message: str, 
                        metadata: Dict[str, Any] = None, severity: str = "ERROR"):
        """Log system errors"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_errors 
                    (timestamp, component, error_message, severity, metadata, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    component,
                    error_message,
                    severity,
                    json.dumps(metadata or {}),
                    str(self.session_start)
                ))
                
                conn.commit()
                conn.close()
                
                self.session_stats['system_errors'] += 1
                self.logger.error(f"🚨 System error in {component}: {error_message}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log system error: {e}")

    def log_trade_entry(self, symbol: str, action: str, quantity: float, price: float,
                       strategy: str, metadata: Dict[str, Any] = None):
        """Log trade entry"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trade_history 
                    (timestamp, symbol, action, quantity, price, strategy, metadata, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    symbol,
                    action,
                    quantity,
                    price,
                    strategy,
                    json.dumps(metadata or {}),
                    str(self.session_start)
                ))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"📈 Trade logged: {action} {quantity} {symbol} @ ${price:.2f}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log trade: {e}")

    def log_system_health(self, component: str, metric_name: str, metric_value: float,
                         status: str = "OK", metadata: Dict[str, Any] = None):
        """Log system health metrics"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_health 
                    (timestamp, component, metric_name, metric_value, status, metadata, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    component,
                    metric_name,
                    metric_value,
                    status,
                    json.dumps(metadata or {}),
                    str(self.session_start)
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log health metric: {e}")

    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get session statistics
                now = datetime.now()
                session_duration = (now - self.session_start).total_seconds() / 3600  # hours
                
                # Calculate opportunities per hour
                opportunities_per_hour = (
                    self.session_stats['opportunities_detected'] / session_duration 
                    if session_duration > 0 else 0
                )
                
                # Calculate capture rate
                total_opportunities = self.session_stats['opportunities_detected']
                executed_opportunities = self.session_stats['opportunities_executed']
                capture_rate = (
                    (executed_opportunities / total_opportunities * 100) 
                    if total_opportunities > 0 else 0
                )
                
                # Calculate execution rate
                execution_rate = (
                    (self.session_stats['trades_executed'] / executed_opportunities * 100)
                    if executed_opportunities > 0 else 0
                )
                
                # Get recent error count
                cursor.execute('''
                    SELECT COUNT(*) FROM system_errors 
                    WHERE timestamp > datetime('now', '-1 hour')
                ''')
                recent_errors = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'session_duration_hours': session_duration,
                    'opportunities_detected': self.session_stats['opportunities_detected'],
                    'opportunities_executed': self.session_stats['opportunities_executed'],
                    'opportunities_missed': self.session_stats['opportunities_missed'],
                    'trades_executed': self.session_stats['trades_executed'],
                    'system_errors': self.session_stats['system_errors'],
                    'opportunities_per_hour': opportunities_per_hour,
                    'capture_rate_percent': capture_rate,
                    'execution_rate_percent': execution_rate,
                    'recent_errors': recent_errors,
                    'last_updated': now.isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate health report: {e}")
            return {}

    def get_missed_opportunity_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze missed opportunities"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                cursor.execute('''
                    SELECT metadata FROM opportunities 
                    WHERE executed = FALSE 
                    AND timestamp > ? 
                    AND json_extract(metadata, '$.miss_reason') IS NOT NULL
                ''', (cutoff_time.isoformat(),))
                
                missed_opportunities = cursor.fetchall()
                
                # Analyze miss reasons
                reasons = defaultdict(int)
                for (metadata_json,) in missed_opportunities:
                    try:
                        metadata = json.loads(metadata_json)
                        reason = metadata.get('miss_reason', 'unknown')
                        reasons[reason] += 1
                    except:
                        reasons['unknown'] += 1
                
                total_missed = len(missed_opportunities)
                
                conn.close()
                
                return {
                    'total_missed': total_missed,
                    'time_period_hours': hours,
                    'reasons': dict(reasons),
                    'top_reason': max(reasons, key=reasons.get) if reasons else None
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze missed opportunities: {e}")
            return {}

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get daily trading summary"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                today = datetime.now().date()
                
                # Get today's opportunities
                cursor.execute('''
                    SELECT COUNT(*) FROM opportunities 
                    WHERE date(timestamp) = ?
                ''', (today,))
                opportunities_today = cursor.fetchone()[0]
                
                # Get today's executions
                cursor.execute('''
                    SELECT COUNT(*) FROM opportunities 
                    WHERE date(timestamp) = ? AND executed = TRUE
                ''', (today,))
                executed_today = cursor.fetchone()[0]
                
                # Get today's trades
                cursor.execute('''
                    SELECT COUNT(*) FROM trade_history 
                    WHERE date(timestamp) = ?
                ''', (today,))
                trades_today = cursor.fetchone()[0]
                
                # Calculate capture rate
                capture_rate = (
                    (executed_today / opportunities_today * 100) 
                    if opportunities_today > 0 else 0
                )
                
                conn.close()
                
                return {
                    'date': today.isoformat(),
                    'opportunities_detected': opportunities_today,
                    'opportunities_executed': executed_today,
                    'trades_executed': trades_today,
                    'capture_rate': capture_rate
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate daily summary: {e}")
            return {}

    def export_session_data(self, format: str = "json") -> str:
        """Export session data to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/session_export_{timestamp}.{format}"
            
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                
                # Get all session data
                session_data = {
                    'session_start': self.session_start.isoformat(),
                    'export_time': datetime.now().isoformat(),
                    'statistics': self.session_stats,
                    'health_report': self.get_system_health_report(),
                    'missed_analysis': self.get_missed_opportunity_analysis(),
                    'daily_summary': self.get_daily_summary()
                }
                
                conn.close()
                
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2, default=str)
                
                self.logger.info(f"📁 Session data exported to {filename}")
                return filename
                
        except Exception as e:
            self.logger.error(f"❌ Failed to export session data: {e}")
            return ""

    def cleanup_old_data(self, days: int = 30):
        """Clean up data older than specified days"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Clean up old records
                tables = ['opportunities', 'system_errors', 'trade_history', 'system_health']
                total_deleted = 0
                
                for table in tables:
                    cursor.execute(f'''
                        DELETE FROM {table} 
                        WHERE timestamp < ?
                    ''', (cutoff_date.isoformat(),))
                    deleted = cursor.rowcount
                    total_deleted += deleted
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"🧹 Cleaned up {total_deleted} old records")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old data: {e}")

# For backward compatibility
Logger = ComprehensiveLogger

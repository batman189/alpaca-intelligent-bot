"""
Comprehensive Logging System - Track Everything, Miss Nothing
Advanced logging for opportunities, trades, misses, and system health
UPDATED VERSION - Fixed thread safety and database race conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import threading
from collections import defaultdict, deque
import warnings
import atexit
warnings.filterwarnings('ignore')

class EventType(Enum):
    OPPORTUNITY_DETECTED = "opportunity_detected"
    OPPORTUNITY_EXECUTED = "opportunity_executed"
    OPPORTUNITY_MISSED = "opportunity_missed"
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    SYSTEM_ERROR = "system_error"
    DATA_FETCH_ERROR = "data_fetch_error"
    VALIDATION_FAILURE = "validation_failure"
    RISK_LIMIT_HIT = "risk_limit_hit"
    MARKET_CONDITION_CHANGE = "market_condition_change"

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class TradingEvent:
    timestamp: datetime
    event_type: EventType
    symbol: str
    log_level: LogLevel
    message: str
    metadata: Dict[str, Any]
    session_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['log_level'] = self.log_level.value
        return data

class ComprehensiveLogger:
    def __init__(self, log_directory: str = "logs"):
        """Initialize comprehensive logging system with thread safety"""
        
        # Setup directory structure
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_directory / "opportunities").mkdir(exist_ok=True)
        (self.log_directory / "trades").mkdir(exist_ok=True)
        (self.log_directory / "errors").mkdir(exist_ok=True)
        (self.log_directory / "performance").mkdir(exist_ok=True)
        (self.log_directory / "system").mkdir(exist_ok=True)
        
        # Initialize SQLite database for structured logging
        self.db_path = self.log_directory / "trading_events.db"
        
        # Thread safety
        self.db_lock = threading.RLock()  # Reentrant lock for database operations
        self.cache_lock = threading.Lock()  # Lock for in-memory cache
        
        # Initialize database with thread safety
        self._init_database()
        
        # In-memory storage for real-time analytics
        self.recent_events = deque(maxlen=1000)  # Last 1000 events
        self.opportunity_cache = defaultdict(list)
        self.performance_metrics = defaultdict(float)
        
        # Session tracking
        self.session_id = f"session_{int(datetime.now().timestamp())}"
        self.session_start = datetime.now()
        
        # Configure Python logging
        self._setup_python_logging()
        
        # Performance tracking
        self.capture_rate_window = 50  # Track last 50 opportunities
        self.missed_opportunities = []
        self.executed_opportunities = []
        
        # Database connection pool
        self._connection_pool = []
        self._pool_size = 5
        self._init_connection_pool()
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
        
        self.logger.info(f"🔍 Comprehensive Logger initialized - Session: {self.session_id}")

    def _init_database(self):
        """Initialize SQLite database for event storage with thread safety"""
        with self.db_lock:
            try:
                # Check if database already exists and is valid
                if self.db_path.exists():
                    try:
                        with sqlite3.connect(self.db_path) as conn:
                            # Test the database by querying a table
                            conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                            self.logger.info("✅ Existing database validated")
                            return
                    except sqlite3.Error:
                        self.logger.warning("⚠️ Database corrupted, recreating...")
                        self.db_path.unlink()  # Delete corrupted database
                
                # Create new database
                with sqlite3.connect(self.db_path) as conn:
                    # Enable WAL mode for better concurrency
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=memory")
                    
                    # Create tables
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS trading_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            log_level TEXT NOT NULL,
                            message TEXT NOT NULL,
                            metadata TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS opportunities (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            signal_type TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            direction TEXT NOT NULL,
                            executed INTEGER DEFAULT 0,
                            reason_skipped TEXT,
                            metadata TEXT NOT NULL,
                            session_id TEXT NOT NULL
                        )
                    """)
                    
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS performance_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            metric_name TEXT NOT NULL,
                            metric_value REAL NOT NULL,
                            timeframe TEXT NOT NULL,
                            session_id TEXT NOT NULL
                        )
                    """)
                    
                    # Create indexes for better query performance
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON trading_events(timestamp)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_symbol ON trading_events(symbol)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON trading_events(session_id)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_opportunities_symbol ON opportunities(symbol)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_opportunities_executed ON opportunities(executed)")
                    
                    conn.commit()
                    self.logger.info("✅ Database initialized successfully")
                    
            except Exception as e:
                self.logger.error(f"❌ Database initialization failed: {e}")
                raise

    def _init_connection_pool(self):
        """Initialize database connection pool"""
        try:
            for _ in range(self._pool_size):
                conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0
                )
                conn.execute("PRAGMA journal_mode=WAL")
                self._connection_pool.append(conn)
        except Exception as e:
            self.logger.error(f"❌ Connection pool initialization failed: {e}")

    def _get_connection(self):
        """Get a database connection from pool"""
        with self.db_lock:
            if self._connection_pool:
                return self._connection_pool.pop()
            else:
                # Create new connection if pool is empty
                conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0
                )
                conn.execute("PRAGMA journal_mode=WAL")
                return conn

    def _return_connection(self, conn):
        """Return connection to pool"""
        with self.db_lock:
            if len(self._connection_pool) < self._pool_size:
                self._connection_pool.append(conn)
            else:
                conn.close()

    def _setup_python_logging(self):
        """Setup Python logging configuration"""
        # Create logger
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        try:
            # File handlers for different log levels
            debug_handler = logging.FileHandler(self.log_directory / "debug.log")
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(detailed_formatter)
            
            info_handler = logging.FileHandler(self.log_directory / "trading.log")
            info_handler.setLevel(logging.INFO)
            info_handler.setFormatter(simple_formatter)
            
            error_handler = logging.FileHandler(self.log_directory / "errors.log")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            
            # Add handlers to logger
            self.logger.addHandler(debug_handler)
            self.logger.addHandler(info_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"❌ Error setting up logging handlers: {e}")

    def log_opportunity_detected(self, symbol: str, signal_type: str, confidence: float,
                               direction: str, metadata: Dict[str, Any] = None):
        """Log detected opportunity with thread safety"""
        try:
            if metadata is None:
                metadata = {}
                
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=EventType.OPPORTUNITY_DETECTED,
                symbol=symbol,
                log_level=LogLevel.INFO,
                message=f"Opportunity detected: {signal_type} signal for {symbol} "
                       f"({direction}, confidence: {confidence}%)",
                metadata={
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'direction': direction,
                    **metadata
                },
                session_id=self.session_id
            )
            
            self._log_event(event)
            
            # Store in opportunities table
            self._insert_opportunity(symbol, signal_type, confidence, direction, metadata)
            
            # Update opportunity cache (thread-safe)
            with self.cache_lock:
                self.opportunity_cache[symbol].append(event)
                # Limit cache size per symbol
                if len(self.opportunity_cache[symbol]) > 100:
                    self.opportunity_cache[symbol] = self.opportunity_cache[symbol][-50:]
                    
        except Exception as e:
            self.logger.error(f"❌ Error logging opportunity detected: {e}")

    def _insert_opportunity(self, symbol: str, signal_type: str, confidence: float, 
                          direction: str, metadata: Dict[str, Any]):
        """Insert opportunity into database with proper error handling"""
        conn = None
        try:
            conn = self._get_connection()
            conn.execute("""
                INSERT INTO opportunities 
                (timestamp, symbol, signal_type, confidence, direction, metadata, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                symbol,
                signal_type,
                confidence,
                direction,
                json.dumps(metadata),
                self.session_id
            ))
            conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Database insert error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._return_connection(conn)

    def log_opportunity_executed(self, symbol: str, signal_type: str, confidence: float,
                               entry_price: float, position_size: float, metadata: Dict[str, Any] = None):
        """Log executed opportunity with thread safety"""
        try:
            if metadata is None:
                metadata = {}
                
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=EventType.OPPORTUNITY_EXECUTED,
                symbol=symbol,
                log_level=LogLevel.INFO,
                message=f"Opportunity executed: {symbol} at ${entry_price} "
                       f"(size: {position_size}, confidence: {confidence}%)",
                metadata={
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    **metadata
                },
                session_id=self.session_id
            )
            
            self._log_event(event)
            
            with self.cache_lock:
                self.executed_opportunities.append(event)
                # Limit cache size
                if len(self.executed_opportunities) > 1000:
                    self.executed_opportunities = self.executed_opportunities[-500:]
            
            # Update opportunities table
            self._update_opportunity_executed(symbol, signal_type)
            
        except Exception as e:
            self.logger.error(f"❌ Error logging opportunity executed: {e}")

    def _update_opportunity_executed(self, symbol: str, signal_type: str):
        """Update opportunity as executed in database"""
        conn = None
        try:
            conn = self._get_connection()
            conn.execute("""
                UPDATE opportunities 
                SET executed = 1 
                WHERE symbol = ? AND signal_type = ? AND executed = 0
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol, signal_type))
            conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Database update error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._return_connection(conn)

    def log_opportunity_missed(self, symbol: str, signal_type: str, reason: str,
                             confidence: float, metadata: Dict[str, Any] = None):
        """Log missed opportunity with reason"""
        try:
            if metadata is None:
                metadata = {}
                
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=EventType.OPPORTUNITY_MISSED,
                symbol=symbol,
                log_level=LogLevel.WARNING,
                message=f"Opportunity missed: {symbol} {signal_type} - Reason: {reason}",
                metadata={
                    'signal_type': signal_type,
                    'reason': reason,
                    'confidence': confidence,
                    **metadata
                },
                session_id=self.session_id
            )
            
            self._log_event(event)
            
            with self.cache_lock:
                self.missed_opportunities.append(event)
                # Limit cache size
                if len(self.missed_opportunities) > 1000:
                    self.missed_opportunities = self.missed_opportunities[-500:]
            
            # Update opportunities table
            self._update_opportunity_missed(symbol, signal_type, reason)
            
        except Exception as e:
            self.logger.error(f"❌ Error logging opportunity missed: {e}")

    def _update_opportunity_missed(self, symbol: str, signal_type: str, reason: str):
        """Update opportunity as missed in database"""
        conn = None
        try:
            conn = self._get_connection()
            conn.execute("""
                UPDATE opportunities 
                SET reason_skipped = ? 
                WHERE symbol = ? AND signal_type = ? AND executed = 0
                ORDER BY timestamp DESC LIMIT 1
            """, (reason, symbol, signal_type))
            conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Database update error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._return_connection(conn)

    def log_trade_entry(self, symbol: str, side: str, quantity: float, price: float,
                       strategy: str, metadata: Dict[str, Any] = None):
        """Log trade entry"""
        try:
            if metadata is None:
                metadata = {}
                
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=EventType.TRADE_ENTRY,
                symbol=symbol,
                log_level=LogLevel.INFO,
                message=f"Trade entry: {side} {quantity} {symbol} at ${price} "
                       f"(strategy: {strategy})",
                metadata={
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'strategy': strategy,
                    **metadata
                },
                session_id=self.session_id
            )
            
            self._log_event(event)
            
        except Exception as e:
            self.logger.error(f"❌ Error logging trade entry: {e}")

    def log_trade_exit(self, symbol: str, side: str, quantity: float, price: float,
                      pnl: float, reason: str, metadata: Dict[str, Any] = None):
        """Log trade exit"""
        try:
            if metadata is None:
                metadata = {}
                
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=EventType.TRADE_EXIT,
                symbol=symbol,
                log_level=LogLevel.INFO,
                message=f"Trade exit: {side} {quantity} {symbol} at ${price} "
                       f"(PnL: ${pnl:.2f}, reason: {reason})",
                metadata={
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'pnl': pnl,
                    'reason': reason,
                    **metadata
                },
                session_id=self.session_id
            )
            
            self._log_event(event)
            
            # Update performance metrics (thread-safe)
            with self.cache_lock:
                self.performance_metrics['total_pnl'] += pnl
                self.performance_metrics['total_trades'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ Error logging trade exit: {e}")

    def log_system_error(self, component: str, error_message: str, 
                        metadata: Dict[str, Any] = None):
        """Log system error"""
        try:
            if metadata is None:
                metadata = {}
                
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=EventType.SYSTEM_ERROR,
                symbol="SYSTEM",
                log_level=LogLevel.ERROR,
                message=f"System error in {component}: {error_message}",
                metadata={
                    'component': component,
                    'error': error_message,
                    **metadata
                },
                session_id=self.session_id
            )
            
            self._log_event(event)
            
        except Exception as e:
            # Use print as fallback if logger fails
            print(f"❌ Critical error in logging system: {e}")

    def log_market_condition_change(self, condition: str, confidence: float,
                                  metadata: Dict[str, Any] = None):
        """Log market condition change"""
        try:
            if metadata is None:
                metadata = {}
                
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=EventType.MARKET_CONDITION_CHANGE,
                symbol="MARKET",
                log_level=LogLevel.INFO,
                message=f"Market condition changed to: {condition} (confidence: {confidence}%)",
                metadata={
                    'condition': condition,
                    'confidence': confidence,
                    **metadata
                },
                session_id=self.session_id
            )
            
            self._log_event(event)
            
        except Exception as e:
            self.logger.error(f"❌ Error logging market condition change: {e}")

    def _log_event(self, event: TradingEvent):
        """Internal method to log event to all storage systems with thread safety"""
        try:
            # Add to recent events (thread-safe)
            with self.cache_lock:
                self.recent_events.append(event)
            
            # Store in database
            self._insert_event(event)
            
            # Log to Python logger
            if event.log_level == LogLevel.DEBUG:
                self.logger.debug(event.message)
            elif event.log_level == LogLevel.INFO:
                self.logger.info(event.message)
            elif event.log_level == LogLevel.WARNING:
                self.logger.warning(event.message)
            elif event.log_level == LogLevel.ERROR:
                self.logger.error(event.message)
            elif event.log_level == LogLevel.CRITICAL:
                self.logger.critical(event.message)
                
        except Exception as e:
            print(f"❌ Critical error in _log_event: {e}")

    def _insert_event(self, event: TradingEvent):
        """Insert event into database with proper error handling"""
        conn = None
        try:
            conn = self._get_connection()
            conn.execute("""
                INSERT INTO trading_events 
                (timestamp, event_type, symbol, log_level, message, metadata, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp.isoformat(),
                event.event_type.value,
                event.symbol,
                event.log_level.value,
                event.message,
                json.dumps(event.metadata),
                event.session_id
            ))
            conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Database insert error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._return_connection(conn)

    def calculate_opportunity_capture_rate(self) -> float:
        """Calculate the rate of opportunities we actually execute"""
        try:
            with self.cache_lock:
                total_opportunities = sum(len(opportunities) for opportunities in self.opportunity_cache.values())
                total_executed = len(self.executed_opportunities)
            
            if total_opportunities == 0:
                return 0.0
                
            return (total_executed / total_opportunities) * 100
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating capture rate: {e}")
            return 0.0

    def get_missed_opportunity_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze missed opportunities in the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.cache_lock:
                recent_missed = [
                    event for event in self.missed_opportunities 
                    if event.timestamp >= cutoff_time
                ]
            
            if not recent_missed:
                return {"total_missed": 0, "reasons": {}, "symbols": {}}
            
            # Analyze reasons for missing opportunities
            reasons = defaultdict(int)
            symbols = defaultdict(int)
            
            for event in recent_missed:
                reason = event.metadata.get('reason', 'Unknown')
                reasons[reason] += 1
                symbols[event.symbol] += 1
            
            return {
                "total_missed": len(recent_missed),
                "reasons": dict(reasons),
                "symbols": dict(symbols),
                "capture_rate": self.calculate_opportunity_capture_rate()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing missed opportunities: {e}")
            return {"total_missed": 0, "reasons": {}, "symbols": {}}

    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        try:
            current_time = datetime.now()
            session_duration = current_time - self.session_start
            
            # Count events by type in last hour
            hour_ago = current_time - timedelta(hours=1)
            
            with self.cache_lock:
                recent_events = [e for e in self.recent_events if e.timestamp >= hour_ago]
            
            event_counts = defaultdict(int)
            error_counts = defaultdict(int)
            
            for event in recent_events:
                event_counts[event.event_type.value] += 1
                if event.log_level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                    error_counts[event.event_type.value] += 1
            
            # Calculate rates
            opportunities_per_hour = event_counts.get('opportunity_detected', 0)
            execution_rate = (event_counts.get('opportunity_executed', 0) / 
                             max(opportunities_per_hour, 1)) * 100
            
            with self.cache_lock:
                total_events = len(self.recent_events)
                performance_metrics_copy = dict(self.performance_metrics)
            
            return {
                "session_id": self.session_id,
                "session_duration_hours": session_duration.total_seconds() / 3600,
                "total_events": total_events,
                "events_last_hour": len(recent_events),
                "event_breakdown": dict(event_counts),
                "error_breakdown": dict(error_counts),
                "opportunities_per_hour": opportunities_per_hour,
                "execution_rate_percent": execution_rate,
                "capture_rate_percent": self.calculate_opportunity_capture_rate(),
                "performance_metrics": performance_metrics_copy,
                "database_health": self._check_database_health()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating health report: {e}")
            return {"error": str(e)}

    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health and performance"""
        conn = None
        try:
            conn = self._get_connection()
            
            # Check table sizes
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trading_events")
            events_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM opportunities")
            opportunities_count = cursor.fetchone()[0]
            
            # Check database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            db_size_mb = (page_count * page_size) / (1024 * 1024)
            
            return {
                "events_count": events_count,
                "opportunities_count": opportunities_count,
                "database_size_mb": round(db_size_mb, 2),
                "connection_pool_size": len(self._connection_pool),
                "status": "healthy"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            if conn:
                self._return_connection(conn)

    def export_session_data(self, format: str = "json") -> str:
        """Export session data for analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                filename = self.log_directory / f"session_export_{timestamp}.json"
                
                with self.cache_lock:
                    recent_events_copy = [event.to_dict() for event in list(self.recent_events)]
                
                export_data = {
                    "session_info": {
                        "session_id": self.session_id,
                        "start_time": self.session_start.isoformat(),
                        "export_time": datetime.now().isoformat()
                    },
                    "health_report": self.get_system_health_report(),
                    "missed_opportunities": self.get_missed_opportunity_analysis(),
                    "recent_events": recent_events_copy
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif format.lower() == "csv":
                filename = self.log_directory / f"session_export_{timestamp}.csv"
                
                with self.cache_lock:
                    events_df = pd.DataFrame([event.to_dict() for event in self.recent_events])
                
                if not events_df.empty:
                    events_df.to_csv(filename, index=False)
                else:
                    # Create empty CSV with headers
                    empty_df = pd.DataFrame(columns=['timestamp', 'event_type', 'symbol', 'log_level', 'message'])
                    empty_df.to_csv(filename, index=False)
            
            return str(filename)
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting session data: {e}")
            return ""

    def query_events(self, symbol: str = None, event_type: str = None, 
                    hours: int = 24) -> List[Dict[str, Any]]:
        """Query events from database with thread safety"""
        conn = None
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = "SELECT * FROM trading_events WHERE timestamp >= ?"
            params = [cutoff_time.isoformat()]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
                
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            query += " ORDER BY timestamp DESC LIMIT 1000"  # Limit results
            
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error querying events: {e}")
            return []
        finally:
            if conn:
                self._return_connection(conn)

    def cleanup_old_logs(self, days: int = 7):
        """Clean up logs older than specified days"""
        conn = None
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            conn = self._get_connection()
            
            # Delete old events
            cursor = conn.execute("""
                DELETE FROM trading_events 
                WHERE timestamp < ?
            """, (cutoff_time.isoformat(),))
            events_deleted = cursor.rowcount
            
            cursor = conn.execute("""
                DELETE FROM opportunities 
                WHERE timestamp < ?
            """, (cutoff_time.isoformat(),))
            opportunities_deleted = cursor.rowcount
            
            cursor = conn.execute("""
                DELETE FROM performance_metrics 
                WHERE timestamp < ?
            """, (cutoff_time.isoformat(),))
            metrics_deleted = cursor.rowcount
            
            conn.commit()
            
            # Vacuum database to reclaim space
            conn.execute("VACUUM")
            
            self.logger.info(f"🧹 Cleaned up logs older than {days} days: "
                           f"{events_deleted} events, {opportunities_deleted} opportunities, "
                           f"{metrics_deleted} metrics deleted")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up logs: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._return_connection(conn)

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get daily trading summary"""
        try:
            today = datetime.now().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            
            with self.cache_lock:
                day_events = [e for e in self.recent_events if e.timestamp >= start_of_day]
            
            opportunities_detected = len([e for e in day_events if e.event_type == EventType.OPPORTUNITY_DETECTED])
            opportunities_executed = len([e for e in day_events if e.event_type == EventType.OPPORTUNITY_EXECUTED])
            opportunities_missed = len([e for e in day_events if e.event_type == EventType.OPPORTUNITY_MISSED])
            
            trades_entered = len([e for e in day_events if e.event_type == EventType.TRADE_ENTRY])
            trades_exited = len([e for e in day_events if e.event_type == EventType.TRADE_EXIT])
            
            system_errors = len([e for e in day_events if e.event_type == EventType.SYSTEM_ERROR])
            
            return {
                "date": today.isoformat(),
                "opportunities_detected": opportunities_detected,
                "opportunities_executed": opportunities_executed,
                "opportunities_missed": opportunities_missed,
                "capture_rate": (opportunities_executed / max(opportunities_detected, 1)) * 100,
                "trades_entered": trades_entered,
                "trades_exited": trades_exited,
                "system_errors": system_errors,
                "total_events": len(day_events)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating daily summary: {e}")
            return {}

    def _cleanup(self):
        """Cleanup resources on exit"""
        try:
            # Close all database connections
            with self.db_lock:
                for conn in self._connection_pool:
                    try:
                        conn.close()
                    except:
                        pass
                self._connection_pool.clear()
                
            self.logger.info("🧹 Cleanup completed")
        except:
            pass

if __name__ == "__main__":
    # Test the comprehensive logger
    logger = ComprehensiveLogger()
    
    print("🧪 Testing Comprehensive Logger...")
    
    # Test opportunity logging
    logger.log_opportunity_detected(
        "AAPL", "momentum_breakout", 85.5, "bullish",
        {"price": 150.25, "volume": 500000}
    )
    
    logger.log_opportunity_executed(
        "AAPL", "momentum_breakout", 85.5, 150.30, 100,
        {"order_id": "12345"}
    )
    
    logger.log_opportunity_missed(
        "GOOGL", "volume_spike", "insufficient_capital", 78.2,
        {"required_capital": 50000, "available": 30000}
    )
    
    logger.log_trade_entry(
        "AAPL", "BUY", 100, 150.30, "momentum_strategy",
        {"stop_loss": 148.00, "target": 155.00}
    )
    
    logger.log_system_error(
        "data_manager", "API rate limit exceeded",
        {"retry_in": 60, "requests_made": 100}
    )
    
    # Generate reports
    health = logger.get_system_health_report()
    print(f"📊 System Health: {health.get('execution_rate_percent', 0):.1f}% execution rate")
    
    missed_analysis = logger.get_missed_opportunity_analysis()
    print(f"📈 Capture Rate: {missed_analysis.get('capture_rate', 0):.1f}%")
    
    daily_summary = logger.get_daily_summary()
    print(f"📅 Daily Summary: {daily_summary.get('opportunities_detected', 0)} opportunities detected")
    
    # Export data
    export_file = logger.export_session_data("json")
    print(f"📁 Session data exported to: {export_file}")
    
    print("✅ Comprehensive Logger test completed!")

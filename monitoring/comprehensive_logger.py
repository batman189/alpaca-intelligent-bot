"""
Comprehensive Logging System - Track Everything, Miss Nothing
Advanced logging for opportunities, trades, misses, and system health
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
        """Initialize comprehensive logging system"""
        
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
        self._init_database()
        
        # In-memory storage for real-time analytics
        self.recent_events = deque(maxlen=1000)  # Last 1000 events
        self.opportunity_cache = defaultdict(list)
        self.performance_metrics = defaultdict(float)
        
        # Session tracking
        self.session_id = f"session_{int(datetime.now().timestamp())}"
        self.session_start = datetime.now()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Configure Python logging
        self._setup_python_logging()
        
        # Performance tracking
        self.capture_rate_window = 50  # Track last 50 opportunities
        self.missed_opportunities = []
        self.executed_opportunities = []
        
        self.logger.info(f"🔍 Comprehensive Logger initialized - Session: {self.session_id}")

    def _init_database(self):
        """Initialize SQLite database for event storage"""
        with sqlite3.connect(self.db_path) as conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_opportunities_symbol ON opportunities(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_opportunities_executed ON opportunities(executed)")

    def _setup_python_logging(self):
        """Setup Python logging configuration"""
        # Create logger
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
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

    def log_opportunity_detected(self, symbol: str, signal_type: str, confidence: float,
                               direction: str, metadata: Dict[str, Any]):
        """Log detected opportunity"""
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO opportunities 
                (timestamp, symbol, signal_type, confidence, direction, metadata, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp.isoformat(),
                symbol,
                signal_type,
                confidence,
                direction,
                json.dumps(metadata),
                self.session_id
            ))
        
        # Update opportunity cache
        self.opportunity_cache[symbol].append(event)

    def log_opportunity_executed(self, symbol: str, signal_type: str, confidence: float,
                               entry_price: float, position_size: float, metadata: Dict[str, Any]):
        """Log executed opportunity"""
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
        self.executed_opportunities.append(event)
        
        # Update opportunities table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE opportunities 
                SET executed = 1 
                WHERE symbol = ? AND signal_type = ? AND executed = 0
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol, signal_type))

    def log_opportunity_missed(self, symbol: str, signal_type: str, reason: str,
                             confidence: float, metadata: Dict[str, Any]):
        """Log missed opportunity with reason"""
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
        self.missed_opportunities.append(event)
        
        # Update opportunities table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE opportunities 
                SET reason_skipped = ? 
                WHERE symbol = ? AND signal_type = ? AND executed = 0
                ORDER BY timestamp DESC LIMIT 1
            """, (reason, symbol, signal_type))

    def log_trade_entry(self, symbol: str, side: str, quantity: float, price: float,
                       strategy: str, metadata: Dict[str, Any]):
        """Log trade entry"""
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

    def log_trade_exit(self, symbol: str, side: str, quantity: float, price: float,
                      pnl: float, reason: str, metadata: Dict[str, Any]):
        """Log trade exit"""
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
        
        # Update performance metrics
        self.performance_metrics['total_pnl'] += pnl
        self.performance_metrics['total_trades'] += 1

    def log_system_error(self, component: str, error_message: str, 
                        metadata: Dict[str, Any] = None):
        """Log system error"""
        event = TradingEvent(
            timestamp=datetime.now(),
            event_type=EventType.SYSTEM_ERROR,
            symbol="SYSTEM",
            log_level=LogLevel.ERROR,
            message=f"System error in {component}: {error_message}",
            metadata={
                'component': component,
                'error': error_message,
                **(metadata or {})
            },
            session_id=self.session_id
        )
        
        self._log_event(event)

    def log_market_condition_change(self, condition: str, confidence: float,
                                  metadata: Dict[str, Any]):
        """Log market condition change"""
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

    def _log_event(self, event: TradingEvent):
        """Internal method to log event to all storage systems"""
        with self.lock:
            # Add to recent events
            self.recent_events.append(event)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
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

    def calculate_opportunity_capture_rate(self) -> float:
        """Calculate the rate of opportunities we actually execute"""
        if not self.opportunity_cache:
            return 0.0
        
        total_opportunities = sum(len(opportunities) for opportunities in self.opportunity_cache.values())
        total_executed = len(self.executed_opportunities)
        
        if total_opportunities == 0:
            return 0.0
            
        return (total_executed / total_opportunities) * 100

    def get_missed_opportunity_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze missed opportunities in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
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

    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        current_time = datetime.now()
        session_duration = current_time - self.session_start
        
        # Count events by type in last hour
        hour_ago = current_time - timedelta(hours=1)
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
        
        return {
            "session_id": self.session_id,
            "session_duration_hours": session_duration.total_seconds() / 3600,
            "total_events": len(self.recent_events),
            "events_last_hour": len(recent_events),
            "event_breakdown": dict(event_counts),
            "error_breakdown": dict(error_counts),
            "opportunities_per_hour": opportunities_per_hour,
            "execution_rate_percent": execution_rate,
            "capture_rate_percent": self.calculate_opportunity_capture_rate(),
            "performance_metrics": dict(self.performance_metrics)
        }

    def export_session_data(self, format: str = "json") -> str:
        """Export session data for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = self.log_directory / f"session_export_{timestamp}.json"
            
            export_data = {
                "session_info": {
                    "session_id": self.session_id,
                    "start_time": self.session_start.isoformat(),
                    "export_time": datetime.now().isoformat()
                },
                "health_report": self.get_system_health_report(),
                "missed_opportunities": self.get_missed_opportunity_analysis(),
                "recent_events": [event.to_dict() for event in list(self.recent_events)]
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format.lower() == "csv":
            filename = self.log_directory / f"session_export_{timestamp}.csv"
            
            events_df = pd.DataFrame([event.to_dict() for event in self.recent_events])
            events_df.to_csv(filename, index=False)
        
        return str(filename)

    def query_events(self, symbol: str = None, event_type: str = None, 
                    hours: int = 24) -> List[Dict[str, Any]]:
        """Query events from database"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = "SELECT * FROM trading_events WHERE timestamp >= ?"
        params = [cutoff_time.isoformat()]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
        
        return results

    def cleanup_old_logs(self, days: int = 7):
        """Clean up logs older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Archive old events to backup table first (optional)
            conn.execute("""
                DELETE FROM trading_events 
                WHERE timestamp < ?
            """, (cutoff_time.isoformat(),))
            
            conn.execute("""
                DELETE FROM opportunities 
                WHERE timestamp < ?
            """, (cutoff_time.isoformat(),))
            
            conn.execute("""
                DELETE FROM performance_metrics 
                WHERE timestamp < ?
            """, (cutoff_time.isoformat(),))
        
        self.logger.info(f"🧹 Cleaned up logs older than {days} days")

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get daily trading summary"""
        today = datetime.now().date()
        start_of_day = datetime.combine(today, datetime.min.time())
        
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
    print(f"📊 System Health: {health['execution_rate_percent']:.1f}% execution rate")
    
    missed_analysis = logger.get_missed_opportunity_analysis()
    print(f"📈 Capture Rate: {missed_analysis['capture_rate']:.1f}%")
    
    daily_summary = logger.get_daily_summary()
    print(f"📅 Daily Summary: {daily_summary['opportunities_detected']} opportunities detected")
    
    # Export data
    export_file = logger.export_session_data("json")
    print(f"📁 Session data exported to: {export_file}")
    
    print("✅ Comprehensive Logger test completed!")
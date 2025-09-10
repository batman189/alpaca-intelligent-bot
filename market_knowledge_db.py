"""
MARKET KNOWLEDGE DATABASE
Advanced historical pattern storage and retrieval system

This module manages the bot's memory of market patterns, outcomes, and learned behaviors.
It stores pattern recognition results, trading outcomes, and builds knowledge over time.
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
import logging
from dataclasses import dataclass

@dataclass
class PatternRecord:
    """Record of a market pattern and its outcome"""
    timestamp: datetime
    symbol: str
    pattern_data: Dict
    outcome: float  # Future return
    confidence: float
    pattern_type: str
    market_conditions: Dict

@dataclass
class TradingOutcome:
    """Record of a trading decision and its result"""
    timestamp: datetime
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    profit_loss: Optional[float]
    pattern_match_confidence: float
    decision_reasoning: List[str]

class MarketKnowledgeDatabase:
    """Advanced database for storing and retrieving market knowledge"""
    
    def __init__(self, db_path: str = "market_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Pattern records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                outcome REAL NOT NULL,
                confidence REAL NOT NULL,
                pattern_type TEXT NOT NULL,
                market_conditions TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trading outcomes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                profit_loss REAL,
                pattern_match_confidence REAL NOT NULL,
                decision_reasoning TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Market conditions table for context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                vix_level REAL,
                market_trend TEXT,
                sector_performance TEXT,
                volume_profile TEXT,
                volatility_regime TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                total_pnl REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL,
                pattern_accuracy REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_symbol_time ON pattern_records(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_symbol_time ON trading_outcomes(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_records(pattern_type)')
        
        conn.commit()
        conn.close()
    
    def store_pattern_record(self, record: PatternRecord) -> None:
        """Store a pattern recognition record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO pattern_records 
                (timestamp, symbol, pattern_data, outcome, confidence, pattern_type, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp.isoformat(),
                record.symbol,
                json.dumps(record.pattern_data),
                record.outcome,
                record.confidence,
                record.pattern_type,
                json.dumps(record.market_conditions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing pattern record: {e}")
    
    def store_trading_outcome(self, outcome: TradingOutcome) -> None:
        """Store a trading outcome record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_outcomes 
                (timestamp, symbol, action, entry_price, exit_price, profit_loss, 
                 pattern_match_confidence, decision_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                outcome.timestamp.isoformat(),
                outcome.symbol,
                outcome.action,
                outcome.entry_price,
                outcome.exit_price,
                outcome.profit_loss,
                outcome.pattern_match_confidence,
                json.dumps(outcome.decision_reasoning)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing trading outcome: {e}")
    
    def get_similar_patterns(self, symbol: str, pattern_data: Dict, 
                           pattern_type: str, lookback_days: int = 365) -> List[PatternRecord]:
        """Find similar historical patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, symbol, pattern_data, outcome, confidence, pattern_type, market_conditions
                FROM pattern_records
                WHERE symbol = ? AND pattern_type = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (symbol, pattern_type, cutoff_date))
            
            records = []
            for row in cursor.fetchall():
                record = PatternRecord(
                    timestamp=datetime.fromisoformat(row[0]),
                    symbol=row[1],
                    pattern_data=json.loads(row[2]),
                    outcome=row[3],
                    confidence=row[4],
                    pattern_type=row[5],
                    market_conditions=json.loads(row[6])
                )
                records.append(record)
            
            conn.close()
            
            # Filter by similarity
            similar_records = []
            for record in records:
                similarity = self._calculate_pattern_similarity(pattern_data, record.pattern_data)
                if similarity > 0.7:  # High similarity threshold
                    similar_records.append((record, similarity))
            
            # Sort by similarity and return top matches
            similar_records.sort(key=lambda x: x[1], reverse=True)
            return [record for record, similarity in similar_records[:20]]
            
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two patterns"""
        try:
            # Extract numerical features from both patterns
            features1 = self._extract_numerical_features(pattern1)
            features2 = self._extract_numerical_features(pattern2)
            
            if not features1 or not features2:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = sum(f1 * f2 for f1, f2 in zip(features1, features2))
            magnitude1 = sum(f1 * f1 for f1 in features1) ** 0.5
            magnitude2 = sum(f2 * f2 for f2 in features2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def _extract_numerical_features(self, pattern: Dict) -> List[float]:
        """Extract numerical features from pattern dictionary"""
        features = []
        
        def extract_recursive(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    extract_recursive(value, f"{prefix}_{key}" if prefix else key)
            elif isinstance(data, (int, float)):
                features.append(float(data))
            elif isinstance(data, bool):
                features.append(1.0 if data else 0.0)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    extract_recursive(item, f"{prefix}_{i}")
        
        extract_recursive(pattern)
        return features
    
    def get_pattern_success_rate(self, pattern_type: str, symbol: str = None, 
                               lookback_days: int = 90) -> Dict:
        """Get success rate for a specific pattern type"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
            
            if symbol:
                cursor.execute('''
                    SELECT outcome FROM pattern_records
                    WHERE pattern_type = ? AND symbol = ? AND timestamp >= ?
                ''', (pattern_type, symbol, cutoff_date))
            else:
                cursor.execute('''
                    SELECT outcome FROM pattern_records
                    WHERE pattern_type = ? AND timestamp >= ?
                ''', (pattern_type, cutoff_date))
            
            outcomes = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not outcomes:
                return {'success_rate': 0.0, 'avg_outcome': 0.0, 'sample_size': 0}
            
            positive_outcomes = [o for o in outcomes if o > 0]
            success_rate = len(positive_outcomes) / len(outcomes)
            avg_outcome = sum(outcomes) / len(outcomes)
            
            return {
                'success_rate': success_rate,
                'avg_outcome': avg_outcome,
                'sample_size': len(outcomes),
                'positive_outcomes': len(positive_outcomes)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pattern success rate: {e}")
            return {'success_rate': 0.0, 'avg_outcome': 0.0, 'sample_size': 0}
    
    def get_trading_performance(self, days: int = 30) -> Dict:
        """Get recent trading performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT action, profit_loss, pattern_match_confidence
                FROM trading_outcomes
                WHERE timestamp >= ? AND profit_loss IS NOT NULL
            ''', (cutoff_date,))
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0}
            
            winning_trades = [t for t in trades if t[1] > 0]
            total_pnl = sum(t[1] for t in trades)
            win_rate = len(winning_trades) / len(trades)
            avg_confidence = sum(t[2] for t in trades) / len(trades)
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': sum(t[1] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                'avg_loss': sum(t[1] for t in trades if t[1] < 0) / len([t for t in trades if t[1] < 0]) if any(t[1] < 0 for t in trades) else 0,
                'avg_confidence': avg_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading performance: {e}")
            return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0}
    
    def get_best_performing_patterns(self, min_sample_size: int = 10) -> List[Dict]:
        """Get the best performing pattern types"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pattern_type, AVG(outcome) as avg_outcome, COUNT(*) as count,
                       AVG(confidence) as avg_confidence
                FROM pattern_records
                WHERE timestamp >= date('now', '-90 days')
                GROUP BY pattern_type
                HAVING count >= ?
                ORDER BY avg_outcome DESC
            ''', (min_sample_size,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'pattern_type': row[0],
                    'avg_outcome': row[1],
                    'sample_size': row[2],
                    'avg_confidence': row[3]
                })
            
            conn.close()
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting best patterns: {e}")
            return []
    
    def learn_from_outcome(self, symbol: str, pattern_data: Dict, actual_outcome: float,
                          predicted_outcome: float, confidence: float) -> None:
        """Learn from actual vs predicted outcomes"""
        try:
            # Store the learning data
            record = PatternRecord(
                timestamp=datetime.now(),
                symbol=symbol,
                pattern_data=pattern_data,
                outcome=actual_outcome,
                confidence=confidence,
                pattern_type=self._classify_pattern_type(pattern_data),
                market_conditions=self._get_current_market_context()
            )
            
            self.store_pattern_record(record)
            
            # Update pattern accuracy metrics
            self._update_pattern_accuracy(pattern_data, actual_outcome, predicted_outcome)
            
        except Exception as e:
            self.logger.error(f"Error learning from outcome: {e}")
    
    def _classify_pattern_type(self, pattern_data: Dict) -> str:
        """Classify the type of pattern"""
        # Simplified classification logic
        if pattern_data.get('breakout_patterns', {}).get('breakout_type'):
            return 'breakout'
        elif pattern_data.get('reversal_signals', {}).get('rsi_oversold') or pattern_data.get('reversal_signals', {}).get('rsi_overbought'):
            return 'reversal'
        elif pattern_data.get('volatility_expansion', {}).get('volatility_expansion'):
            return 'volatility_expansion'
        elif pattern_data.get('trend_channels', {}).get('trend_direction'):
            return 'trend_following'
        else:
            return 'mixed'
    
    def _get_current_market_context(self) -> Dict:
        """Get current market conditions for context"""
        return {
            'timestamp': datetime.now().isoformat(),
            'market_hours': self._is_market_hours(),
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month
        }
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return (now.weekday() < 5 and  # Monday to Friday
                market_open <= now <= market_close)
    
    def _update_pattern_accuracy(self, pattern_data: Dict, actual: float, predicted: float) -> None:
        """Update accuracy metrics for pattern predictions"""
        try:
            # Calculate prediction error
            error = abs(actual - predicted)
            accuracy = max(0, 1 - (error / max(abs(actual), abs(predicted), 0.01)))
            
            # Store accuracy data for future reference
            # This could be expanded to maintain running averages per pattern type
            
        except Exception as e:
            self.logger.error(f"Error updating pattern accuracy: {e}")
    
    def get_market_regime_analysis(self, lookback_days: int = 60) -> Dict:
        """Analyze current market regime based on historical patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
            
            cursor.execute('''
                SELECT pattern_type, AVG(outcome) as avg_outcome, COUNT(*) as frequency
                FROM pattern_records
                WHERE timestamp >= ?
                GROUP BY pattern_type
                ORDER BY frequency DESC
            ''', (cutoff_date,))
            
            patterns = cursor.fetchall()
            conn.close()
            
            if not patterns:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            # Analyze dominant patterns
            total_patterns = sum(p[2] for p in patterns)
            dominant_pattern = patterns[0]
            
            regime_analysis = {
                'dominant_pattern': dominant_pattern[0],
                'pattern_frequency': dominant_pattern[2] / total_patterns,
                'avg_outcome': dominant_pattern[1],
                'regime_confidence': min(dominant_pattern[2] / 50, 1.0),  # Confidence based on sample size
                'all_patterns': [{'type': p[0], 'outcome': p[1], 'frequency': p[2]} for p in patterns]
            }
            
            # Classify market regime
            if dominant_pattern[1] > 0.02:
                regime_analysis['regime'] = 'bullish'
            elif dominant_pattern[1] < -0.02:
                regime_analysis['regime'] = 'bearish'
            else:
                regime_analysis['regime'] = 'neutral'
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}
    
    def cleanup_old_records(self, days_to_keep: int = 365) -> None:
        """Clean up old records to manage database size"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            cursor.execute('DELETE FROM pattern_records WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM trading_outcomes WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM market_conditions WHERE timestamp < ?', (cutoff_date,))
            
            # Keep performance metrics for longer
            perf_cutoff = (datetime.now() - timedelta(days=days_to_keep * 2)).isoformat()
            cursor.execute('DELETE FROM performance_metrics WHERE created_at < ?', (perf_cutoff,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up records older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old records: {e}")
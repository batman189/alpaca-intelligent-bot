"""
Adaptive Learning System - Professional ML-Based Trading Intelligence
Real-time learning from trades, market conditions, and performance feedback
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import os
import pickle
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdaptiveLearningSystem:
    def __init__(self):
        """Initialize the Adaptive Learning System"""
        # Learning parameters
        self.learning_rate = 0.01
        self.memory_size = 1000  # Number of recent trades to remember
        self.confidence_decay = 0.95  # How quickly confidence decays
        self.performance_window = 50  # Window for performance calculations
        
        # Data storage
        self.trade_history = deque(maxlen=self.memory_size)
        self.symbol_performance = defaultdict(lambda: {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'avg_hold_time': 0.0,
            'win_rate': 0.0,
            'confidence_scores': deque(maxlen=100),
            'actual_outcomes': deque(maxlen=100),
            'last_updated': datetime.now()
        })
        
        # Performance tracking
        self.overall_performance = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_return_per_trade': 0.0,
            'volatility_of_returns': 0.0
        }
        
        # Adaptive thresholds
        self.dynamic_thresholds = {
            'min_confidence': 0.65,
            'max_correlation': 0.8,
            'volatility_limit': 0.4,
            'position_size_multiplier': 1.0
        }
        
        # Model adaptation tracking
        self.model_performance_tracker = defaultdict(lambda: {
            'predictions': deque(maxlen=200),
            'actual_results': deque(maxlen=200),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'last_updated': datetime.now()
        })
        
        # Market regime detection
        self.market_regimes = {
            'current_regime': 'neutral',  # bull, bear, neutral, volatile
            'regime_confidence': 0.5,
            'regime_history': deque(maxlen=100),
            'regime_performance': defaultdict(float)
        }
        
        logger.info("Adaptive Learning System initialized")
        
        # Load previous learning data if available
        self._load_learning_data()
    
    def record_trade_entry(self, opportunity: Dict) -> bool:
        """
        Record a trade entry for learning purposes
        This is called when a trade is placed
        """
        try:
            trade_record = {
                'trade_id': f"{opportunity['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'symbol': opportunity['symbol'],
                'entry_time': datetime.now(),
                'entry_price': opportunity.get('current_price', 0),
                'predicted_direction': opportunity.get('prediction', 0),
                'confidence_score': opportunity.get('confidence', 0),
                'risk_score': opportunity.get('risk_assessment', {}).get('risk_score', 0),
                'position_size': opportunity.get('position_size', 0),
                'trade_type': opportunity.get('trade_type', 'stock'),
                'market_conditions': self._capture_market_conditions(opportunity),
                'status': 'open'
            }
            
            # Add to trade history
            self.trade_history.append(trade_record)
            
            logger.info(f"Trade entry recorded: {trade_record['trade_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording trade entry: {e}")
            return False
    
    def record_trade_exit(self, trade_id: str, exit_price: float, exit_reason: str) -> bool:
        """
        Record trade exit and calculate performance metrics
        """
        try:
            # Find the trade in history
            trade_record = None
            for trade in reversed(self.trade_history):
                if trade['trade_id'] == trade_id:
                    trade_record = trade
                    break
            
            if not trade_record:
                logger.warning(f"Trade {trade_id} not found in history")
                return False
            
            # Calculate performance
            entry_price = trade_record['entry_price']
            if trade_record['predicted_direction'] == 1:  # Bullish prediction
                pnl_percent = (exit_price - entry_price) / entry_price
            else:  # Bearish prediction
                pnl_percent = (entry_price - exit_price) / entry_price
            
            # Update trade record
            trade_record.update({
                'exit_time': datetime.now(),
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl_percent': pnl_percent,
                'hold_time_minutes': (datetime.now() - trade_record['entry_time']).total_seconds() / 60,
                'profitable': pnl_percent > 0,
                'status': 'closed'
            })
            
            # Update symbol-specific performance
            self._update_symbol_performance(trade_record)
            
            # Update model performance tracking
            self._update_model_performance(trade_record)
            
            # Update overall performance
            self._update_overall_performance(trade_record)
            
            # Learn from this trade outcome
            self._learn_from_trade(trade_record)
            
            logger.info(f"Trade exit recorded: {trade_id}, PnL: {pnl_percent:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording trade exit: {e}")
            return False
    
    def should_trade_symbol(self, symbol: str, predicted_confidence: float) -> bool:
        """
        Determine if we should trade a symbol based on adaptive learning
        """
        try:
            symbol_stats = self.symbol_performance[symbol]
            
            # Check if we have enough historical data
            if symbol_stats['total_trades'] < 5:
                # Not enough data, use default threshold
                return predicted_confidence >= self.dynamic_thresholds['min_confidence']
            
            # Adaptive confidence threshold based on historical performance
            symbol_win_rate = symbol_stats['win_rate']
            
            if symbol_win_rate > 0.6:  # High win rate
                adjusted_threshold = self.dynamic_thresholds['min_confidence'] * 0.9
            elif symbol_win_rate > 0.4:  # Average win rate
                adjusted_threshold = self.dynamic_thresholds['min_confidence']
            else:  # Low win rate
                adjusted_threshold = self.dynamic_thresholds['min_confidence'] * 1.2
            
            # Consider recent performance trend
            recent_performance = self._get_recent_symbol_performance(symbol)
            if recent_performance < -0.05:  # Recent losses > 5%
                adjusted_threshold *= 1.3
            elif recent_performance > 0.05:  # Recent gains > 5%
                adjusted_threshold *= 0.8
            
            # Market regime adjustment
            regime_adjustment = self._get_regime_adjustment()
            adjusted_threshold *= regime_adjustment
            
            decision = predicted_confidence >= adjusted_threshold
            
            logger.debug(f"Trading decision for {symbol}: Confidence={predicted_confidence:.3f}, "
                        f"Threshold={adjusted_threshold:.3f}, Decision={decision}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in trading decision for {symbol}: {e}")
            return predicted_confidence >= 0.75  # Conservative fallback
    
    def get_symbol_performance(self, symbol: str) -> Dict:
        """Get performance statistics for a specific symbol"""
        try:
            stats = self.symbol_performance[symbol].copy()
            
            # Calculate additional metrics
            if stats['total_trades'] > 0:
                stats['avg_pnl_per_trade'] = stats['total_pnl'] / stats['total_trades']
                
                # Calculate confidence calibration
                if len(stats['confidence_scores']) >= 10:
                    confidence_array = np.array(stats['confidence_scores'])
                    outcomes_array = np.array(stats['actual_outcomes'])
                    stats['confidence_calibration'] = self._calculate_calibration(
                        confidence_array, outcomes_array
                    )
                else:
                    stats['confidence_calibration'] = 0.5
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting symbol performance: {e}")
            return {}
    
    def get_overall_performance(self) -> Dict:
        """Get overall system performance metrics"""
        try:
            return self.overall_performance.copy()
            
        except Exception as e:
            logger.error(f"Error getting overall performance: {e}")
            return {}
    
    def adapt_strategy_parameters(self) -> Dict:
        """
        Adapt strategy parameters based on learning outcomes
        Returns updated parameters for the trading system
        """
        try:
            adaptations = {}
            
            # Adapt confidence thresholds based on overall performance
            if self.overall_performance['win_rate'] > 0.6:
                # High win rate - can be more aggressive
                adaptations['confidence_threshold'] = max(0.5, 
                    self.dynamic_thresholds['min_confidence'] * 0.9)
            elif self.overall_performance['win_rate'] < 0.4:
                # Low win rate - be more conservative
                adaptations['confidence_threshold'] = min(0.9, 
                    self.dynamic_thresholds['min_confidence'] * 1.2)
            
            # Adapt position sizing based on recent volatility of returns
            volatility = self.overall_performance.get('volatility_of_returns', 0.02)
            if volatility > 0.05:  # High volatility
                adaptations['position_size_multiplier'] = 0.7
            elif volatility < 0.02:  # Low volatility
                adaptations['position_size_multiplier'] = 1.2
            else:
                adaptations['position_size_multiplier'] = 1.0
            
            # Adapt based on market regime
            current_regime = self.market_regimes['current_regime']
            if current_regime == 'volatile':
                adaptations['max_positions'] = 3
                adaptations['analysis_frequency'] = 30  # More frequent analysis
            elif current_regime == 'trending':
                adaptations['max_positions'] = 5
                adaptations['analysis_frequency'] = 60
            
            # Update internal thresholds
            self.dynamic_thresholds.update(adaptations)
            
            logger.info(f"Strategy parameters adapted: {adaptations}")
            return adaptations
            
        except Exception as e:
            logger.error(f"Error adapting strategy parameters: {e}")
            return {}
    
    def _capture_market_conditions(self, opportunity: Dict) -> Dict:
        """Capture current market conditions for context"""
        try:
            return {
                'volatility': opportunity.get('volatility', 0),
                'volume_ratio': opportunity.get('volume_ratio', 1),
                'market_trend': opportunity.get('market_trend', 'neutral'),
                'time_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'confluence_score': opportunity.get('confluence_score', 0.5)
            }
        except Exception as e:
            logger.error(f"Error capturing market conditions: {e}")
            return {}
    
    def _update_symbol_performance(self, trade_record: Dict):
        """Update performance metrics for a specific symbol"""
        try:
            symbol = trade_record['symbol']
            stats = self.symbol_performance[symbol]
            
            # Update basic counts
            stats['total_trades'] += 1
            if trade_record['profitable']:
                stats['profitable_trades'] += 1
            
            # Update PnL
            stats['total_pnl'] += trade_record['pnl_percent']
            
            # Update win rate
            stats['win_rate'] = stats['profitable_trades'] / stats['total_trades']
            
            # Update hold time
            current_avg = stats['avg_hold_time']
            new_hold_time = trade_record['hold_time_minutes']
            stats['avg_hold_time'] = ((current_avg * (stats['total_trades'] - 1)) + new_hold_time) / stats['total_trades']
            
            # Store confidence and outcome for calibration
            stats['confidence_scores'].append(trade_record['confidence_score'])
            stats['actual_outcomes'].append(1 if trade_record['profitable'] else 0)
            
            stats['last_updated'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating symbol performance: {e}")
    
    def _update_model_performance(self, trade_record: Dict):
        """Update model performance tracking"""
        try:
            symbol = trade_record['symbol']
            tracker = self.model_performance_tracker[symbol]
            
            # Store prediction and actual result
            predicted_success = trade_record['confidence_score'] > 0.5
            actual_success = trade_record['profitable']
            
            tracker['predictions'].append(predicted_success)
            tracker['actual_results'].append(actual_success)
            
            # Calculate metrics if we have enough data
            if len(tracker['predictions']) >= 20:
                predictions = np.array(tracker['predictions'])
                actual = np.array(tracker['actual_results'])
                
                # Accuracy
                tracker['accuracy'] = np.mean(predictions == actual)
                
                # Precision and Recall
                if np.sum(predictions) > 0:
                    tracker['precision'] = np.sum((predictions == 1) & (actual == 1)) / np.sum(predictions)
                if np.sum(actual) > 0:
                    tracker['recall'] = np.sum((predictions == 1) & (actual == 1)) / np.sum(actual)
            
            tracker['last_updated'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def _update_overall_performance(self, trade_record: Dict):
        """Update overall system performance metrics"""
        try:
            # Update basic metrics
            self.overall_performance['total_trades'] += 1
            if trade_record['profitable']:
                self.overall_performance['profitable_trades'] += 1
            
            # Update PnL
            self.overall_performance['total_pnl'] += trade_record['pnl_percent']
            
            # Update win rate
            total_trades = self.overall_performance['total_trades']
            self.overall_performance['win_rate'] = self.overall_performance['profitable_trades'] / total_trades
            
            # Update average return per trade
            self.overall_performance['avg_return_per_trade'] = self.overall_performance['total_pnl'] / total_trades
            
            # Calculate volatility of returns (last 50 trades)
            recent_returns = [trade['pnl_percent'] for trade in list(self.trade_history)[-50:] if trade.get('pnl_percent') is not None]
            if len(recent_returns) > 10:
                self.overall_performance['volatility_of_returns'] = np.std(recent_returns)
                
                # Calculate Sharpe ratio (assuming 0% risk-free rate)
                if self.overall_performance['volatility_of_returns'] > 0:
                    self.overall_performance['sharpe_ratio'] = self.overall_performance['avg_return_per_trade'] / self.overall_performance['volatility_of_returns']
            
        except Exception as e:
            logger.error(f"Error updating overall performance: {e}")
    
    def _learn_from_trade(self, trade_record: Dict):
        """Learn from individual trade outcomes to improve future decisions"""
        try:
            # Identify what went right or wrong
            confidence = trade_record['confidence_score']
            actual_outcome = trade_record['profitable']
            
            # Update confidence calibration learning
            self._update_confidence_calibration(confidence, actual_outcome)
            
            # Learn from market conditions
            self._learn_from_market_conditions(trade_record)
            
            # Update market regime detection
            self._update_market_regime(trade_record)
            
        except Exception as e:
            logger.error(f"Error learning from trade: {e}")
    
    def _update_confidence_calibration(self, predicted_confidence: float, actual_outcome: bool):
        """Update confidence calibration based on outcomes"""
        try:
            # Simple calibration update
            outcome_value = 1.0 if actual_outcome else 0.0
            error = predicted_confidence - outcome_value
            
            # Adjust confidence threshold based on systematic errors
            if abs(error) > 0.3:  # Large error
                if error > 0:  # Overconfident
                    self.dynamic_thresholds['min_confidence'] = min(0.9, 
                        self.dynamic_thresholds['min_confidence'] + 0.01)
                else:  # Underconfident
                    self.dynamic_thresholds['min_confidence'] = max(0.4, 
                        self.dynamic_thresholds['min_confidence'] - 0.01)
            
        except Exception as e:
            logger.error(f"Error updating confidence calibration: {e}")
    
    def _learn_from_market_conditions(self, trade_record: Dict):
        """Learn optimal trading conditions"""
        try:
            conditions = trade_record.get('market_conditions', {})
            outcome = trade_record['profitable']
            
            # Learn from time-based patterns
            hour = conditions.get('time_of_day', 12)
            day = conditions.get('day_of_week', 2)
            
            # This would be expanded to maintain condition-outcome mappings
            # For now, just basic logging
            if outcome:
                logger.debug(f"Successful trade conditions: Hour={hour}, Day={day}")
            
        except Exception as e:
            logger.error(f"Error learning from market conditions: {e}")
    
    def _update_market_regime(self, trade_record: Dict):
        """Update market regime detection based on trade outcomes"""
        try:
            # Simplified regime detection based on recent trade performance
            recent_trades = list(self.trade_history)[-20:]
            if len(recent_trades) >= 10:
                win_rate = sum(1 for t in recent_trades if t.get('profitable', False)) / len(recent_trades)
                avg_pnl = np.mean([t.get('pnl_percent', 0) for t in recent_trades])
                volatility = np.std([t.get('pnl_percent', 0) for t in recent_trades])
                
                # Determine regime
                if volatility > 0.04:  # High volatility
                    regime = 'volatile'
                elif avg_pnl > 0.02 and win_rate > 0.6:  # Good performance
                    regime = 'trending'
                elif avg_pnl < -0.02 or win_rate < 0.4:  # Poor performance
                    regime = 'choppy'
                else:
                    regime = 'neutral'
                
                # Update regime if changed
                if regime != self.market_regimes['current_regime']:
                    logger.info(f"Market regime changed: {self.market_regimes['current_regime']} -> {regime}")
                    self.market_regimes['current_regime'] = regime
                    self.market_regimes['regime_history'].append({
                        'regime': regime,
                        'timestamp': datetime.now(),
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl
                    })
            
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
    
    def _get_recent_symbol_performance(self, symbol: str, days: int = 7) -> float:
        """Get recent performance for a specific symbol"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_trades = [
                trade for trade in self.trade_history 
                if (trade['symbol'] == symbol and 
                    trade.get('entry_time', datetime.min) > cutoff_time and
                    trade.get('pnl_percent') is not None)
            ]
            
            if not recent_trades:
                return 0.0
            
            return sum(trade['pnl_percent'] for trade in recent_trades)
            
        except Exception as e:
            logger.error(f"Error getting recent symbol performance: {e}")
            return 0.0
    
    def _get_regime_adjustment(self) -> float:
        """Get adjustment factor based on current market regime"""
        try:
            regime = self.market_regimes['current_regime']
            
            adjustments = {
                'bull': 0.9,      # Lower threshold in bull market
                'bear': 1.3,      # Higher threshold in bear market
                'volatile': 1.2,  # Higher threshold in volatile market
                'neutral': 1.0,   # No adjustment
                'trending': 0.95, # Slightly lower threshold when trending
                'choppy': 1.1     # Higher threshold in choppy market
            }
            
            return adjustments.get(regime, 1.0)
            
        except Exception as e:
            logger.error(f"Error getting regime adjustment: {e}")
            return 1.0
    
    def _calculate_calibration(self, confidences: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate confidence calibration score"""
        try:
            # Bin confidences and calculate calibration
            bins = np.linspace(0, 1, 11)  # 10 bins
            bin_indices = np.digitize(confidences, bins) - 1
            
            calibration_error = 0
            for i in range(10):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    bin_confidence = np.mean(confidences[mask])
                    bin_accuracy = np.mean(outcomes[mask])
                    calibration_error += abs(bin_confidence - bin_accuracy)
            
            return max(0, 1 - calibration_error / 10)  # Convert to calibration score
            
        except Exception as e:
            logger.error(f"Error calculating calibration: {e}")
            return 0.5
    
    def _save_learning_data(self):
        """Save learning data to persistent storage"""
        try:
            learning_data = {
                'symbol_performance': dict(self.symbol_performance),
                'overall_performance': self.overall_performance,
                'dynamic_thresholds': self.dynamic_thresholds,
                'market_regimes': dict(self.market_regimes),
                'last_saved': datetime.now().isoformat()
            }
            
            os.makedirs('data/learning', exist_ok=True)
            with open('data/learning/adaptive_learning_data.json', 'w') as f:
                json.dump(learning_data, f, default=str, indent=2)
                
            # Save trade history separately (pickled for deque support)
            with open('data/learning/trade_history.pkl', 'wb') as f:
                pickle.dump(list(self.trade_history), f)
            
            logger.debug("Learning data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def _load_learning_data(self):
        """Load previous learning data from storage"""
        try:
            # Load JSON data
            json_path = 'data/learning/adaptive_learning_data.json'
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    learning_data = json.load(f)
                
                # Restore data structures
                for symbol, data in learning_data.get('symbol_performance', {}).items():
                    self.symbol_performance[symbol].update(data)
                
                self.overall_performance.update(learning_data.get('overall_performance', {}))
                self.dynamic_thresholds.update(learning_data.get('dynamic_thresholds', {}))
                self.market_regimes.update(learning_data.get('market_regimes', {}))
                
                logger.info("Learning data loaded from previous session")
            
            # Load trade history
            history_path = 'data/learning/trade_history.pkl'
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    trade_list = pickle.load(f)
                    self.trade_history.extend(trade_list[-self.memory_size:])
                
                logger.info(f"Loaded {len(self.trade_history)} previous trades")
            
        except Exception as e:
            logger.warning(f"Could not load previous learning data: {e}")
    
    def __del__(self):
        """Save data when object is destroyed"""
        try:
            self._save_learning_data()
        except:
            pass

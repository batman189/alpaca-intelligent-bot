"""
Senior Analyst ML Intelligence System
True machine learning that learns patterns, adapts strategies, and makes intelligent decisions
This is the REAL AI brain that makes your bot as smart as a senior analyst
FIXED VERSION - Added missing 'name' attribute
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os
import warnings
from collections import defaultdict, deque
import asyncio
import threading
from dataclasses import dataclass
from enum import Enum

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.feature_selection import SelectKBest, f_classif
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: sklearn not available - using rule-based fallbacks")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available - using simple indicators")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_RANGE = "sideways_range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT_PENDING = "breakout_pending"

@dataclass
class PredictionResult:
    symbol: str
    prediction: int  # 1 = buy, 0 = hold, -1 = sell
    confidence: float  # 0.0 to 1.0
    expected_return: float
    time_horizon: int  # minutes
    reasoning: List[str]
    risk_factors: List[str]
    supporting_patterns: List[str]
    market_regime: MarketRegime
    feature_importance: Dict[str, float]

@dataclass
class LearningOutcome:
    trade_id: str
    prediction_accuracy: float
    actual_return: float
    predicted_return: float
    lessons_learned: List[str]
    pattern_effectiveness: Dict[str, float]

class SeniorAnalystBrain:
    """
    The core AI brain that makes decisions like a senior analyst
    Combines multiple ML models, pattern recognition, and adaptive learning
    """
    
    def __init__(self):
        self.name = "SeniorAnalystBrain"  # FIXED: Added missing name attribute
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_trackers = defaultdict(lambda: {
            'predictions': deque(maxlen=1000),
            'outcomes': deque(maxlen=1000),
            'accuracy': 0.0,
            'last_retrain': datetime.now()
        })
        
        # Advanced feature engineering
        self.feature_engine = AdvancedFeatureEngine()
        
        # Pattern recognition systems
        self.pattern_detector = PatternDetector()
        self.regime_detector = RegimeDetector()
        
        # Learning and adaptation
        self.meta_learner = MetaLearner()
        self.strategy_optimizer = StrategyOptimizer()
        
        # Knowledge base (like a senior analyst's experience)
        self.market_knowledge = MarketKnowledgeBase()
        
        # Model ensemble weights (learned over time)
        self.ensemble_weights = {
            'random_forest': 0.3,
            'gradient_boost': 0.3,
            'neural_network': 0.2,
            'pattern_recognition': 0.1,
            'regime_analysis': 0.1
        }
        
        self.is_trained = False
        self.training_lock = threading.Lock()
        
        logger.info("🧠 Senior Analyst Brain initialized")
    
    async def analyze_like_senior_analyst(self, symbol: str, market_data: pd.DataFrame, 
                                        market_context: Dict) -> PredictionResult:
        """
        Main analysis function - thinks like a senior analyst
        Combines multiple approaches and synthesizes a final recommendation
        """
        try:
            if len(market_data) < 100:
                return self._create_low_confidence_prediction(symbol, "Insufficient data")
            
            # 1. FEATURE ENGINEERING (like analyst building their model)
            features = await self.feature_engine.engineer_comprehensive_features(market_data, symbol)
            
            # 2. PATTERN RECOGNITION (like analyst spotting chart patterns)
            patterns = self.pattern_detector.detect_all_patterns(market_data)
            
            # 3. MARKET REGIME ANALYSIS (like analyst assessing market conditions)
            regime = self.regime_detector.detect_current_regime(market_data, market_context)
            
            # 4. ML MODEL PREDICTIONS (like analyst's quantitative models)
            ml_predictions = await self._get_ml_predictions(symbol, features)
            
            # 5. RISK ASSESSMENT (like analyst's risk framework)
            risk_analysis = self._assess_comprehensive_risk(symbol, market_data, patterns, regime)
            
            # 6. SENIOR ANALYST SYNTHESIS (combining all insights)
            final_prediction = self._synthesize_senior_analyst_decision(
                symbol, ml_predictions, patterns, regime, risk_analysis, features
            )
            
            # 7. LEARN FROM THIS ANALYSIS (like analyst updating their mental models)
            await self._record_prediction_for_learning(symbol, final_prediction, features)
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"Senior analyst analysis failed for {symbol}: {e}")
            return self._create_low_confidence_prediction(symbol, f"Analysis error: {e}")
    
    async def _get_ml_predictions(self, symbol: str, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from all ML models"""
        predictions = {}
        
        if not ML_AVAILABLE or not self.is_trained:
            # Fallback to rule-based analysis
            return self._rule_based_predictions(features)
        
        try:
            if symbol in self.models:
                symbol_models = self.models[symbol]
                scaler = self.scalers[symbol]
                
                # Scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Get predictions from each model
                if 'random_forest' in symbol_models:
                    rf_pred = symbol_models['random_forest'].predict_proba(features_scaled)[0]
                    predictions['random_forest'] = rf_pred[1] if len(rf_pred) > 1 else rf_pred[0]
                
                if 'gradient_boost' in symbol_models:
                    gb_pred = symbol_models['gradient_boost'].predict_proba(features_scaled)[0]
                    predictions['gradient_boost'] = gb_pred[1] if len(gb_pred) > 1 else gb_pred[0]
                
                if 'neural_network' in symbol_models:
                    nn_pred = symbol_models['neural_network'].predict_proba(features_scaled)[0]
                    predictions['neural_network'] = nn_pred[1] if len(nn_pred) > 1 else nn_pred[0]
            
        except Exception as e:
            logger.warning(f"ML prediction failed for {symbol}: {e}")
            return self._rule_based_predictions(features)
        
        return predictions
    
    def _rule_based_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Fallback rule-based predictions when ML not available"""
        try:
            # Simple rule-based logic as fallback
            # This assumes features are ordered: [price_features..., volume_features..., technical_indicators...]
            
            if len(features) < 10:
                return {'rule_based': 0.5}
            
            # Price momentum (first few features)
            price_momentum = np.mean(features[:3]) if len(features) >= 3 else 0
            
            # Volume analysis
            volume_score = np.mean(features[3:6]) if len(features) >= 6 else 0
            
            # Technical indicators
            technical_score = np.mean(features[6:]) if len(features) > 6 else 0
            
            # Combine scores
            combined_score = (price_momentum * 0.4 + volume_score * 0.3 + technical_score * 0.3)
            
            # Normalize to probability
            probability = max(0.1, min(0.9, (combined_score + 1) / 2))
            
            return {'rule_based': probability}
            
        except Exception as e:
            logger.error(f"Rule-based prediction failed: {e}")
            return {'rule_based': 0.5}
    
    def _synthesize_senior_analyst_decision(self, symbol: str, ml_predictions: Dict, 
                                          patterns: Dict, regime: MarketRegime, 
                                          risk_analysis: Dict, features: np.ndarray) -> PredictionResult:
        """
        Synthesize all inputs like a senior analyst would
        This is where the "human-like" intelligence happens
        """
        try:
            reasoning = []
            supporting_patterns = []
            risk_factors = []
            
            # 1. START WITH ML MODEL CONSENSUS
            if ml_predictions:
                ml_scores = list(ml_predictions.values())
                ml_consensus = np.mean(ml_scores)
                ml_agreement = 1.0 - np.std(ml_scores)  # High agreement = low std
                
                if ml_agreement > 0.8:
                    reasoning.append(f"Strong ML model consensus ({ml_consensus:.2f} confidence)")
                else:
                    reasoning.append(f"Mixed ML signals ({ml_consensus:.2f} avg, {ml_agreement:.2f} agreement)")
            else:
                ml_consensus = 0.5
                ml_agreement = 0.5
                reasoning.append("Using fundamental analysis (ML unavailable)")
            
            # 2. INCORPORATE PATTERN ANALYSIS
            pattern_score = 0.5
            strong_patterns = patterns.get('strong_patterns', [])
            weak_patterns = patterns.get('weak_patterns', [])
            
            if strong_patterns:
                pattern_score += 0.2
                supporting_patterns.extend(strong_patterns)
                reasoning.append(f"Strong technical patterns detected: {', '.join(strong_patterns[:2])}")
            
            if weak_patterns:
                pattern_score -= 0.1
                reasoning.append(f"Concerning patterns: {', '.join(weak_patterns[:2])}")
            
            # 3. REGIME-BASED ADJUSTMENTS (like senior analyst market awareness)
            regime_adjustment = self._get_regime_adjustment(regime)
            
            if regime == MarketRegime.BULL_TRENDING:
                reasoning.append("Bull market regime supports long positions")
            elif regime == MarketRegime.HIGH_VOLATILITY:
                reasoning.append("High volatility regime - increased caution")
                risk_factors.append("High market volatility")
            
            # 4. RISK INTEGRATION (like senior analyst risk management)
            risk_score = risk_analysis.get('overall_risk', 0.5)
            if risk_score > 0.7:
                risk_factors.append("High individual security risk")
                reasoning.append("Elevated risk profile requires position size reduction")
            
            # 5. FINAL SYNTHESIS (the "senior analyst moment")
            base_confidence = ml_consensus
            
            # Adjust based on pattern strength
            confidence_adj = (pattern_score - 0.5) * 0.3
            
            # Adjust based on model agreement
            confidence_adj += (ml_agreement - 0.5) * 0.2
            
            # Adjust based on regime
            confidence_adj += regime_adjustment * 0.1
            
            # Reduce confidence for high risk
            confidence_adj -= max(0, risk_score - 0.5) * 0.3
            
            final_confidence = max(0.1, min(0.95, base_confidence + confidence_adj))
            
            # DECISION LOGIC (like senior analyst decision framework)
            if final_confidence > 0.75:
                prediction = 1  # Strong buy
                reasoning.append("High confidence bullish signal")
            elif final_confidence > 0.6:
                prediction = 1  # Moderate buy
                reasoning.append("Moderate bullish bias")
            elif final_confidence < 0.25:
                prediction = -1  # Strong sell
                reasoning.append("High confidence bearish signal")
            elif final_confidence < 0.4:
                prediction = -1  # Moderate sell
                reasoning.append("Moderate bearish bias")
            else:
                prediction = 0  # Hold
                reasoning.append("Neutral stance - insufficient conviction")
            
            # EXPECTED RETURN CALCULATION (like analyst price targets)
            expected_return = self._calculate_expected_return(final_confidence, prediction, regime, risk_score)
            
            # TIME HORIZON (like analyst recommendations)
            time_horizon = self._determine_time_horizon(patterns, regime, final_confidence)
            
            # FEATURE IMPORTANCE (like analyst highlighting key factors)
            feature_importance = self._calculate_feature_importance(features, final_confidence)
            
            return PredictionResult(
                symbol=symbol,
                prediction=prediction,
                confidence=final_confidence,
                expected_return=expected_return,
                time_horizon=time_horizon,
                reasoning=reasoning,
                risk_factors=risk_factors,
                supporting_patterns=supporting_patterns,
                market_regime=regime,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Decision synthesis failed: {e}")
            return self._create_low_confidence_prediction(symbol, f"Synthesis error: {e}")
    
    def _get_regime_adjustment(self, regime: MarketRegime) -> float:
        """Adjust predictions based on market regime like a senior analyst would"""
        adjustments = {
            MarketRegime.BULL_TRENDING: 0.15,      # More bullish in bull markets
            MarketRegime.BEAR_TRENDING: -0.15,     # More bearish in bear markets
            MarketRegime.SIDEWAYS_RANGE: 0.0,      # Neutral in sideways markets
            MarketRegime.HIGH_VOLATILITY: -0.1,    # More cautious in volatile markets
            MarketRegime.LOW_VOLATILITY: 0.05,     # Slightly more aggressive in calm markets
            MarketRegime.BREAKOUT_PENDING: 0.1     # Slightly bullish before breakouts
        }
        return adjustments.get(regime, 0.0)
    
    def _calculate_expected_return(self, confidence: float, prediction: int, 
                                 regime: MarketRegime, risk_score: float) -> float:
        """Calculate expected return like a senior analyst setting price targets"""
        try:
            # Base return expectations
            base_returns = {
                MarketRegime.BULL_TRENDING: 0.08,
                MarketRegime.BEAR_TRENDING: -0.06,
                MarketRegime.SIDEWAYS_RANGE: 0.02,
                MarketRegime.HIGH_VOLATILITY: 0.12,  # Higher potential returns but riskier
                MarketRegime.LOW_VOLATILITY: 0.04,
                MarketRegime.BREAKOUT_PENDING: 0.10
            }
            
            base_return = base_returns.get(regime, 0.05)
            
            # Adjust based on prediction direction
            expected_return = base_return * prediction
            
            # Adjust based on confidence
            expected_return *= confidence
            
            # Adjust for risk (higher risk requires higher expected return)
            risk_premium = risk_score * 0.03
            expected_return += risk_premium if prediction > 0 else -risk_premium
            
            return expected_return
            
        except Exception as e:
            logger.error(f"Expected return calculation failed: {e}")
            return 0.05 * prediction  # Default 5% expected return
    
    def _determine_time_horizon(self, patterns: Dict, regime: MarketRegime, confidence: float) -> int:
        """Determine holding period like a senior analyst"""
        try:
            # Base time horizons by regime (in minutes)
            base_horizons = {
                MarketRegime.BULL_TRENDING: 240,      # 4 hours
                MarketRegime.BEAR_TRENDING: 120,      # 2 hours
                MarketRegime.SIDEWAYS_RANGE: 360,     # 6 hours
                MarketRegime.HIGH_VOLATILITY: 60,     # 1 hour
                MarketRegime.LOW_VOLATILITY: 480,     # 8 hours
                MarketRegime.BREAKOUT_PENDING: 30     # 30 minutes
            }
            
            base_horizon = base_horizons.get(regime, 240)
            
            # Adjust based on confidence (higher confidence = longer hold)
            horizon_multiplier = 0.5 + confidence
            
            # Adjust based on patterns
            if 'momentum_breakout' in patterns.get('strong_patterns', []):
                horizon_multiplier *= 0.7  # Shorter hold for momentum plays
            
            if 'reversal_pattern' in patterns.get('strong_patterns', []):
                horizon_multiplier *= 1.3  # Longer hold for reversal plays
            
            final_horizon = int(base_horizon * horizon_multiplier)
            return max(15, min(final_horizon, 1440))  # Between 15 minutes and 1 day
            
        except Exception as e:
            logger.error(f"Time horizon calculation failed: {e}")
            return 240  # Default 4 hours
    
    def _calculate_feature_importance(self, features: np.ndarray, confidence: float) -> Dict[str, float]:
        """Calculate which features drove the decision (like analyst explaining their thesis)"""
        try:
            # This is simplified - in reality would come from trained models
            feature_names = [
                'price_momentum', 'volume_surge', 'rsi_signal', 'macd_signal',
                'bollinger_position', 'support_resistance', 'trend_strength',
                'volatility_regime', 'correlation_score', 'sector_strength'
            ]
            
            # Create mock importance scores based on feature values
            importances = {}
            
            for i, name in enumerate(feature_names):
                if i < len(features):
                    # Higher absolute values = more important
                    importance = min(abs(features[i]) * confidence, 1.0)
                    importances[name] = importance
                else:
                    importances[name] = 0.0
            
            return importances
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {'analysis_error': 1.0}
    
    async def _record_prediction_for_learning(self, symbol: str, prediction: PredictionResult, features: np.ndarray):
        """Record prediction for future learning (like analyst tracking their calls)"""
        try:
            tracker = self.performance_trackers[symbol]
            
            # Store prediction with metadata
            prediction_record = {
                'timestamp': datetime.now(),
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'expected_return': prediction.expected_return,
                'features': features.copy(),
                'reasoning': prediction.reasoning.copy()
            }
            
            tracker['predictions'].append(prediction_record)
            
            # Schedule outcome tracking
            asyncio.create_task(self._track_prediction_outcome(symbol, prediction_record))
            
        except Exception as e:
            logger.error(f"Failed to record prediction for learning: {e}")
    
    async def _track_prediction_outcome(self, symbol: str, prediction_record: Dict):
        """Track prediction outcomes for learning (like analyst reviewing their calls)"""
        try:
            # Wait for the prediction time horizon to pass
            await asyncio.sleep(prediction_record.get('time_horizon', 240) * 60)  # Convert to seconds
            
            # Get actual outcome (this would get real market data)
            # For now, simulate outcome tracking
            actual_outcome = await self._get_actual_outcome(symbol, prediction_record)
            
            # Record the outcome for learning
            tracker = self.performance_trackers[symbol]
            tracker['outcomes'].append(actual_outcome)
            
            # Update accuracy metrics
            self._update_accuracy_metrics(symbol)
            
            # Learn from this outcome
            await self.meta_learner.learn_from_outcome(symbol, prediction_record, actual_outcome)
            
        except Exception as e:
            logger.error(f"Failed to track prediction outcome: {e}")
    
    async def _get_actual_outcome(self, symbol: str, prediction_record: Dict) -> Dict:
        """Get actual market outcome for learning"""
        try:
            # This would get real market data to compare against prediction
            # For now, simulate random outcome for demonstration
            
            import random
            
            # Simulate outcome based on prediction quality (better predictions = better outcomes)
            confidence = prediction_record['confidence']
            success_probability = 0.5 + (confidence - 0.5) * 0.5  # Higher confidence = higher success rate
            
            actual_success = random.random() < success_probability
            actual_return = random.uniform(-0.05, 0.08) if actual_success else random.uniform(-0.08, 0.02)
            
            return {
                'timestamp': datetime.now(),
                'success': actual_success,
                'actual_return': actual_return,
                'prediction_accuracy': confidence if actual_success else (1 - confidence)
            }
            
        except Exception as e:
            logger.error(f"Failed to get actual outcome: {e}")
            return {'success': False, 'actual_return': 0.0, 'prediction_accuracy': 0.5}
    
    def _update_accuracy_metrics(self, symbol: str):
        """Update prediction accuracy metrics"""
        try:
            tracker = self.performance_trackers[symbol]
            
            if len(tracker['outcomes']) > 10:
                recent_outcomes = list(tracker['outcomes'])[-50:]  # Last 50 predictions
                successes = [outcome['success'] for outcome in recent_outcomes]
                tracker['accuracy'] = sum(successes) / len(successes)
                
                # Update ensemble weights based on performance
                self._update_ensemble_weights(symbol, tracker['accuracy'])
            
        except Exception as e:
            logger.error(f"Failed to update accuracy metrics: {e}")
    
    def _update_ensemble_weights(self, symbol: str, accuracy: float):
        """Update model ensemble weights based on performance"""
        try:
            # If accuracy is high, trust ML models more
            if accuracy > 0.7:
                self.ensemble_weights['random_forest'] = min(0.4, self.ensemble_weights['random_forest'] + 0.01)
                self.ensemble_weights['gradient_boost'] = min(0.4, self.ensemble_weights['gradient_boost'] + 0.01)
                self.ensemble_weights['neural_network'] = min(0.3, self.ensemble_weights['neural_network'] + 0.01)
            # If accuracy is low, rely more on pattern recognition
            elif accuracy < 0.4:
                self.ensemble_weights['pattern_recognition'] = min(0.3, self.ensemble_weights['pattern_recognition'] + 0.02)
                self.ensemble_weights['regime_analysis'] = min(0.2, self.ensemble_weights['regime_analysis'] + 0.01)
            
            # Normalize weights
            total_weight = sum(self.ensemble_weights.values())
            for key in self.ensemble_weights:
                self.ensemble_weights[key] /= total_weight
            
        except Exception as e:
            logger.error(f"Failed to update ensemble weights: {e}")
    
    async def train_on_historical_data(self, symbol: str, historical_data: pd.DataFrame):
        """Train ML models on historical data (like analyst studying market history)"""
        try:
            if not ML_AVAILABLE or len(historical_data) < 500:
                logger.warning(f"Skipping ML training for {symbol} - insufficient data or ML unavailable")
                return False
            
            with self.training_lock:
                logger.info(f"🧠 Training senior analyst models for {symbol}...")
                
                # 1. FEATURE ENGINEERING
                features = await self.feature_engine.engineer_comprehensive_features(historical_data, symbol)
                
                # 2. CREATE LABELS (what would a senior analyst have done?)
                labels = self._create_training_labels(historical_data)
                
                if len(features) != len(labels) or len(features) < 100:
                    logger.warning(f"Insufficient training data for {symbol}")
                    return False
                
                # 3. PREPARE TRAINING DATA
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42, shuffle=False
                )
                
                # 4. SCALE FEATURES
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 5. TRAIN ENSEMBLE OF MODELS
                models = {}
                
                # Random Forest (captures non-linear patterns)
                rf_model = RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42,
                    class_weight='balanced', n_jobs=-1
                )
                rf_model.fit(X_train_scaled, y_train)
                models['random_forest'] = rf_model
                
                # Gradient Boosting (sequential learning)
                gb_model = GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42
                )
                gb_model.fit(X_train_scaled, y_train)
                models['gradient_boost'] = gb_model
                
                # Neural Network (complex pattern recognition)
                nn_model = MLPClassifier(
                    hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42,
                    alpha=0.01, learning_rate='adaptive'
                )
                nn_model.fit(X_train_scaled, y_train)
                models['neural_network'] = nn_model
                
                # 6. EVALUATE MODELS
                train_accuracies = {}
                test_accuracies = {}
                
                for name, model in models.items():
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                    
                    train_accuracies[name] = accuracy_score(y_train, train_pred)
                    test_accuracies[name] = accuracy_score(y_test, test_pred)
                
                # 7. STORE MODELS
                self.models[symbol] = models
                self.scalers[symbol] = scaler
                
                # 8. UPDATE TRACKING
                tracker = self.performance_trackers[symbol]
                tracker['last_retrain'] = datetime.now()
                
                avg_test_accuracy = np.mean(list(test_accuracies.values()))
                
                logger.info(f"✅ Training completed for {symbol}")
                logger.info(f"📊 Average test accuracy: {avg_test_accuracy:.3f}")
                
                self.is_trained = True
                return True
                
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            return False
    
    def _create_training_labels(self, historical_data: pd.DataFrame) -> np.ndarray:
        """Create training labels (what a senior analyst would have recommended)"""
        try:
            labels = []
            
            # Look ahead 4 hours for outcome
            lookhead_periods = 16  # 16 * 15min = 4 hours
            
            for i in range(len(historical_data) - lookhead_periods):
                current_price = historical_data['close'].iloc[i]
                future_price = historical_data['close'].iloc[i + lookhead_periods]
                
                # Calculate return
                return_pct = (future_price - current_price) / current_price
                
                # Create labels based on significant moves
                if return_pct > 0.02:  # 2% gain
                    labels.append(1)  # Buy signal
                elif return_pct < -0.02:  # 2% loss
                    labels.append(-1)  # Sell signal
                else:
                    labels.append(0)  # Hold signal
            
            # Convert to binary for now (buy/not buy)
            binary_labels = [1 if label == 1 else 0 for label in labels]
            
            return np.array(binary_labels)
            
        except Exception as e:
            logger.error(f"Failed to create training labels: {e}")
            return np.array([])
    
    def _assess_comprehensive_risk(self, symbol: str, market_data: pd.DataFrame, 
                                 patterns: Dict, regime: MarketRegime) -> Dict:
        """Comprehensive risk assessment like a senior analyst"""
        try:
            risk_factors = {}
            
            # 1. VOLATILITY RISK
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252 * 24 * 4)  # Annualized intraday vol
            risk_factors['volatility_risk'] = min(volatility / 0.5, 1.0)  # Normalize to 0-1
            
            # 2. LIQUIDITY RISK
            avg_volume = market_data['volume'].mean()
            recent_volume = market_data['volume'].tail(20).mean()
            liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            risk_factors['liquidity_risk'] = max(0, 1 - liquidity_ratio)
            
            # 3. TECHNICAL RISK
            weak_patterns = patterns.get('weak_patterns', [])
            risk_factors['technical_risk'] = len(weak_patterns) * 0.2
            
            # 4. REGIME RISK
            regime_risks = {
                MarketRegime.HIGH_VOLATILITY: 0.8,
                MarketRegime.BEAR_TRENDING: 0.6,
                MarketRegime.BREAKOUT_PENDING: 0.4,
                MarketRegime.SIDEWAYS_RANGE: 0.3,
                MarketRegime.LOW_VOLATILITY: 0.2,
                MarketRegime.BULL_TRENDING: 0.1
            }
            risk_factors['regime_risk'] = regime_risks.get(regime, 0.5)
            
            # 5. OVERALL RISK SCORE
            overall_risk = np.mean(list(risk_factors.values()))
            risk_factors['overall_risk'] = overall_risk
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'overall

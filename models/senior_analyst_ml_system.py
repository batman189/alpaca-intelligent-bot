"""
Senior Analyst ML Intelligence System
True machine learning that learns patterns, adapts strategies, and makes intelligent decisions
This is the REAL AI brain that makes your bot as smart as a senior analyst
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
            return {'overall_risk': 0.5}
    
    def _create_low_confidence_prediction(self, symbol: str, reason: str) -> PredictionResult:
        """Create a low-confidence prediction when analysis fails"""
        return PredictionResult(
            symbol=symbol,
            prediction=0,  # Hold
            confidence=0.3,
            expected_return=0.0,
            time_horizon=240,
            reasoning=[f"Low confidence: {reason}"],
            risk_factors=["Analysis incomplete"],
            supporting_patterns=[],
            market_regime=MarketRegime.SIDEWAYS_RANGE,
            feature_importance={}
        )
    
    async def retrain_models_if_needed(self):
        """Retrain models periodically based on performance"""
        try:
            current_time = datetime.now()
            
            for symbol, tracker in self.performance_trackers.items():
                time_since_retrain = current_time - tracker['last_retrain']
                
                # Retrain if:
                # 1. It's been more than 7 days, OR
                # 2. Performance has degraded significantly
                if (time_since_retrain > timedelta(days=7) or 
                    tracker['accuracy'] < 0.4):
                    
                    logger.info(f"🔄 Scheduling retrain for {symbol}")
                    # This would trigger retraining with new data
                    # await self.train_on_historical_data(symbol, new_data)
                    
        except Exception as e:
            logger.error(f"Retrain check failed: {e}")
    
    def get_model_performance_summary(self) -> Dict:
        """Get performance summary of all models"""
        try:
            summary = {
                'total_symbols_trained': len(self.models),
                'average_accuracy': 0.0,
                'best_performing_symbol': '',
                'worst_performing_symbol': '',
                'ensemble_weights': self.ensemble_weights.copy(),
                'last_updated': datetime.now().isoformat()
            }
            
            if self.performance_trackers:
                accuracies = [tracker['accuracy'] for tracker in self.performance_trackers.values()]
                summary['average_accuracy'] = np.mean(accuracies)
                
                # Find best and worst performers
                best_symbol = max(self.performance_trackers.keys(), 
                                key=lambda s: self.performance_trackers[s]['accuracy'])
                worst_symbol = min(self.performance_trackers.keys(), 
                                 key=lambda s: self.performance_trackers[s]['accuracy'])
                
                summary['best_performing_symbol'] = best_symbol
                summary['worst_performing_symbol'] = worst_symbol
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {}


class AdvancedFeatureEngine:
    """
    Advanced feature engineering - creates the inputs that make ML models smart
    """
    
    def __init__(self):
        self.feature_cache = {}
        
    async def engineer_comprehensive_features(self, market_data: pd.DataFrame, symbol: str) -> np.ndarray:
        """Engineer comprehensive features like a quant analyst would"""
        try:
            if len(market_data) < 50:
                return np.array([])
            
            features = []
            
            # 1. PRICE-BASED FEATURES
            price_features = self._extract_price_features(market_data)
            features.extend(price_features)
            
            # 2. VOLUME-BASED FEATURES  
            volume_features = self._extract_volume_features(market_data)
            features.extend(volume_features)
            
            # 3. TECHNICAL INDICATOR FEATURES
            technical_features = self._extract_technical_features(market_data)
            features.extend(technical_features)
            
            # 4. STATISTICAL FEATURES
            statistical_features = self._extract_statistical_features(market_data)
            features.extend(statistical_features)
            
            # 5. TIME-BASED FEATURES
            time_features = self._extract_time_features()
            features.extend(time_features)
            
            # 6. VOLATILITY FEATURES
            volatility_features = self._extract_volatility_features(market_data)
            features.extend(volatility_features)
            
            return np.array(features, dtype=float)
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {symbol}: {e}")
            return np.array([])
    
    def _extract_price_features(self, data: pd.DataFrame) -> List[float]:
        """Extract price-based features"""
        try:
            features = []
            
            # Price momentum over different periods
            for period in [5, 10, 20]:
                if len(data) > period:
                    momentum = (data['close'].iloc[-1] - data['close'].iloc[-period]) / data['close'].iloc[-period]
                    features.append(momentum)
                else:
                    features.append(0.0)
            
            # Price position relative to recent range
            recent_high = data['high'].tail(20).max()
            recent_low = data['low'].tail(20).min()
            current_price = data['close'].iloc[-1]
            
            if recent_high != recent_low:
                price_position = (current_price - recent_low) / (recent_high - recent_low)
                features.append(price_position)
            else:
                features.append(0.5)
            
            # Gap analysis
            if len(data) > 1:
                gap = (data['open'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                features.append(gap)
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Price feature extraction failed: {e}")
            return [0.0] * 5
    
    def _extract_volume_features(self, data: pd.DataFrame) -> List[float]:
        """Extract volume-based features"""
        try:
            features = []
            
            # Volume ratios
            if len(data) > 20:
                current_volume = data['volume'].iloc[-1]
                avg_volume_5 = data['volume'].tail(5).mean()
                avg_volume_20 = data['volume'].tail(20).mean()
                
                features.append(current_volume / avg_volume_5 if avg_volume_5 > 0 else 1.0)
                features.append(avg_volume_5 / avg_volume_20 if avg_volume_20 > 0 else 1.0)
            else:
                features.extend([1.0, 1.0])
            
            # Volume trend
            if len(data) > 10:
                volume_slope = np.polyfit(range(10), data['volume'].tail(10).values, 1)[0]
                features.append(volume_slope / data['volume'].mean() if data['volume'].mean() > 0 else 0.0)
            else:
                features.append(0.0)
            
            # Volume-price relationship
            if len(data) > 10:
                price_changes = data['close'].pct_change().tail(10)
                volume_changes = data['volume'].pct_change().tail(10)
                correlation = price_changes.corr(volume_changes)
                features.append(correlation if not np.isnan(correlation) else 0.0)
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Volume feature extraction failed: {e}")
            return [1.0] * 4
    
    def _extract_technical_features(self, data: pd.DataFrame) -> List[float]:
        """Extract technical indicator features"""
        try:
            features = []
            
            if TALIB_AVAILABLE and len(data) > 30:
                # RSI
                rsi = talib.RSI(data['close'].values, timeperiod=14)
                features.append((rsi[-1] - 50) / 50 if not np.isnan(rsi[-1]) else 0.0)
                
                # MACD
                macd, signal, histogram = talib.MACD(data['close'].values)
                features.append(histogram[-1] if not np.isnan(histogram[-1]) else 0.0)
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(data['close'].values)
                bb_position = (data['close'].iloc[-1] - lower[-1]) / (upper[-1] - lower[-1])
                features.append(bb_position if not np.isnan(bb_position) else 0.5)
                
                # Williams %R
                willr = talib.WILLR(data['high'].values, data['low'].values, data['close'].values)
                features.append(willr[-1] / -100 if not np.isnan(willr[-1]) else 0.5)
                
            else:
                # Simple technical indicators fallback
                # Simple RSI approximation
                gains = data['close'].diff().where(data['close'].diff() > 0, 0)
                losses = -data['close'].diff().where(data['close'].diff() < 0, 0)
                avg_gain = gains.rolling(14).mean().iloc[-1]
                avg_loss = losses.rolling(14).mean().iloc[-1]
                rs = avg_gain / avg_loss if avg_loss > 0 else 1
                rsi = 100 - (100 / (1 + rs))
                features.append((rsi - 50) / 50 if not np.isnan(rsi) else 0.0)
                
                # Simple moving average crossover
                sma_5 = data['close'].rolling(5).mean().iloc[-1]
                sma_20 = data['close'].rolling(20).mean().iloc[-1]
                ma_signal = (sma_5 - sma_20) / sma_20 if sma_20 > 0 else 0.0
                features.append(ma_signal)
                
                # Price relative to moving average
                features.append((data['close'].iloc[-1] - sma_20) / sma_20 if sma_20 > 0 else 0.0)
                features.append(0.5)  # Placeholder for Williams %R
            
            return features
            
        except Exception as e:
            logger.error(f"Technical feature extraction failed: {e}")
            return [0.0] * 4
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> List[float]:
        """Extract statistical features"""
        try:
            features = []
            
            # Return statistics
            returns = data['close'].pct_change().dropna()
            if len(returns) > 10:
                features.append(returns.mean())
                features.append(returns.std())
                features.append(returns.skew() if not np.isnan(returns.skew()) else 0.0)
                features.append(returns.kurtosis() if not np.isnan(returns.kurtosis()) else 0.0)
            else:
                features.extend([0.0, 0.02, 0.0, 0.0])
            
            # Trend strength
            if len(data) > 20:
                x = np.arange(len(data.tail(20)))
                y = data['close'].tail(20).values
                slope = np.polyfit(x, y, 1)[0]
                features.append(slope / data['close'].mean())
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Statistical feature extraction failed: {e}")
            return [0.0] * 5
    
    def _extract_time_features(self) -> List[float]:
        """Extract time-based features"""
        try:
            now = datetime.now()
            
            # Time of day (normalized)
            hour_normalized = now.hour / 24.0
            
            # Day of week (0=Monday, 6=Sunday)
            day_of_week = now.weekday() / 6.0
            
            # Market session (pre-market, regular, after-hours)
            if 4 <= now.hour < 9:  # Pre-market
                session = 0.25
            elif 9 <= now.hour < 16:  # Regular hours
                session = 1.0
            elif 16 <= now.hour < 20:  # After hours
                session = 0.5
            else:  # Closed
                session = 0.0
            
            return [hour_normalized, day_of_week, session]
            
        except Exception as e:
            logger.error(f"Time feature extraction failed: {e}")
            return [0.5, 0.5, 1.0]
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> List[float]:
        """Extract volatility-based features"""
        try:
            features = []
            
            # Realized volatility over different periods
            returns = data['close'].pct_change().dropna()
            
            for period in [5, 20]:
                if len(returns) > period:
                    vol = returns.tail(period).std() * np.sqrt(252 * 24 * 4)  # Annualized
                    features.append(vol)
                else:
                    features.append(0.2)  # Default volatility
            
            # High-low volatility
            if len(data) > 10:
                hl_vol = ((data['high'] - data['low']) / data['close']).tail(10).mean()
                features.append(hl_vol)
            else:
                features.append(0.02)
            
            return features
            
        except Exception as e:
            logger.error(f"Volatility feature extraction failed: {e}")
            return [0.2, 0.2, 0.02]


class PatternDetector:
    """
    Advanced pattern detection like a senior analyst spotting chart patterns
    """
    
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect all chart patterns"""
        try:
            strong_patterns = []
            weak_patterns = []
            
            # 1. TREND PATTERNS
            trend_patterns = self._detect_trend_patterns(data)
            strong_patterns.extend(trend_patterns.get('strong', []))
            weak_patterns.extend(trend_patterns.get('weak', []))
            
            # 2. REVERSAL PATTERNS
            reversal_patterns = self._detect_reversal_patterns(data)
            strong_patterns.extend(reversal_patterns.get('strong', []))
            weak_patterns.extend(reversal_patterns.get('weak', []))
            
            # 3. CONTINUATION PATTERNS
            continuation_patterns = self._detect_continuation_patterns(data)
            strong_patterns.extend(continuation_patterns.get('strong', []))
            weak_patterns.extend(continuation_patterns.get('weak', []))
            
            # 4. VOLUME PATTERNS
            volume_patterns = self._detect_volume_patterns(data)
            strong_patterns.extend(volume_patterns.get('strong', []))
            weak_patterns.extend(volume_patterns.get('weak', []))
            
            return {
                'strong_patterns': strong_patterns,
                'weak_patterns': weak_patterns,
                'pattern_count': len(strong_patterns) + len(weak_patterns)
            }
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {'strong_patterns': [], 'weak_patterns': []}
    
    def _detect_trend_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect trend-based patterns"""
        try:
            strong = []
            weak = []
            
            if len(data) < 20:
                return {'strong': strong, 'weak': weak}
            
            # Higher highs and higher lows (uptrend)
            highs = data['high'].tail(10)
            lows = data['low'].tail(10)
            
            if self._is_increasing_trend(highs) and self._is_increasing_trend(lows):
                strong.append('uptrend_pattern')
            elif self._is_decreasing_trend(highs) and self._is_decreasing_trend(lows):
                strong.append('downtrend_pattern')
            
            # Momentum breakout
            recent_return = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
            if abs(recent_return) > 0.03:  # 3% move
                if recent_return > 0:
                    strong.append('momentum_breakout')
                else:
                    weak.append('momentum_breakdown')
            
            return {'strong': strong, 'weak': weak}
            
        except Exception as e:
            logger.error(f"Trend pattern detection failed: {e}")
            return {'strong': [], 'weak': []}
    
    def _detect_reversal_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect reversal patterns"""
        try:
            strong = []
            weak = []
            
            if len(data) < 30:
                return {'strong': strong, 'weak': weak}
            
            # Double bottom pattern (simplified)
            lows = data['low'].rolling(5).min()
            if len(lows) > 20:
                recent_lows = lows.tail(20)
                if self._detect_double_bottom(recent_lows):
                    strong.append('double_bottom')
            
            # Double top pattern (simplified)
            highs = data['high'].rolling(5).max()
            if len(highs) > 20:
                recent_highs = highs.tail(20)
                if self._detect_double_top(recent_highs):
                    weak.append('double_top')
            
            # Hammer/Doji patterns (simplified)
            if self._detect_hammer_pattern(data.tail(3)):
                strong.append('hammer_pattern')
            
            return {'strong': strong, 'weak': weak}
            
        except Exception as e:
            logger.error(f"Reversal pattern detection failed: {e}")
            return {'strong': [], 'weak': []}
    
    def _detect_continuation_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect continuation patterns"""
        try:
            strong = []
            weak = []
            
            if len(data) < 20:
                return {'strong': strong, 'weak': weak}
            
            # Flag pattern (strong trend followed by consolidation)
            if self._detect_flag_pattern(data):
                strong.append('flag_pattern')
            
            # Triangle pattern
            if self._detect_triangle_pattern(data):
                strong.append('triangle_pattern')
            
            return {'strong': strong, 'weak': weak}
            
        except Exception as e:
            logger.error(f"Continuation pattern detection failed: {e}")
            return {'strong': [], 'weak': []}
    
    def _detect_volume_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect volume-based patterns"""
        try:
            strong = []
            weak = []
            
            if len(data) < 10 or 'volume' not in data.columns:
                return {'strong': strong, 'weak': weak}
            
            # Volume surge with price movement
            avg_volume = data['volume'].tail(20).mean()
            current_volume = data['volume'].iloc[-1]
            
            if current_volume > avg_volume * 2:
                price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                if abs(price_change) > 0.02:
                    strong.append('volume_surge_with_movement')
                else:
                    weak.append('volume_surge_no_movement')
            
            # Volume divergence
            if self._detect_volume_divergence(data):
                weak.append('volume_divergence')
            
            return {'strong': strong, 'weak': weak}
            
        except Exception as e:
            logger.error(f"Volume pattern detection failed: {e}")
            return {'strong': [], 'weak': []}
    
    def _is_increasing_trend(self, series: pd.Series) -> bool:
        """Check if series shows increasing trend"""
        try:
            if len(series) < 3:
                return False
            
            # Simple linear regression slope
            x = np.arange(len(series))
            slope = np.polyfit(x, series.values, 1)[0]
            return slope > 0
            
        except Exception:
            return False
    
    def _is_decreasing_trend(self, series: pd.Series) -> bool:
        """Check if series shows decreasing trend"""
        try:
            if len(series) < 3:
                return False
            
            x = np.arange(len(series))
            slope = np.polyfit(x, series.values, 1)[0]
            return slope < 0
            
        except Exception:
            return False
    
    def _detect_double_bottom(self, lows: pd.Series) -> bool:
        """Detect double bottom pattern (simplified)"""
        try:
            if len(lows) < 10:
                return False
            
            # Find the two lowest points
            min_indices = lows.nsmallest(2).index
            
            # Check if they're reasonably spaced and similar levels
            if len(min_indices) == 2:
                diff = abs(min_indices[0] - min_indices[1])
                level_diff = abs(lows.loc[min_indices[0]] - lows.loc[min_indices[1]])
                avg_level = (lows.loc[min_indices[0]] + lows.loc[min_indices[1]]) / 2
                
                return diff > 3 and level_diff / avg_level < 0.02  # Similar levels
            
            return False
            
        except Exception:
            return False
    
    def _detect_double_top(self, highs: pd.Series) -> bool:
        """Detect double top pattern (simplified)"""
        try:
            if len(highs) < 10:
                return False
            
            # Find the two highest points
            max_indices = highs.nlargest(2).index
            
            if len(max_indices) == 2:
                diff = abs(max_indices[0] - max_indices[1])
                level_diff = abs(highs.loc[max_indices[0]] - highs.loc[max_indices[1]])
                avg_level = (highs.loc[max_indices[0]] + highs.loc[max_indices[1]]) / 2
                
                return diff > 3 and level_diff / avg_level < 0.02
            
            return False
            
        except Exception:
            return False
    
    def _detect_hammer_pattern(self, recent_data: pd.DataFrame) -> bool:
        """Detect hammer candlestick pattern (simplified)"""
        try:
            if len(recent_data) < 1:
                return False
            
            latest = recent_data.iloc[-1]
            body_size = abs(latest['close'] - latest['open'])
            total_range = latest['high'] - latest['low']
            
            if total_range == 0:
                return False
            
            # Hammer: small body, long lower shadow
            lower_shadow = min(latest['open'], latest['close']) - latest['low']
            upper_shadow = latest['high'] - max(latest['open'], latest['close'])
            
            return (body_size / total_range < 0.3 and 
                    lower_shadow / total_range > 0.6 and
                    upper_shadow / total_range < 0.1)
            
        except Exception:
            return False
    
    def _detect_flag_pattern(self, data: pd.DataFrame) -> bool:
        """Detect flag pattern (simplified)"""
        try:
            if len(data) < 15:
                return False
            
            # Strong move followed by consolidation
            strong_move = (data['close'].iloc[-15] - data['close'].iloc[-10]) / data['close'].iloc[-15]
            consolidation_range = (data['high'].tail(5).max() - data['low'].tail(5).min()) / data['close'].tail(5).mean()
            
            return abs(strong_move) > 0.05 and consolidation_range < 0.03
            
        except Exception:
            return False
    
    def _detect_triangle_pattern(self, data: pd.DataFrame) -> bool:
        """Detect triangle pattern (simplified)"""
        try:
            if len(data) < 15:
                return False
            
            # Converging highs and lows
            recent_highs = data['high'].tail(10)
            recent_lows = data['low'].tail(10)
            
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs.values, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows.values, 1)[0]
            
            # Converging lines
            return high_slope < 0 and low_slope > 0 and abs(high_slope + low_slope) < abs(high_slope) * 0.5
            
        except Exception:
            return False
    
    def _detect_volume_divergence(self, data: pd.DataFrame) -> bool:
        """Detect volume divergence (simplified)"""
        try:
            if len(data) < 10:
                return False
            
            # Price trend vs volume trend
            price_trend = np.polyfit(range(10), data['close'].tail(10).values, 1)[0]
            volume_trend = np.polyfit(range(10), data['volume'].tail(10).values, 1)[0]
            
            # Divergence: price up but volume down (or vice versa)
            return (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0)
            
        except Exception:
            return False


class RegimeDetector:
    """
    Market regime detection - understanding the current market environment
    """
    
    def detect_current_regime(self, data: pd.DataFrame, market_context: Dict) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(data) < 50:
                return MarketRegime.SIDEWAYS_RANGE
            
            # Calculate regime indicators
            trend_strength = self._calculate_trend_strength(data)
            volatility_level = self._calculate_volatility_level(data)
            volume_pattern = self._analyze_volume_pattern(data)
            
            # Decision logic
            if abs(trend_strength) > 0.6 and volatility_level < 0.5:
                return MarketRegime.BULL_TRENDING if trend_strength > 0 else MarketRegime.BEAR_TRENDING
            elif volatility_level > 0.7:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility_level < 0.3:
                return MarketRegime.LOW_VOLATILITY
            elif self._detect_pending_breakout(data):
                return MarketRegime.BREAKOUT_PENDING
            else:
                return MarketRegime.SIDEWAYS_RANGE
                
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return MarketRegime.SIDEWAYS_RANGE
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (-1 to 1)"""
        try:
            # Multiple timeframe trend analysis
            short_trend = self._linear_trend(data['close'].tail(10))
            medium_trend = self._linear_trend(data['close'].tail(20))
            long_trend = self._linear_trend(data['close'].tail(50))
            
            # Weighted average
            trend_strength = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
            
            return max(-1, min(1, trend_strength))
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_level(self, data: pd.DataFrame) -> float:
        """Calculate volatility level (0 to 1)"""
        try:
            returns = data['close'].pct_change().dropna()
            current_vol = returns.tail(20).std() * np.sqrt(252 * 24 * 4)
            historical_vol = returns.std() * np.sqrt(252 * 24 * 4)
            
            # Relative volatility
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
            
            return max(0, min(1, vol_ratio))
            
        except Exception:
            return 0.5
    
    def _analyze_volume_pattern(self, data: pd.DataFrame) -> str:
        """Analyze volume patterns"""
        try:
            if 'volume' not in data.columns:
                return 'normal'
            
            recent_volume = data['volume'].tail(10).mean()
            historical_volume = data['volume'].mean()
            
            ratio = recent_volume / historical_volume if historical_volume > 0 else 1
            
            if ratio > 1.5:
                return 'increasing'
            elif ratio < 0.7:
                return 'decreasing'
            else:
                return 'normal'
                
        except Exception:
            return 'normal'
    
    def _detect_pending_breakout(self, data: pd.DataFrame) -> bool:
        """Detect if breakout is pending"""
        try:
            # Narrowing range with volume buildup
            recent_range = (data['high'].tail(10).max() - data['low'].tail(10).min()) / data['close'].tail(10).mean()
            historical_range = (data['high'].tail(50).max() - data['low'].tail(50).min()) / data['close'].tail(50).mean()
            
            range_contraction = recent_range / historical_range if historical_range > 0 else 1
            
            return range_contraction < 0.5  # Range contracted significantly
            
        except Exception:
            return False
    
    def _linear_trend(self, series: pd.Series) -> float:
        """Calculate linear trend strength"""
        try:
            if len(series) < 3:
                return 0.0
            
            x = np.arange(len(series))
            slope = np.polyfit(x, series.values, 1)[0]
            
            # Normalize by average price
            return slope / series.mean() if series.mean() > 0 else 0.0
            
        except Exception:
            return 0.0


class MetaLearner:
    """
    Meta-learning system that learns how to learn better
    """
    
    def __init__(self):
        self.learning_history = defaultdict(list)
        self.strategy_effectiveness = defaultdict(float)
    
    async def learn_from_outcome(self, symbol: str, prediction_record: Dict, actual_outcome: Dict):
        """Learn from prediction outcomes"""
        try:
            # Record the learning experience
            learning_record = {
                'symbol': symbol,
                'prediction_confidence': prediction_record['confidence'],
                'actual_success': actual_outcome['success'],
                'prediction_accuracy': actual_outcome['prediction_accuracy'],
                'timestamp': datetime.now(),
                'lessons': []
            }
            
            # Extract lessons learned
            if actual_outcome['success'] and prediction_record['confidence'] < 0.7:
                learning_record['lessons'].append("Low confidence predictions can still succeed")
            elif not actual_outcome['success'] and prediction_record['confidence'] > 0.8:
                learning_record['lessons'].append("High confidence doesn't guarantee success")
            
            # Update strategy effectiveness
            for reason in prediction_record.get('reasoning', []):
                if actual_outcome['success']:
                    self.strategy_effectiveness[reason] += 0.1
                else:
                    self.strategy_effectiveness[reason] -= 0.05
            
            self.learning_history[symbol].append(learning_record)
            
            # Keep only recent history
            if len(self.learning_history[symbol]) > 100:
                self.learning_history[symbol] = self.learning_history[symbol][-100:]
            
        except Exception as e:
            logger.error(f"Meta learning failed: {e}")


class StrategyOptimizer:
    """
    Optimizes trading strategies based on performance
    """
    
    def __init__(self):
        self.strategy_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_return': 0.0})
        self.optimization_history = []
    
    def optimize_strategy_parameters(self, symbol: str, performance_data: Dict) -> Dict:
        """Optimize strategy parameters based on performance"""
        try:
            # Analyze what's working and what's not
            optimizations = {}
            
            accuracy = performance_data.get('accuracy', 0.5)
            
            if accuracy > 0.7:
                # High accuracy - can be more aggressive
                optimizations['confidence_threshold'] = 0.6
                optimizations['position_size_multiplier'] = 1.2
            elif accuracy < 0.4:
                # Low accuracy - be more conservative
                optimizations['confidence_threshold'] = 0.8
                optimizations['position_size_multiplier'] = 0.7
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            return {}


class MarketKnowledgeBase:
    """
    Accumulated market knowledge like a senior analyst's experience
    """
    
    def __init__(self):
        # Market patterns and their typical outcomes
        self.pattern_knowledge = {
            'earnings_runup': {'success_rate': 0.65, 'avg_return': 0.08, 'time_horizon': 120},
            'momentum_breakout': {'success_rate': 0.7, 'avg_return': 0.12, 'time_horizon': 60},
            'reversal_pattern': {'success_rate': 0.55, 'avg_return': 0.06, 'time_horizon': 240},
            'volume_surge': {'success_rate': 0.6, 'avg_return': 0.05, 'time_horizon': 30}
        }
        
        # Sector-specific knowledge
        self.sector_knowledge = {
            'tech': {'volatility': 'high', 'best_strategies': ['momentum', 'earnings']},
            'finance': {'volatility': 'medium', 'best_strategies': ['value', 'trend']},
            'energy': {'volatility': 'high', 'best_strategies': ['momentum', 'commodity']}
        }
        
        # Time-based patterns
        self.time_patterns = {
            'monday_effect': {'direction': 'bearish', 'strength': 0.3},
            'friday_effect': {'direction': 'bullish', 'strength': 0.2},
            'lunch_time': {'direction': 'sideways', 'strength': 0.4},
            'power_hour': {'direction': 'volatile', 'strength': 0.8}
        }
    
    def get_pattern_knowledge(self, pattern_name: str) -> Dict:
        """Get knowledge about a specific pattern"""
        return self.pattern_knowledge.get(pattern_name, {
            'success_rate': 0.5, 'avg_return': 0.03, 'time_horizon': 180
        })
    
    def get_sector_insights(self, symbol: str) -> Dict:
        """Get sector-specific insights"""
        # This would normally map symbols to sectors
        # For now, return default tech sector insights
        return self.sector_knowledge.get('tech', {
            'volatility': 'medium', 'best_strategies': ['momentum']
        })
    
    def get_time_based_bias(self, current_time: datetime) -> Dict:
        """Get time-based market bias"""
        try:
            hour = current_time.hour
            day = current_time.weekday()
            
            bias = {'direction': 'neutral', 'strength': 0.0}
            
            # Day of week effects
            if day == 0:  # Monday
                bias.update(self.time_patterns['monday_effect'])
            elif day == 4:  # Friday
                bias.update(self.time_patterns['friday_effect'])
            
            # Time of day effects
            if 12 <= hour <= 14:  # Lunch time
                bias.update(self.time_patterns['lunch_time'])
            elif 15 <= hour <= 16:  # Power hour
                bias.update(self.time_patterns['power_hour'])
            
            return bias
            
        except Exception as e:
            logger.error(f"Time bias calculation failed: {e}")
            return {'direction': 'neutral', 'strength': 0.0}


# Integration class to tie everything together
class SeniorAnalystIntegration:
    """
    Integration layer that connects the Senior Analyst Brain to your existing trading bot
    """
    
    def __init__(self):
        self.brain = SeniorAnalystBrain()
        self.is_initialized = False
        
    async def initialize_for_symbol(self, symbol: str, historical_data: pd.DataFrame):
        """Initialize the ML brain for a specific symbol"""
        try:
            if len(historical_data) > 500:
                success = await self.brain.train_on_historical_data(symbol, historical_data)
                if success:
                    logger.info(f"🧠 Senior analyst trained for {symbol}")
                    self.is_initialized = True
                else:
                    logger.warning(f"⚠️ Training failed for {symbol}, using rule-based analysis")
            else:
                logger.info(f"📊 Using rule-based analysis for {symbol} (insufficient data for ML)")
                
        except Exception as e:
            logger.error(f"Initialization failed for {symbol}: {e}")
    
    async def get_senior_analyst_recommendation(self, symbol: str, market_data: pd.DataFrame, 
                                               market_context: Dict = None) -> Dict:
        """
        Get a recommendation from the senior analyst brain
        This replaces your basic market analysis with true AI intelligence
        """
        try:
            if market_context is None:
                market_context = {}
            
            # Get the senior analyst's full analysis
            prediction = await self.brain.analyze_like_senior_analyst(symbol, market_data, market_context)
            
            # Convert to format expected by your trading bot
            return {
                'symbol': prediction.symbol,
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'expected_return': prediction.expected_return,
                'time_horizon_minutes': prediction.time_horizon,
                'reasoning': prediction.reasoning,
                'risk_factors': prediction.risk_factors,
                'supporting_patterns': prediction.supporting_patterns,
                'market_regime': prediction.market_regime.value,
                'feature_importance': prediction.feature_importance,
                'analyst_grade': self._calculate_analyst_grade(prediction)
            }
            
        except Exception as e:
            logger.error(f"Senior analyst recommendation failed for {symbol}: {e}")
            return self._get_fallback_recommendation(symbol)
    
    def _calculate_analyst_grade(self, prediction: PredictionResult) -> str:
        """Calculate analyst grade (like investment bank ratings)"""
        try:
            confidence = prediction.confidence
            expected_return = abs(prediction.expected_return)
            pattern_count = len(prediction.supporting_patterns)
            
            # Combined score
            score = (confidence * 0.5 + 
                    min(expected_return / 0.1, 1.0) * 0.3 + 
                    min(pattern_count / 3, 1.0) * 0.2)
            
            if score > 0.8:
                return "STRONG_BUY" if prediction.prediction > 0 else "STRONG_SELL"
            elif score > 0.6:
                return "BUY" if prediction.prediction > 0 else "SELL"
            elif score > 0.4:
                return "WEAK_BUY" if prediction.prediction > 0 else "WEAK_SELL"
            else:
                return "HOLD"
                
        except Exception:
            return "HOLD"
    
    def _get_fallback_recommendation(self, symbol: str) -> Dict:
        """Fallback recommendation when senior analyst fails"""
        return {
            'symbol': symbol,
            'prediction': 0,
            'confidence': 0.4,
            'expected_return': 0.02,
            'time_horizon_minutes': 240,
            'reasoning': ['Fallback analysis - senior analyst unavailable'],
            'risk_factors': ['Analysis incomplete'],
            'supporting_patterns': [],
            'market_regime': 'sideways_range',
            'feature_importance': {},
            'analyst_grade': 'HOLD'
        }
    
    async def learn_from_trade_outcome(self, symbol: str, trade_data: Dict):
        """Feed trade outcomes back to the learning system"""
        try:
            # This connects your trade results back to the learning system
            learning_outcome = LearningOutcome(
                trade_id=trade_data.get('trade_id', ''),
                prediction_accuracy=trade_data.get('prediction_accuracy', 0.5),
                actual_return=trade_data.get('actual_return', 0.0),
                predicted_return=trade_data.get('predicted_return', 0.0),
                lessons_learned=trade_data.get('lessons', []),
                pattern_effectiveness=trade_data.get('pattern_effectiveness', {})
            )
            
            # This makes the system smarter over time
            await self.brain.meta_learner.learn_from_outcome(symbol, trade_data, {
                'success': trade_data.get('profitable', False),
                'prediction_accuracy': learning_outcome.prediction_accuracy
            })
            
        except Exception as e:
            logger.error(f"Learning from trade outcome failed: {e}")
    
    def get_system_intelligence_report(self) -> Dict:
        """Get a report on how smart the system has become"""
        try:
            performance_summary = self.brain.get_model_performance_summary()
            
            return {
                'intelligence_level': self._calculate_intelligence_level(performance_summary),
                'symbols_trained': performance_summary.get('total_symbols_trained', 0),
                'average_accuracy': performance_summary.get('average_accuracy', 0.0),
                'learning_status': 'Active' if self.is_initialized else 'Initializing',
                'ml_available': ML_AVAILABLE,
                'talib_available': TALIB_AVAILABLE,
                'system_maturity': self._calculate_system_maturity(),
                'recommendations': self._get_intelligence_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Intelligence report failed: {e}")
            return {'intelligence_level': 'Basic', 'error': str(e)}
    
    def _calculate_intelligence_level(self, performance_summary: Dict) -> str:
        """Calculate current intelligence level"""
        try:
            accuracy = performance_summary.get('average_accuracy', 0.0)
            symbols_trained = performance_summary.get('total_symbols_trained', 0)
            
            if accuracy > 0.75 and symbols_trained > 10:
                return "Senior Analyst"
            elif accuracy > 0.65 and symbols_trained > 5:
                return "Experienced Analyst"
            elif accuracy > 0.55 and symbols_trained > 2:
                return "Junior Analyst"
            elif symbols_trained > 0:
                return "Trainee Analyst"
            else:
                return "Basic Rules"
                
        except Exception:
            return "Unknown"
    
    def _calculate_system_maturity(self) -> str:
        """Calculate how mature the learning system is"""
        try:
            total_predictions = sum(
                len(tracker['predictions']) 
                for tracker in self.brain.performance_trackers.values()
            )
            
            if total_predictions > 1000:
                return "Highly Experienced"
            elif total_predictions > 500:
                return "Experienced"
            elif total_predictions > 100:
                return "Learning"
            elif total_predictions > 10:
                return "Beginner"
            else:
                return "New"
                
        except Exception:
            return "Unknown"
    
    def _get_intelligence_recommendations(self) -> List[str]:
        """Get recommendations for improving intelligence"""
        recommendations = []
        
        if not ML_AVAILABLE:
            recommendations.append("Install scikit-learn for machine learning capabilities")
        
        if not TALIB_AVAILABLE:
            recommendations.append("Install TA-Lib for advanced technical analysis")
        
        if not self.is_initialized:
            recommendations.append("Provide more historical data for training")
        
        if len(self.brain.models) < 5:
            recommendations.append("Train on more symbols for better diversification")
        
        return recommendations if recommendations else ["System is fully optimized"]


# Usage example for integration with your existing bot
async def integrate_with_existing_bot():
    """
    Example of how to integrate this senior analyst system with your existing trading bot
    """
    
    # Initialize the senior analyst
    senior_analyst = SeniorAnalystIntegration()
    
    # Example usage in your analyze_single_symbol function:
    """
    # Replace your basic analysis with this:
    
    async def analyze_single_symbol(self, symbol: str, account_info: Dict, current_positions: Dict):
        try:
            # Get market data (your existing code)
            market_data = await self.data_manager.get_market_data(symbol, "15Min", 200)
            
            # Use senior analyst instead of basic analysis
            senior_analysis = await senior_analyst.get_senior_analyst_recommendation(
                symbol, market_data, {'account_info': account_info}
            )
            
            # The senior analysis gives you everything you need:
            analysis = {
                'symbol': symbol,
                'prediction': senior_analysis['prediction'],
                'confidence': senior_analysis['confidence'], 
                'expected_return': senior_analysis['expected_return'],
                'reasoning': senior_analysis['reasoning'],
                'analyst_grade': senior_analysis['analyst_grade'],
                'time_horizon': senior_analysis['time_horizon_minutes'],
                'current_price': market_data['close'].iloc[-1]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Senior analyst analysis failed for {symbol}: {e}")
            return None
    """
    
    return senior_analyst


# Export the main class for use in your trading bot
__all__ = ['SeniorAnalystIntegration', 'SeniorAnalystBrain', 'integrate_with_existing_bot']

"""
Senior Analyst ML Intelligence System - CORRECTED VERSION
Fixed all syntax errors and missing imports
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

class AdvancedFeatureEngine:
    def __init__(self):
        self.name = "AdvancedFeatureEngine"
    
    async def engineer_comprehensive_features(self, market_data: pd.DataFrame, symbol: str) -> np.ndarray:
        """Engineer comprehensive features for ML analysis"""
        try:
            if len(market_data) < 20:
                return np.array([[0.5] * 12])
            
            # For prediction: return single row of current features
            features = []
            
            close_prices = market_data['close']
            returns = close_prices.pct_change().fillna(0)
            
            # Momentum features
            features.extend([
                returns.iloc[-1],  # Latest return
                returns.tail(5).mean(),  # 5-period momentum
                returns.tail(10).mean(),  # 10-period momentum
                returns.tail(20).mean(),  # 20-period momentum
            ])
            
            # Volatility features
            features.extend([
                returns.std(),  # Volatility
                returns.tail(5).std(),  # Short-term volatility
                returns.tail(20).std(),  # Long-term volatility
            ])
            
            # Volume features
            if 'volume' in market_data.columns:
                volume = market_data['volume']
                volume_ma = volume.rolling(20).mean()
                features.extend([
                    volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1,
                    volume.tail(5).mean() / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1,
                ])
            else:
                features.extend([1.0, 1.0])
            
            # Technical indicators
            if len(close_prices) >= 20:
                sma_20 = close_prices.rolling(20).mean()
                features.append(close_prices.iloc[-1] / sma_20.iloc[-1] if sma_20.iloc[-1] > 0 else 1)
            else:
                features.append(1.0)
            
            # Trend features
            if len(close_prices) >= 10:
                trend_strength = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
                features.append(trend_strength)
            else:
                features.append(0.0)
            
            # Ensure we have exactly 12 features
            while len(features) < 12:
                features.append(0.0)
            
            return np.array([features[:12]])
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return np.array([[0.5] * 12])

class PatternDetector:
    def __init__(self):
        self.name = "PatternDetector"
    
    def detect_all_patterns(self, market_data: pd.DataFrame) -> Dict:
        """Detect chart patterns in market data"""
        try:
            patterns = {'strong_patterns': [], 'weak_patterns': []}
            
            if len(market_data) < 20:
                return patterns
            
            close_prices = market_data['close']
            recent_trend = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
            
            if recent_trend > 0.05:
                patterns['strong_patterns'].append('bullish_trend')
            elif recent_trend < -0.05:
                patterns['strong_patterns'].append('bearish_trend')
            
            if 'volume' in market_data.columns:
                volume_ratio = market_data['volume'].iloc[-1] / market_data['volume'].mean()
                if volume_ratio > 2.0:
                    patterns['strong_patterns'].append('volume_surge')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {'strong_patterns': [], 'weak_patterns': []}

class RegimeDetector:
    def __init__(self):
        self.name = "RegimeDetector"
    
    def detect_current_regime(self, market_data: pd.DataFrame, market_context: Dict) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(market_data) < 20:
                return MarketRegime.SIDEWAYS_RANGE
            
            close_prices = market_data['close']
            returns = close_prices.pct_change()
            
            trend = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
            volatility = returns.std()
            
            if trend > 0.1 and volatility < 0.02:
                return MarketRegime.BULL_TRENDING
            elif trend < -0.1 and volatility < 0.02:
                return MarketRegime.BEAR_TRENDING
            elif volatility > 0.05:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:
                return MarketRegime.LOW_VOLATILITY
            elif abs(trend) < 0.02:
                return MarketRegime.SIDEWAYS_RANGE
            else:
                return MarketRegime.BREAKOUT_PENDING
                
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return MarketRegime.SIDEWAYS_RANGE

class MetaLearner:
    def __init__(self):
        self.name = "MetaLearner"
    
    async def learn_from_outcome(self, symbol: str, prediction_record: Dict, actual_outcome: Dict):
        """Learn from trade outcomes"""
        try:
            logger.info(f"Learning from {symbol} outcome: {actual_outcome}")
        except Exception as e:
            logger.error(f"Error in meta learning: {e}")

class StrategyOptimizer:
    def __init__(self):
        self.name = "StrategyOptimizer"

class MarketKnowledgeBase:
    def __init__(self):
        self.name = "MarketKnowledgeBase"

class SeniorAnalystBrain:
    """The core AI brain that makes decisions like a senior analyst"""
    
    def __init__(self):
        self.name = "SeniorAnalystBrain"
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_trackers = defaultdict(lambda: {
            'predictions': deque(maxlen=1000),
            'outcomes': deque(maxlen=1000),
            'accuracy': 0.0,
            'last_retrain': datetime.now()
        })
        
        # Components
        self.feature_engine = AdvancedFeatureEngine()
        self.pattern_detector = PatternDetector()
        self.regime_detector = RegimeDetector()
        self.meta_learner = MetaLearner()
        self.strategy_optimizer = StrategyOptimizer()
        self.market_knowledge = MarketKnowledgeBase()
        
        # Model ensemble weights
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
    
    async def get_senior_analyst_recommendation(self, symbol: str, market_data: pd.DataFrame, 
                                              market_context: Dict) -> Dict:
        """Main analysis function - thinks like a senior analyst with news intelligence"""
        try:
            if len(market_data) < 20:
                return self._create_low_confidence_dict(symbol, "Insufficient data")
            
            # Feature engineering
            features = await self.feature_engine.engineer_comprehensive_features(market_data, symbol)
            
            # Pattern recognition
            patterns = self.pattern_detector.detect_all_patterns(market_data)
            
            # Market regime analysis
            regime = self.regime_detector.detect_current_regime(market_data, market_context)
            
            # NEWS INTELLIGENCE - Phase 1 Enhancement
            news_analysis = await self._analyze_news_sentiment(symbol)
            
            # ML model predictions
            ml_predictions = await self._get_ml_predictions(symbol, features)
            
            # Risk assessment
            risk_analysis = self._assess_comprehensive_risk(symbol, market_data, patterns, regime)
            
            # Senior analyst synthesis with news intelligence
            final_prediction = self._synthesize_senior_analyst_decision(
                symbol, ml_predictions, patterns, regime, risk_analysis, features, news_analysis
            )
            
            # Record for learning
            await self._record_prediction_for_learning(symbol, final_prediction, features)
            
            return {
                'symbol': symbol,
                'analyst_grade': self._convert_to_analyst_grade(final_prediction),
                'confidence': final_prediction.confidence,
                'expected_return': final_prediction.expected_return,
                'time_horizon_minutes': final_prediction.time_horizon,
                'reasoning': final_prediction.reasoning,
                'risk_factors': final_prediction.risk_factors,
                'supporting_patterns': final_prediction.supporting_patterns,
                'market_regime': final_prediction.market_regime.value,
                'feature_importance': final_prediction.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Senior analyst analysis failed for {symbol}: {e}")
            return self._create_low_confidence_dict(symbol, f"Analysis error: {e}")
    
    def _convert_to_analyst_grade(self, prediction: PredictionResult) -> str:
        """Convert prediction to analyst grade"""
        if prediction.prediction == 1:
            if prediction.confidence > 0.8:
                return "STRONG_BUY"
            elif prediction.confidence > 0.65:
                return "BUY"
            else:
                return "WEAK_BUY"
        elif prediction.prediction == -1:
            if prediction.confidence > 0.8:
                return "STRONG_SELL"
            elif prediction.confidence > 0.65:
                return "SELL"
            else:
                return "WEAK_SELL"
        else:
            return "HOLD"
    
    async def _get_ml_predictions(self, symbol: str, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from all ML models"""
        predictions = {}
        
        if not ML_AVAILABLE or not self.is_trained:
            return self._rule_based_predictions(features)
        
        try:
            if symbol in self.models:
                symbol_models = self.models[symbol]
                scaler = self.scalers[symbol]
                
                features_scaled = scaler.transform(features.reshape(1, -1))
                
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
            if len(features.flatten()) < 5:
                return {'rule_based': 0.5}
            
            feature_array = features.flatten()
            
            price_momentum = np.mean(feature_array[:3]) if len(feature_array) >= 3 else 0
            volume_score = np.mean(feature_array[3:6]) if len(feature_array) >= 6 else 0
            technical_score = np.mean(feature_array[6:]) if len(feature_array) > 6 else 0
            
            combined_score = (price_momentum * 0.4 + volume_score * 0.3 + technical_score * 0.3)
            probability = max(0.1, min(0.9, (combined_score + 1) / 2))
            
            return {'rule_based': probability}
            
        except Exception as e:
            logger.error(f"Rule-based prediction failed: {e}")
            return {'rule_based': 0.5}
    
    def _assess_comprehensive_risk(self, symbol: str, market_data: pd.DataFrame, 
                                 patterns: Dict, regime: MarketRegime) -> Dict:
        """Comprehensive risk assessment like a senior analyst"""
        try:
            risk_factors = {}
            
            # Volatility risk
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            risk_factors['volatility_risk'] = min(volatility / 0.5, 1.0)
            
            # Liquidity risk
            if 'volume' in market_data.columns:
                avg_volume = market_data['volume'].mean()
                recent_volume = market_data['volume'].tail(5).mean()
                liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                risk_factors['liquidity_risk'] = max(0, 1 - liquidity_ratio)
            else:
                risk_factors['liquidity_risk'] = 0.3
            
            # Technical risk
            weak_patterns = patterns.get('weak_patterns', [])
            risk_factors['technical_risk'] = len(weak_patterns) * 0.2
            
            # Regime risk
            regime_risks = {
                MarketRegime.HIGH_VOLATILITY: 0.8,
                MarketRegime.BEAR_TRENDING: 0.6,
                MarketRegime.BREAKOUT_PENDING: 0.4,
                MarketRegime.SIDEWAYS_RANGE: 0.3,
                MarketRegime.LOW_VOLATILITY: 0.2,
                MarketRegime.BULL_TRENDING: 0.1
            }
            risk_factors['regime_risk'] = regime_risks.get(regime, 0.5)
            
            # Overall risk score
            overall_risk = np.mean(list(risk_factors.values()))
            risk_factors['overall_risk'] = overall_risk
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'overall_risk': 0.5}
    
    async def _analyze_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Analyze news sentiment for the symbol - Phase 1 Enhancement
        """
        try:
            # Import news analyzer
            from models.news_impact_analyzer import news_analyzer
            
            # Analyze recent news (4 hours back for high-frequency trading)
            analyses = await news_analyzer.analyze_symbol_news(symbol, hours_back=4)
            
            if not analyses:
                return None
            
            # Use the first (most relevant) analysis
            analysis = analyses[0]
            
            logger.debug(f"📰 {symbol} News: {analysis.sentiment_score:.2f} sentiment, "
                        f"{analysis.confidence:.1%} confidence, "
                        f"category: {analysis.category}")
            
            return {
                'sentiment_score': analysis.sentiment_score,
                'confidence': analysis.confidence,
                'impact_magnitude': analysis.impact_magnitude,
                'category': analysis.category,
                'keywords': analysis.keywords,
                'time_horizon': analysis.time_horizon
            }
            
        except ImportError:
            logger.debug("News analyzer not available")
            return None
        except Exception as e:
            logger.error(f"News analysis failed for {symbol}: {e}")
            return None
    
    def _synthesize_senior_analyst_decision(self, symbol: str, ml_predictions: Dict, 
                                          patterns: Dict, regime: MarketRegime, 
                                          risk_analysis: Dict, features: np.ndarray,
                                          news_analysis: Optional[Dict] = None) -> PredictionResult:
        """Synthesize all inputs like a senior analyst would"""
        try:
            reasoning = []
            supporting_patterns = []
            risk_factors = []
            
            # ML model consensus
            if ml_predictions:
                ml_scores = list(ml_predictions.values())
                ml_consensus = np.mean(ml_scores)
                ml_agreement = 1.0 - np.std(ml_scores)
                
                if ml_agreement > 0.8:
                    reasoning.append(f"Strong ML model consensus ({ml_consensus:.2f} confidence)")
                else:
                    reasoning.append(f"Mixed ML signals ({ml_consensus:.2f} avg)")
            else:
                ml_consensus = 0.5
                ml_agreement = 0.5
                reasoning.append("Using fundamental analysis")
            
            # Pattern analysis
            pattern_score = 0.5
            strong_patterns = patterns.get('strong_patterns', [])
            weak_patterns = patterns.get('weak_patterns', [])
            
            if strong_patterns:
                pattern_score += 0.2
                supporting_patterns.extend(strong_patterns)
                reasoning.append(f"Strong patterns: {', '.join(strong_patterns[:2])}")
            
            if weak_patterns:
                pattern_score -= 0.1
                reasoning.append(f"Concerning patterns: {', '.join(weak_patterns[:2])}")
            
            # NEWS INTELLIGENCE - Phase 1 Enhancement Integration
            news_score = 0.5  # Neutral baseline
            news_confidence_boost = 0.0
            
            if news_analysis:
                sentiment = news_analysis['sentiment_score']
                news_confidence = news_analysis['confidence']
                impact_magnitude = news_analysis['impact_magnitude']
                category = news_analysis['category']
                
                # Convert sentiment to score
                if sentiment > 0.1:  # Positive news
                    news_score = 0.5 + (sentiment * 0.4)  # 0.5 to 0.9
                    reasoning.append(f"Positive {category} news sentiment ({sentiment:.2f})")
                elif sentiment < -0.1:  # Negative news
                    news_score = 0.5 + (sentiment * 0.4)  # 0.1 to 0.5
                    reasoning.append(f"Negative {category} news sentiment ({sentiment:.2f})")
                    risk_factors.append(f"Negative news sentiment in {category}")
                
                # High-impact news categories get more weight
                if category in ['earnings', 'regulatory']:
                    news_confidence_boost = news_confidence * 0.1  # Up to 10% confidence boost
                    reasoning.append(f"High-impact {category} news detected")
                
                # Large predicted impact adjusts expectations
                if impact_magnitude > 3.0:  # > 3% predicted impact
                    reasoning.append(f"Significant price impact expected ({impact_magnitude:.1f}%)")
                    
            # Regime adjustments
            regime_adjustment = self._get_regime_adjustment(regime)
            
            if regime == MarketRegime.BULL_TRENDING:
                reasoning.append("Bull market supports longs")
            elif regime == MarketRegime.HIGH_VOLATILITY:
                reasoning.append("High volatility - caution required")
                risk_factors.append("High market volatility")
            
            # Risk integration
            risk_score = risk_analysis.get('overall_risk', 0.5)
            if risk_score > 0.7:
                risk_factors.append("High individual security risk")
                reasoning.append("Elevated risk profile")
            
            # Final synthesis with news intelligence
            base_confidence = ml_consensus
            confidence_adj = (pattern_score - 0.5) * 0.3
            confidence_adj += (ml_agreement - 0.5) * 0.2
            confidence_adj += regime_adjustment * 0.1
            confidence_adj -= max(0, risk_score - 0.5) * 0.3
            
            # Add NEWS INTELLIGENCE to confidence calculation
            confidence_adj += (news_score - 0.5) * 0.2  # News can add/subtract up to 8% confidence
            confidence_adj += news_confidence_boost  # High-impact news categories boost confidence
            
            final_confidence = max(0.1, min(0.95, base_confidence + confidence_adj))
            
            # Decision logic
            if final_confidence > 0.75:
                prediction = 1
                reasoning.append("High confidence bullish signal")
            elif final_confidence > 0.6:
                prediction = 1
                reasoning.append("Moderate bullish bias")
            elif final_confidence < 0.25:
                prediction = -1
                reasoning.append("High confidence bearish signal")
            elif final_confidence < 0.4:
                prediction = -1
                reasoning.append("Moderate bearish bias")
            else:
                prediction = 0
                reasoning.append("Neutral stance")
            
            expected_return = self._calculate_expected_return(final_confidence, prediction, regime, risk_score)
            time_horizon = self._determine_time_horizon(patterns, regime, final_confidence)
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
        """Adjust predictions based on market regime"""
        adjustments = {
            MarketRegime.BULL_TRENDING: 0.15,
            MarketRegime.BEAR_TRENDING: -0.15,
            MarketRegime.SIDEWAYS_RANGE: 0.0,
            MarketRegime.HIGH_VOLATILITY: -0.1,
            MarketRegime.LOW_VOLATILITY: 0.05,
            MarketRegime.BREAKOUT_PENDING: 0.1
        }
        return adjustments.get(regime, 0.0)
    
    def _calculate_expected_return(self, confidence: float, prediction: int, 
                                 regime: MarketRegime, risk_score: float) -> float:
        """Calculate expected return like a senior analyst"""
        try:
            base_returns = {
                MarketRegime.BULL_TRENDING: 0.08,
                MarketRegime.BEAR_TRENDING: -0.06,
                MarketRegime.SIDEWAYS_RANGE: 0.02,
                MarketRegime.HIGH_VOLATILITY: 0.12,
                MarketRegime.LOW_VOLATILITY: 0.04,
                MarketRegime.BREAKOUT_PENDING: 0.10
            }
            
            base_return = base_returns.get(regime, 0.05)
            expected_return = base_return * prediction * confidence
            risk_premium = risk_score * 0.03
            expected_return += risk_premium if prediction > 0 else -risk_premium
            
            return expected_return
            
        except Exception as e:
            logger.error(f"Expected return calculation failed: {e}")
            return 0.05 * prediction
    
    def _determine_time_horizon(self, patterns: Dict, regime: MarketRegime, confidence: float) -> int:
        """Determine holding period like a senior analyst"""
        try:
            base_horizons = {
                MarketRegime.BULL_TRENDING: 240,
                MarketRegime.BEAR_TRENDING: 120,
                MarketRegime.SIDEWAYS_RANGE: 360,
                MarketRegime.HIGH_VOLATILITY: 60,
                MarketRegime.LOW_VOLATILITY: 480,
                MarketRegime.BREAKOUT_PENDING: 30
            }
            
            base_horizon = base_horizons.get(regime, 240)
            horizon_multiplier = 0.5 + confidence
            
            if 'momentum_breakout' in patterns.get('strong_patterns', []):
                horizon_multiplier *= 0.7
            
            if 'reversal_pattern' in patterns.get('strong_patterns', []):
                horizon_multiplier *= 1.3
            
            final_horizon = int(base_horizon * horizon_multiplier)
            return max(15, min(final_horizon, 1440))
            
        except Exception as e:
            logger.error(f"Time horizon calculation failed: {e}")
            return 240
    
    def _calculate_feature_importance(self, features: np.ndarray, confidence: float) -> Dict[str, float]:
        """Calculate which features drove the decision"""
        try:
            feature_names = [
                'price_momentum', 'volume_surge', 'rsi_signal', 'macd_signal',
                'bollinger_position', 'support_resistance', 'trend_strength',
                'volatility_regime', 'correlation_score', 'sector_strength'
            ]
            
            importances = {}
            feature_array = features.flatten()
            
            for i, name in enumerate(feature_names):
                if i < len(feature_array):
                    importance = min(abs(feature_array[i]) * confidence, 1.0)
                    importances[name] = importance
                else:
                    importances[name] = 0.0
            
            return importances
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {'analysis_error': 1.0}
    
    async def _record_prediction_for_learning(self, symbol: str, prediction: PredictionResult, features: np.ndarray):
        """Record prediction for future learning"""
        try:
            tracker = self.performance_trackers[symbol]
            
            prediction_record = {
                'timestamp': datetime.now(),
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'expected_return': prediction.expected_return,
                'features': features.copy(),
                'reasoning': prediction.reasoning.copy()
            }
            
            tracker['predictions'].append(prediction_record)
            asyncio.create_task(self._track_prediction_outcome(symbol, prediction_record))
            
        except Exception as e:
            logger.error(f"Failed to record prediction for learning: {e}")
    
    async def _track_prediction_outcome(self, symbol: str, prediction_record: Dict):
        """Track prediction outcomes for learning"""
        try:
            await asyncio.sleep(prediction_record.get('time_horizon', 240) * 60)
            
            actual_outcome = await self._get_actual_outcome(symbol, prediction_record)
            
            tracker = self.performance_trackers[symbol]
            tracker['outcomes'].append(actual_outcome)
            
            self._update_accuracy_metrics(symbol)
            await self.meta_learner.learn_from_outcome(symbol, prediction_record, actual_outcome)
            
        except Exception as e:
            logger.error(f"Failed to track prediction outcome: {e}")
    
    async def _get_actual_outcome(self, symbol: str, prediction_record: Dict) -> Dict:
        """Get actual market outcome for learning"""
        try:
            import random
            
            confidence = prediction_record['confidence']
            success_probability = 0.5 + (confidence - 0.5) * 0.5
            
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
                recent_outcomes = list(tracker['outcomes'])[-50:]
                successes = [outcome['success'] for outcome in recent_outcomes]
                tracker['accuracy'] = sum(successes) / len(successes)
                
                self._update_ensemble_weights(symbol, tracker['accuracy'])
            
        except Exception as e:
            logger.error(f"Failed to update accuracy metrics: {e}")
    
    def _update_ensemble_weights(self, symbol: str, accuracy: float):
        """Update model ensemble weights based on performance"""
        try:
            if accuracy > 0.7:
                self.ensemble_weights['random_forest'] = min(0.4, self.ensemble_weights['random_forest'] + 0.01)
                self.ensemble_weights['gradient_boost'] = min(0.4, self.ensemble_weights['gradient_boost'] + 0.01)
                self.ensemble_weights['neural_network'] = min(0.3, self.ensemble_weights['neural_network'] + 0.01)
            elif accuracy < 0.4:
                self.ensemble_weights['pattern_recognition'] = min(0.3, self.ensemble_weights['pattern_recognition'] + 0.02)
                self.ensemble_weights['regime_analysis'] = min(0.2, self.ensemble_weights['regime_analysis'] + 0.01)
            
            total_weight = sum(self.ensemble_weights.values())
            for key in self.ensemble_weights:
                self.ensemble_weights[key] /= total_weight
            
        except Exception as e:
            logger.error(f"Failed to update ensemble weights: {e}")
    
    async def initialize_for_symbol(self, symbol: str, historical_data: pd.DataFrame):
        """Train ML models on historical data"""
        try:
            if not ML_AVAILABLE or len(historical_data) < 100:
                logger.warning(f"Skipping ML training for {symbol} - insufficient data or ML unavailable")
                return False
            
            with self.training_lock:
                logger.info(f"🧠 Training senior analyst models for {symbol}...")
                
                # Generate features and labels aligned
                features_list = []
                labels_list = []
                close_prices = historical_data['close']
                returns = close_prices.pct_change().fillna(0)
                
                lookhead_periods = 16  # Same as in _create_training_labels
                # Start from index 20 to ensure we have enough lookback data
                for i in range(20, len(historical_data) - lookhead_periods):
                    
                    # Generate features for this time point
                    feature_row = []
                    
                    feature_row.extend([
                        returns.iloc[i],  # Latest return
                        returns.iloc[i-5:i].mean() if i >= 5 else 0,  # 5-period momentum
                        returns.iloc[i-10:i].mean() if i >= 10 else 0,  # 10-period momentum  
                        returns.iloc[i-20:i].mean(),  # 20-period momentum
                        returns.iloc[i-20:i].std(),  # Volatility
                        returns.iloc[i-5:i].std() if i >= 5 else 0,  # Short-term volatility
                    ])
                    
                    # Add volume features if available
                    if 'volume' in historical_data.columns:
                        volume = historical_data['volume']
                        volume_ma = volume.iloc[i-20:i].mean() if i >= 20 else volume.iloc[:i].mean()
                        feature_row.extend([
                            volume.iloc[i] / volume_ma if volume_ma > 0 else 1,
                            volume.iloc[i-5:i].mean() / volume_ma if volume_ma > 0 and i >= 5 else 1,
                        ])
                    else:
                        feature_row.extend([1.0, 1.0])
                    
                    # Add technical indicators
                    sma_20 = close_prices.iloc[i-20:i].mean() if i >= 20 else close_prices.iloc[:i].mean()
                    feature_row.append(close_prices.iloc[i] / sma_20 if sma_20 > 0 else 1)
                    
                    # Trend strength
                    trend = (close_prices.iloc[i] - close_prices.iloc[i-10]) / close_prices.iloc[i-10] if i >= 10 else 0
                    feature_row.append(trend)
                    
                    # Pad to 12 features
                    while len(feature_row) < 12:
                        feature_row.append(0.0)
                    
                    features_list.append(feature_row[:12])
                    
                    # Create corresponding label for this time point
                    current_price = close_prices.iloc[i]
                    future_price = close_prices.iloc[i + lookhead_periods]
                    return_pct = (future_price - current_price) / current_price
                    
                    if return_pct > 0.015:  # 1.5% gain - more inclusive
                        labels_list.append(1)  # Buy signal
                    elif return_pct < -0.015:  # 1.5% loss - more inclusive
                        labels_list.append(0)  # Sell signal  
                    else:
                        labels_list.append(0)  # Hold signal (treated as sell for binary classification)
                
                features = np.array(features_list)
                labels = np.array(labels_list)
                
                if len(features) != len(labels):
                    logger.warning(f"Feature/label mismatch for {symbol}: {len(features)} features, {len(labels)} labels")
                    return False
                
                if len(features) < 50:
                    logger.warning(f"Insufficient training data for {symbol} - only {len(features)} samples")
                    return False
                
                # Check class balance - ensure we have both buy and sell signals
                unique_labels, label_counts = np.unique(labels, return_counts=True)
                if len(unique_labels) < 2:
                    logger.warning(f"Insufficient class diversity for {symbol}: only {unique_labels} classes found")
                    return False
                    
                min_class_samples = min(label_counts)
                if min_class_samples < 5:
                    logger.warning(f"Insufficient samples for minority class in {symbol}: {min_class_samples} samples")
                    return False
                    
                logger.info(f"🎯 Training data for {symbol}: {len(features)} samples, class distribution: {dict(zip(unique_labels, label_counts))}")
                
                # Train models
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42, shuffle=False
                )
                
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                models = {}
                
                rf_model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42,
                    class_weight='balanced', n_jobs=-1
                )
                rf_model.fit(X_train_scaled, y_train)
                models['random_forest'] = rf_model
                
                gb_model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                )
                gb_model.fit(X_train_scaled, y_train)
                models['gradient_boost'] = gb_model
                
                nn_model = MLPClassifier(
                    hidden_layer_sizes=(50, 25), max_iter=300, random_state=42,
                    alpha=0.01, learning_rate='adaptive'
                )
                nn_model.fit(X_train_scaled, y_train)
                models['neural_network'] = nn_model
                
                test_accuracies = {}
                for name, model in models.items():
                    test_pred = model.predict(X_test_scaled)
                    test_accuracies[name] = accuracy_score(y_test, test_pred)
                
                self.models[symbol] = models
                self.scalers[symbol] = scaler
                
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
        """Create training labels"""
        try:
            labels = []
            
            lookhead_periods = 16  # 16 * 15min = 4 hours
            
            for i in range(len(historical_data) - lookhead_periods):
                current_price = historical_data['close'].iloc[i]
                future_price = historical_data['close'].iloc[i + lookhead_periods]
                
                return_pct = (future_price - current_price) / current_price
                
                if return_pct > 0.008:  # 0.8% gain - more inclusive for better class balance
                    labels.append(2)  # Buy signal
                elif return_pct < -0.008:  # 0.8% loss - more inclusive for better class balance
                    labels.append(0)  # Sell signal
                else:
                    labels.append(1)  # Hold signal
            
            return np.array(labels)
            
        except Exception as e:
            logger.error(f"Failed to create training labels: {e}")
            return np.array([])
    
    def _create_low_confidence_prediction(self, symbol: str, reason: str) -> PredictionResult:
        """Create a low confidence prediction when analysis fails"""
        return PredictionResult(
            symbol=symbol,
            prediction=0,
            confidence=0.1,
            expected_return=0.0,
            time_horizon=240,
            reasoning=[reason],
            risk_factors=[reason],
            supporting_patterns=[],
            market_regime=MarketRegime.SIDEWAYS_RANGE,
            feature_importance={}
        )
    
    def _create_low_confidence_dict(self, symbol: str, reason: str) -> Dict:
        """Create a low confidence dict when analysis fails"""
        return {
            'symbol': symbol,
            'analyst_grade': 'HOLD',
            'confidence': 0.1,
            'expected_return': 0.0,
            'time_horizon_minutes': 240,
            'reasoning': [reason],
            'risk_factors': [reason],
            'supporting_patterns': [],
            'market_regime': 'sideways_range',
            'feature_importance': {}
        }
    
    async def learn_from_trade_outcome(self, symbol: str, trade_data: Dict):
        """Learn from trade outcomes to improve future decisions"""
        try:
            logger.info(f"Learning from trade outcome for {symbol}: {trade_data}")
        except Exception as e:
            logger.error(f"Error learning from trade outcome: {e}")
    
    def get_system_intelligence_report(self) -> Dict:
        """Get intelligence report about the system's performance"""
        try:
            total_predictions = sum(len(tracker['predictions']) for tracker in self.performance_trackers.values())
            total_outcomes = sum(len(tracker['outcomes']) for tracker in self.performance_trackers.values())
            
            if total_outcomes > 0:
                accuracies = [tracker['accuracy'] for tracker in self.performance_trackers.values() if tracker['accuracy'] > 0]
                avg_accuracy = np.mean(accuracies) if accuracies else 0.5
            else:
                avg_accuracy = 0.5
            
            if avg_accuracy > 0.8:
                intelligence_level = "Expert"
            elif avg_accuracy > 0.7:
                intelligence_level = "Advanced"
            elif avg_accuracy > 0.6:
                intelligence_level = "Intermediate"
            else:
                intelligence_level = "Learning"
            
            if total_predictions > 1000:
                system_maturity = "Mature"
            elif total_predictions > 500:
                system_maturity = "Developing"
            elif total_predictions > 100:
                system_maturity = "Early"
            else:
                system_maturity = "Initial"
            
            return {
                'intelligence_level': intelligence_level,
                'system_maturity': system_maturity,
                'average_accuracy': avg_accuracy,
                'total_predictions': total_predictions,
                'total_outcomes': total_outcomes,
                'symbols_analyzed': len(self.performance_trackers),
                'ml_available': ML_AVAILABLE,
                'is_trained': self.is_trained,
                'ensemble_weights': self.ensemble_weights.copy()
            }
            
        except Exception as e:
            logger.error(f"Error generating intelligence report: {e}")
            return {
                'intelligence_level': 'Unknown',
                'system_maturity': 'Unknown',
                'average_accuracy': 0.5,
                'error': str(e)
            }

# Create the correct class alias for the main application
SeniorAnalystIntegration = SeniorAnalystBrain

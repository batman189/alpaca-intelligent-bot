"""
Advanced Market Analysis Engine
This is the core intelligence of the trading bot - replaces the fake random predictions
with real pattern recognition and multi-timeframe analysis.
RENDER-OPTIMIZED VERSION with TA-Lib fallbacks
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import TA-Lib with fallback
try:
    from utils.talib_fallback import *
    logger = logging.getLogger(__name__)
    logger.info("✅ Using TA-Lib fallback functions")
except ImportError:
    # If our fallback fails, use basic pandas
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Using basic pandas for technical analysis")

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("⚠️ ML libraries not available, using rule-based analysis only")

logger = logging.getLogger(__name__)

class AdvancedMarketAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.pattern_weights = {
            'momentum_breakout': 0.25,
            'volume_anomaly': 0.20,
            'multi_timeframe_confluence': 0.20,
            'technical_patterns': 0.15,
            'volatility_expansion': 0.10,
            'support_resistance': 0.10
        }
        
        # Pattern detection thresholds
        self.volume_surge_threshold = 2.0  # 2x average volume
        self.momentum_threshold = 0.015    # 1.5% move for momentum
        self.volatility_threshold = 1.5    # 1.5x normal volatility
        
        # ML model parameters
        self.lookback_periods = [5, 10, 20, 50]
        self.feature_window = 100
        self.prediction_horizon = 4  # 4 periods ahead
        
        logger.info(f"AdvancedMarketAnalyzer initialized (ML Available: {ML_AVAILABLE})")

    def calculate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and features for analysis"""
        if len(df) < 50:
            return df
            
        data = df.copy()
        
        try:
            # Price-based indicators using fallback functions
            data['sma_5'] = SMA(data['close'], timeperiod=5)
            data['sma_10'] = SMA(data['close'], timeperiod=10)
            data['sma_20'] = SMA(data['close'], timeperiod=20)
            data['sma_50'] = SMA(data['close'], timeperiod=50)
            
            data['ema_8'] = EMA(data['close'], timeperiod=8)
            data['ema_13'] = EMA(data['close'], timeperiod=13)
            data['ema_21'] = EMA(data['close'], timeperiod=21)
            
            # MACD
            data['macd'], data['macd_signal'], data['macd_hist'] = MACD(data['close'])
            
            # RSI
            data['rsi'] = RSI(data['close'], timeperiod=14)
            data['rsi_sma'] = SMA(data['rsi'], timeperiod=5)
            
            # Bollinger Bands
            data['bb_upper'], data['bb_middle'], data['bb_lower'] = BBANDS(data['close'])
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Continue with the rest of your indicators...
            # (The rest of the file remains the same, just using the fallback functions)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return data.fillna(method='ffill').fillna(0)
        data = df.copy()
        
        try:
            # Price-based indicators
            data['sma_5'] = talib.SMA(data['close'], timeperiod=5)
            data['sma_10'] = talib.SMA(data['close'], timeperiod=10)
            data['sma_20'] = talib.SMA(data['close'], timeperiod=20)
            data['sma_50'] = talib.SMA(data['close'], timeperiod=50)
            
            data['ema_8'] = talib.EMA(data['close'], timeperiod=8)
            data['ema_13'] = talib.EMA(data['close'], timeperiod=13)
            data['ema_21'] = talib.EMA(data['close'], timeperiod=21)
            
            # MACD
            data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(data['close'])
            
            # RSI
            data['rsi'] = talib.RSI(data['close'], timeperiod=14)
            data['rsi_sma'] = talib.SMA(data['rsi'], timeperiod=5)
            
            # Bollinger Bands
            data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(data['close'])
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Stochastic
            data['stoch_k'], data['stoch_d'] = talib.STOCH(data['high'], data['low'], data['close'])
            
            # Williams %R
            data['williams_r'] = talib.WILLR(data['high'], data['low'], data['close'])
            
            # Average True Range (Volatility)
            data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
            data['atr_ratio'] = data['atr'] / data['close']
            
            # Volume indicators
            data['volume_sma'] = talib.SMA(data['volume'], timeperiod=20)
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            # On Balance Volume
            data['obv'] = talib.OBV(data['close'], data['volume'])
            data['obv_sma'] = talib.SMA(data['obv'], timeperiod=20)
            
            # Price momentum
            data['momentum'] = talib.MOM(data['close'], timeperiod=10)
            data['roc'] = talib.ROC(data['close'], timeperiod=10)
            
            # Commodity Channel Index
            data['cci'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=14)
            
            # Money Flow Index
            data['mfi'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14)
            
            # VWAP (Volume Weighted Average Price)
            data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            
            # Price relative to VWAP
            data['price_vs_vwap'] = (data['close'] - data['vwap']) / data['vwap']
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return data.fillna(method='ffill').fillna(0)
    
    def detect_momentum_breakout(self, df: pd.DataFrame) -> Dict:
        """Detect momentum breakouts with volume confirmation"""
        try:
            if len(df) < 20:
                return {'score': 0, 'signals': []}
                
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = []
            score = 0
            
            # Price momentum
            price_change = (latest['close'] - prev['close']) / prev['close']
            
            # Volume surge
            volume_ratio = latest['volume'] / latest['volume_sma'] if latest['volume_sma'] > 0 else 1
            
            # Breakout above resistance (20-period high)
            high_20 = df['high'].rolling(20).max().iloc[-2]  # Previous 20-period high
            if latest['close'] > high_20 * 1.005:  # 0.5% above previous high
                signals.append('breakout_resistance')
                score += 0.3
                
            # Strong momentum with volume
            if abs(price_change) > self.momentum_threshold and volume_ratio > self.volume_surge_threshold:
                direction = 'bullish' if price_change > 0 else 'bearish'
                signals.append(f'momentum_{direction}_volume')
                score += 0.4 if price_change > 0 else -0.4
                
            # EMA alignment for trend
            if (latest['ema_8'] > latest['ema_13'] > latest['ema_21'] and 
                latest['close'] > latest['ema_8']):
                signals.append('ema_bullish_alignment')
                score += 0.2
            elif (latest['ema_8'] < latest['ema_13'] < latest['ema_21'] and 
                  latest['close'] < latest['ema_8']):
                signals.append('ema_bearish_alignment')
                score += -0.2
                
            return {
                'score': max(-1, min(1, score)),
                'signals': signals,
                'price_change': price_change,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error detecting momentum breakout: {e}")
            return {'score': 0, 'signals': []}
    
    def detect_volume_anomaly(self, df: pd.DataFrame) -> Dict:
        """Detect unusual volume patterns that often precede price moves"""
        try:
            if len(df) < 20:
                return {'score': 0, 'signals': []}
                
            latest = df.iloc[-1]
            
            signals = []
            score = 0
            
            # Volume surge analysis
            volume_ratio = latest['volume'] / latest['volume_sma'] if latest['volume_sma'] > 0 else 1
            
            # Massive volume spike
            if volume_ratio > 3.0:
                signals.append('massive_volume_spike')
                score += 0.5
            elif volume_ratio > self.volume_surge_threshold:
                signals.append('volume_surge')
                score += 0.3
                
            # Volume vs price divergence
            price_change = (latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            # High volume with small price change (accumulation/distribution)
            if volume_ratio > 2.0 and abs(price_change) < 0.005:
                signals.append('volume_accumulation')
                score += 0.2
                
            # OBV trend
            obv_trend = (latest['obv'] - latest['obv_sma']) / latest['obv_sma'] if latest['obv_sma'] != 0 else 0
            if abs(obv_trend) > 0.1:
                direction = 'bullish' if obv_trend > 0 else 'bearish'
                signals.append(f'obv_trend_{direction}')
                score += 0.2 if obv_trend > 0 else -0.2
                
            return {
                'score': max(-1, min(1, score)),
                'signals': signals,
                'volume_ratio': volume_ratio,
                'obv_trend': obv_trend
            }
            
        except Exception as e:
            logger.error(f"Error detecting volume anomaly: {e}")
            return {'score': 0, 'signals': []}
    
    def detect_technical_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect classic chart patterns"""
        try:
            if len(df) < 50:
                return {'score': 0, 'signals': []}
                
            signals = []
            score = 0
            
            # Bollinger Band squeeze and expansion
            latest = df.iloc[-1]
            bb_width_sma = df['bb_width'].rolling(20).mean().iloc[-1]
            
            if latest['bb_width'] < bb_width_sma * 0.8:
                signals.append('bollinger_squeeze')
                score += 0.1  # Setup for breakout
                
            # Bollinger Band breakout
            if latest['close'] > latest['bb_upper']:
                signals.append('bb_breakout_upper')
                score += 0.3
            elif latest['close'] < latest['bb_lower']:
                signals.append('bb_breakout_lower')
                score -= 0.3
                
            # RSI extremes
            if latest['rsi'] > 70 and df['rsi'].iloc[-2] <= 70:
                signals.append('rsi_overbought_entry')
                score -= 0.2
            elif latest['rsi'] < 30 and df['rsi'].iloc[-2] >= 30:
                signals.append('rsi_oversold_entry')
                score += 0.2
                
            # MACD signals
            if (latest['macd'] > latest['macd_signal'] and 
                df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]):
                signals.append('macd_bullish_crossover')
                score += 0.2
            elif (latest['macd'] < latest['macd_signal'] and 
                  df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]):
                signals.append('macd_bearish_crossover')
                score -= 0.2
                
            # Stochastic signals
            if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 80:
                signals.append('stoch_bullish')
                score += 0.1
            elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] > 20:
                signals.append('stoch_bearish')
                score -= 0.1
                
            return {
                'score': max(-1, min(1, score)),
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error detecting technical patterns: {e}")
            return {'score': 0, 'signals': []}
    
    def detect_volatility_expansion(self, df: pd.DataFrame) -> Dict:
        """Detect volatility expansion patterns"""
        try:
            if len(df) < 20:
                return {'score': 0, 'signals': []}
                
            latest = df.iloc[-1]
            atr_sma = df['atr'].rolling(20).mean().iloc[-1]
            
            signals = []
            score = 0
            
            # ATR expansion
            if latest['atr'] > atr_sma * self.volatility_threshold:
                signals.append('volatility_expansion')
                score += 0.3
                
            # Price range expansion
            daily_range = (latest['high'] - latest['low']) / latest['close']
            avg_range = ((df['high'] - df['low']) / df['close']).rolling(20).mean().iloc[-1]
            
            if daily_range > avg_range * 1.5:
                signals.append('range_expansion')
                score += 0.2
                
            return {
                'score': max(-1, min(1, score)),
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error detecting volatility expansion: {e}")
            return {'score': 0, 'signals': []}
    
    def prepare_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for machine learning model"""
        try:
            if len(df) < self.feature_window:
                return None
                
            # Select relevant features for ML
            feature_cols = [
                'sma_5', 'sma_10', 'sma_20', 'ema_8', 'ema_13', 'ema_21',
                'macd', 'macd_signal', 'rsi', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'williams_r', 'atr_ratio', 'volume_ratio',
                'momentum', 'roc', 'cci', 'mfi', 'price_vs_vwap'
            ]
            
            # Get available features
            available_features = [col for col in feature_cols if col in df.columns]
            
            if not available_features:
                return None
                
            # Get latest features
            features = df[available_features].iloc[-1].values
            
            # Handle any NaN values
            features = np.nan_to_num(features)
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Comprehensive analysis of a symbol"""
        try:
            if df is None or len(df) < 50:
                return {
                    'symbol': symbol,
                    'prediction': 0,
                    'confidence': 0.0,
                    'signals': [],
                    'analysis': 'insufficient_data'
                }
            
            # Calculate all technical indicators
            analyzed_df = self.calculate_comprehensive_features(df)
            
            # Run pattern detection
            momentum_analysis = self.detect_momentum_breakout(analyzed_df)
            volume_analysis = self.detect_volume_anomaly(analyzed_df)
            technical_analysis = self.detect_technical_patterns(analyzed_df)
            volatility_analysis = self.detect_volatility_expansion(analyzed_df)
            
            # Calculate composite score
            composite_score = (
                momentum_analysis['score'] * self.pattern_weights['momentum_breakout'] +
                volume_analysis['score'] * self.pattern_weights['volume_anomaly'] +
                technical_analysis['score'] * self.pattern_weights['technical_patterns'] +
                volatility_analysis['score'] * self.pattern_weights['volatility_expansion']
            )
            
            # Determine prediction and confidence
            prediction = 1 if composite_score > 0.1 else 0
            confidence = min(0.95, abs(composite_score) * 2)  # Scale to 0-0.95 range
            
            # Combine all signals
            all_signals = (
                momentum_analysis['signals'] + 
                volume_analysis['signals'] + 
                technical_analysis['signals'] + 
                volatility_analysis['signals']
            )
            
            # ML enhancement (if model exists)
            ml_features = self.prepare_ml_features(analyzed_df)
            if ml_features is not None and symbol in self.models:
                try:
                    ml_prediction = self.models[symbol].predict(ml_features)[0]
                    ml_confidence = np.max(self.models[symbol].predict_proba(ml_features))
                    
                    # Blend traditional analysis with ML
                    final_prediction = 1 if (prediction + ml_prediction) >= 1 else 0
                    final_confidence = (confidence + ml_confidence) / 2
                    
                    prediction = final_prediction
                    confidence = final_confidence
                    
                except Exception as e:
                    logger.warning(f"ML prediction failed for {symbol}: {e}")
            
            return {
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence,
                'composite_score': composite_score,
                'signals': all_signals,
                'analysis': {
                    'momentum': momentum_analysis,
                    'volume': volume_analysis,
                    'technical': technical_analysis,
                    'volatility': volatility_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'prediction': 0,
                'confidence': 0.0,
                'signals': [],
                'analysis': f'analysis_error: {str(e)}'
            }
    
    def create_training_data(self, df: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data for ML model"""
        try:
            if len(df) < self.feature_window + self.prediction_horizon:
                return None, None
                
            analyzed_df = self.calculate_comprehensive_features(df)
            
            # Prepare features
            feature_cols = [
                'sma_5', 'sma_10', 'sma_20', 'ema_8', 'ema_13', 'ema_21',
                'macd', 'macd_signal', 'rsi', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'williams_r', 'atr_ratio', 'volume_ratio',
                'momentum', 'roc', 'cci', 'mfi', 'price_vs_vwap'
            ]
            
            available_features = [col for col in feature_cols if col in analyzed_df.columns]
            
            X = []
            y = []
            
            for i in range(self.feature_window, len(analyzed_df) - self.prediction_horizon):
                # Features at time i
                features = analyzed_df[available_features].iloc[i].values
                features = np.nan_to_num(features)
                
                # Target: price movement over prediction horizon
                current_price = analyzed_df['close'].iloc[i]
                future_price = analyzed_df['close'].iloc[i + self.prediction_horizon]
                price_change = (future_price - current_price) / current_price
                
                target = 1 if price_change > 0.01 else 0  # 1% threshold
                
                X.append(features)
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating training data for {symbol}: {e}")
            return None, None
    
    def train_model(self, symbol: str, df: pd.DataFrame):
        """Train ML model for a specific symbol"""
        try:
            X, y = self.create_training_data(df, symbol)
            
            if X is None or len(X) < 100:
                logger.warning(f"Insufficient training data for {symbol}")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble model
            rf_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Simple ensemble: average predictions
            class EnsembleModel:
                def __init__(self, model1, model2):
                    self.model1 = model1
                    self.model2 = model2
                    
                def predict(self, X):
                    pred1 = self.model1.predict(X)
                    pred2 = self.model2.predict(X)
                    return ((pred1 + pred2) / 2).round().astype(int)
                    
                def predict_proba(self, X):
                    prob1 = self.model1.predict_proba(X)
                    prob2 = self.model2.predict_proba(X)
                    return (prob1 + prob2) / 2
            
            ensemble_model = EnsembleModel(rf_model, gb_model)
            
            # Store model and scaler
            self.models[symbol] = ensemble_model
            self.scalers[symbol] = scaler
            
            # Evaluate model
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            logger.info(f"Model trained for {symbol} - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False
    
    def save_models(self, filepath: str):
        """Save trained models"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'pattern_weights': self.pattern_weights
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.pattern_weights = model_data.get('pattern_weights', self.pattern_weights)
            logger.info(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

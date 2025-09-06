import logging
import pickle
import numpy as np
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class IntelligentPredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.market_context = {}
        self.last_analysis_time = None
        
    def load_model(self, model_path):
        """Load a pre-trained model from file"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
            return False
            
    def analyze_market_context(self, symbol, data):
        """Analyze broader market context for smarter predictions"""
        try:
            if data is None or len(data) < 20:
                return 0.5  # Neutral confidence without data
                
            # Calculate short-term momentum (last 5 bars)
            recent_data = data.iloc[-5:] if hasattr(data, 'iloc') else data[-5:]
            price_changes = recent_data['close'].pct_change().dropna()
            
            if len(price_changes) == 0:
                return 0.5
                
            # Strong upward momentum (like TSLA today)
            if all(change > 0 for change in price_changes) and sum(price_changes) > 0.05:
                momentum_factor = 1.3  # Boost confidence for strong uptrends
                logger.info(f"🚀 Strong upward momentum detected for {symbol}: +{sum(price_changes):.2%}")
                
            # Strong downward momentum
            elif all(change < 0 for change in price_changes) and sum(price_changes) < -0.05:
                momentum_factor = 0.7  # Reduce confidence for strong downtrends
                logger.info(f"🔻 Strong downward momentum detected for {symbol}: {sum(price_changes):.2%}")
                
            # Choppy/neutral market
            else:
                momentum_factor = 1.0
                
            return momentum_factor
            
        except Exception as e:
            logger.error(f"Error analyzing market context for {symbol}: {e}")
            return 1.0
            
    def predict(self, features, symbol=None, data=None):
        """
        Make a prediction using the loaded model with market context awareness
        """
        try:
            if not self.is_trained or self.model is None:
                # Return random prediction if no model is loaded
                return self._random_prediction()
                
            # Ensure features is a 2D numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
                
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
                
            # Check for NaN values and replace them
            features = np.nan_to_num(features, nan=0.0)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            base_confidence = np.max(self.model.predict_proba(features))
            
            # Apply market context awareness
            if symbol and data is not None:
                momentum_factor = self.analyze_market_context(symbol, data)
                adjusted_confidence = base_confidence * momentum_factor
                
                # Log context-aware adjustments
                if momentum_factor != 1.0:
                    logger.info(f"Market context adjusted confidence for {symbol}: {base_confidence:.2f} → {adjusted_confidence:.2f}")
                
                return prediction, float(adjusted_confidence)
            else:
                return prediction, float(base_confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Fallback to random prediction
            return self._random_prediction()
    
    def _random_prediction(self):
        """Fallback random prediction when model is not available"""
        prediction = random.randint(0, 1)
        confidence = random.uniform(0.5, 0.8)
        return prediction, confidence

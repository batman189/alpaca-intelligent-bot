import logging
import pickle
import numpy as np
import random
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)

class IntelligentPredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        
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
            self.model = GradientBoostingClassifier()
            return False
            
    def predict(self, features):
        """
        Make a prediction using the loaded model.
        features: numpy array of shape (1, n_features)
        """
        try:
            if not self.is_trained:
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
            confidence = np.max(self.model.predict_proba(features))
            
            return prediction, float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Fallback to random prediction
            return self._random_prediction()
    
    def _random_prediction(self):
        """Fallback random prediction when model is not available"""
        prediction = random.randint(0, 1)
        confidence = random.uniform(0.5, 0.8)
        return prediction, confidence

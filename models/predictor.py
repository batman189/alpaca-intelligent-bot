import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class IntelligentPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            self.model.fit(X_scaled, y_train)
            self.feature_columns = X_train.columns.tolist()
            
            logger.info("Model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
            
    def predict(self, features: pd.DataFrame) -> Tuple[float, float]:
        if self.model is None or features is None:
            return 0.0, 0.0
            
        try:
            current_features = features.copy()
            missing_cols = set(self.feature_columns) - set(current_features.columns)
            for col in missing_cols:
                current_features[col] = 0
                
            current_features = current_features[self.feature_columns]
            
            X_scaled = self.scaler.transform(current_features)
            prediction = self.model.predict_proba(X_scaled)
            
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            
            return float(predicted_class), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0, 0.0
            
    def save_model(self, filepath: str):
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    def load_model(self, filepath: str):
        try:
            saved_data = joblib.load(filepath)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_columns = saved_data['feature_columns']
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def create_training_labels(self, df: pd.DataFrame, horizon: int = 4, threshold: float = 0.02) -> pd.Series:
        if df is None or len(df) < horizon:
            return None
            
        future_prices = df['close'].shift(-horizon)
        future_returns = (future_prices / df['close'] - 1).fillna(0)
        
        labels = (future_returns > threshold).astype(int)
        return labels

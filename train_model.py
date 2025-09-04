import pandas as pd
import logging
from data.data_client import DataClient
from features.feature_engineer import FeatureEngineer
from models.predictor import IntelligentPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting model training...")
    
    # Initialize components
    data_client = DataClient()
    feature_engineer = FeatureEngineer()
    predictor = IntelligentPredictor()
    
    # Get historical data for training
    logger.info("Fetching historical data for training...")
    data = data_client.get_historical_bars('SPY', '15Min', limit=1000)
    
    if data is None or len(data) < 100:
        logger.error("Not enough data for training")
        return False
    
    # Engineer features
    logger.info("Engineering features...")
    engineered_data = feature_engineer.calculate_technical_indicators(data)
    
    # Create training labels
    logger.info("Creating training labels...")
    labels = predictor.create_training_labels(engineered_data, horizon=4, threshold=0.02)
    
    if labels is None:
        logger.error("Failed to create labels")
        return False
    
    # Prepare features for training (drop rows with NaN)
    features = engineered_data.dropna()
    labels = labels[features.index]  # Align labels with features
    
    if len(features) < 50:
        logger.error("Not enough clean data for training")
        return False
    
    # Train the model
    logger.info("Training the model...")
    success = predictor.train_model(features, labels)
    
    if success:
        logger.info("Saving trained model...")
        predictor.save_model('models/trained_model.pkl')
        logger.info("Model training completed successfully!")
        return True
    else:
        logger.error("Model training failed")
        return False

if __name__ == "__main__":
    train_model()

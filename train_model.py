import pandas as pd
import logging
import os
from data.data_client import DataClient
from features.feature_engineer import FeatureEngineer
from models.predictor import IntelligentPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)

logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting model training process...")
    
    try:
        # Initialize components
        data_client = DataClient()
        feature_engineer = FeatureEngineer()
        predictor = IntelligentPredictor()
        
        # Get historical data for training
        logger.info("Fetching historical data for training...")
        data = data_client.get_historical_bars('SPY', '15Min', limit=1000)
        
        if data is None:
            logger.error("Failed to fetch data for training")
            return False
            
        if len(data) < 100:
            logger.error(f"Not enough data for training. Got {len(data)} bars, need at least 100.")
            return False
            
        logger.info(f"Successfully retrieved {len(data)} bars for training")
        
        # Engineer features
        logger.info("Engineering features...")
        engineered_data = feature_engineer.calculate_technical_indicators(data)
        
        # Create training labels
        logger.info("Creating training labels...")
        labels = predictor.create_training_labels(engineered_data, horizon=4, threshold=0.02)
        
        if labels is None:
            logger.error("Failed to create training labels")
            return False
            
        # Prepare features for training (drop rows with NaN)
        features = engineered_data.dropna()
        labels = labels[features.index]  # Align labels with features
        
        logger.info(f"Clean dataset: {len(features)} samples with {len(features.columns)} features")
        
        if len(features) < 50:
            logger.error("Not enough clean data for training. Need at least 50 samples.")
            return False
        
        # Train the model
        logger.info("Training the machine learning model...")
        success = predictor.train_model(features, labels)
        
        if not success:
            logger.error("Model training failed")
            return False
            
        # Save the trained model
        logger.info("Saving trained model...")
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        predictor.save_model('models/trained_model.pkl')
        logger.info("Model training completed successfully!")
        logger.info(f"Model saved to: models/trained_model.pkl")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        print("Model training completed successfully!")
    else:
        print("Model training failed. Check logs for details.")
        exit(1)

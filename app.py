from flask import Flask
import threading
import logging
import time
import pandas as pd
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)

logger = logging.getLogger(__name__)

# Create a simple Flask app for health checks
app = Flask(__name__)

@app.route('/')
def health_check():
    return {'status': 'healthy', 'service': 'trading-bot'}

@app.route('/health')
def health():
    return {'status': 'ok', 'timestamp': datetime.now().isoformat()}

def run_flask_app():
    app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)

# Direct environment variable access for settings
settings = type('Settings', (), {
    'WATCHLIST': os.getenv('WATCHLIST', 'SPY,QQQ,IWM,TSLA,NVDA,AAPL,MSFT,GOOGL').split(','),
    'RESAMPLE_INTERVAL': os.getenv('RESAMPLE_INTERVAL', '15Min'),
    'LOOKBACK_WINDOW': int(os.getenv('LOOKBACK_WINDOW', '100')),
    'PREDICTION_THRESHOLD': float(os.getenv('PREDICTION_THRESHOLD', '0.65')),
    'MIN_CONFIDENCE': float(os.getenv('MIN_CONFIDENCE', '0.6')),
    'RISK_PER_TRADE': float(os.getenv('RISK_PER_TRADE', '0.02')),
    'MAX_PORTFOLIO_RISK': float(os.getenv('MAX_PORTFOLIO_RISK', '0.1')),
    'ENABLE_TRADING': os.getenv('ENABLE_TRADING', 'false').lower() == 'true',
    'PAPER_TRADING': os.getenv('PAPER_TRADING', 'true').lower() == 'true'
})()

# Import our modules
try:
    from data.data_client import DataClient
    from features.feature_engineer import FeatureEngineer
    from models.predictor import IntelligentPredictor
    from trading.portfolio_manager import PortfolioManager
    from trading.execution_client import ExecutionClient
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Create simple fallback classes
    class DataClient:
        def get_multiple_historical_bars(self, *args, **kwargs):
            return {'SPY': None, 'AAPL': None}
        def get_account_info(self):
            return {'equity': 10000, 'cash': 5000}
    class FeatureEngineer:
        def calculate_technical_indicators(self, df):
            return df
        def prepare_features_for_prediction(self, df):
            return df
    class IntelligentPredictor:
        def load_model(self, path):
            return False
        def predict(self, features):
            return 0, 0.5
    class PortfolioManager:
        def should_enter_trade(self, *args):
            return False
    class ExecutionClient:
        def get_current_positions(self):
            return {}
    
    DataClient = DataClient()
    FeatureEngineer = FeatureEngineer()
    IntelligentPredictor = IntelligentPredictor()
    PortfolioManager = PortfolioManager()
    ExecutionClient = ExecutionClient()

def train_model_if_needed():
    """Train model if it doesn't exist"""
    model_path = 'models/trained_model.pkl'
    
    if not os.path.exists(model_path):
        logger.info("No trained model found. Starting model training...")
        try:
            from train_model import train_model
            success = train_model()
            if success:
                logger.info("Model training completed successfully!")
                return True
            else:
                logger.warning("Model training failed")
                return False
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    return True

class IntelligentTradingBot:
    def __init__(self):
        # Start Flask server in a separate thread
        logger.info("Starting health check server...")
        self.flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        self.flask_thread.start()
        
        # Train model if needed
        self.model_trained = train_model_if_needed()
        
        self.data_client = DataClient()
        self.feature_engineer = FeatureEngineer()
        self.predictor = IntelligentPredictor()
        self.portfolio_manager = PortfolioManager()
        self.execution_client = ExecutionClient()
        
        # Load pre-trained model if available
        if self.model_trained:
            logger.info("Loading trained model...")
            try:
                if hasattr(self.predictor, 'load_model'):
                    success = self.predictor.load_model('models/trained_model.pkl')
                    if not success:
                        logger.warning("Failed to load trained model")
                else:
                    logger.warning("Predictor has no load_model method")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning("No model available. Running in analysis-only mode.")
        
    def run_analysis_cycle(self):
        """Run one complete analysis and trading cycle"""
        logger.info("Starting analysis cycle")
        
        try:
            # 1. Get market data
            market_data = self.data_client.get_multiple_historical_bars(
                settings.WATCHLIST, 
                settings.RESAMPLE_INTERVAL, 
                settings.LOOKBACK_WINDOW
            )
            
            # 2. Get account information
            account_info = self.data_client.get_account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return
                
            # 3. Analyze each symbol
            for symbol in settings.WATCHLIST:
                self.analyze_symbol(symbol, market_data.get(symbol), account_info)
                
            # 4. Manage existing positions
            self.manage_existing_positions(market_data)
            
            logger.info("Analysis cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            
    def analyze_symbol(self, symbol: str, data: pd.DataFrame, account_info: dict):
        """Analyze a single symbol and make trading decisions"""
        if data is None:
            logger.warning(f"No data for {symbol}")
            return
            
        if len(data) < 20:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
            return
            
        try:
            # 1. Feature engineering
            engineered_data = self.feature_engineer.calculate_technical_indicators(data)
            
            # 2. Prepare features for prediction
            prediction_features = self.feature_engineer.prepare_features_for_prediction(engineered_data)
            
            if prediction_features is None:
                return
                
            # 3. Get prediction from model (if model is loaded)
            prediction, confidence = 0, 0.0
            if hasattr(self.predictor, 'predict'):
                prediction, confidence = self.predictor.predict(prediction_features)
                logger.info(f"{symbol} - Prediction: {prediction}, Confidence: {confidence:.2f}")
            else:
                logger.info(f"{symbol} - Analysis complete (no model predictions)")
                
            # 4. Make trading decision (only if model is loaded and confident)
            if (hasattr(self.predictor, 'predict') and 
                prediction == 1 and 
                confidence >= settings.PREDICTION_THRESHOLD):
                self.execute_trade(symbol, engineered_data, account_info, confidence)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
    def execute_trade(self, symbol: str, data: pd.DataFrame, account_info: dict, confidence: float):
        """Execute a trade based on analysis"""
        if not settings.ENABLE_TRADING:
            logger.info(f"Trading disabled. Would trade {symbol} with confidence {confidence:.2f}")
            return
            
        try:
            current_price = data['close'].iloc[-1] if 'close' in data else 100
            volatility = data['volatility'].iloc[-1] if 'volatility' in data else 0.02
            
            # 1. Check if we should enter trade
            current_positions = self.execution_client.get_current_positions()
            if hasattr(self.portfolio_manager, 'should_enter_trade'):
                if not self.portfolio_manager.should_enter_trade(symbol, confidence, current_positions):
                    return
                
            # 2. Determine position size
            stop_loss = current_price * 0.95  # Simple 5% stop loss
            position_size = max(1, int((account_info.get('equity', 10000) * settings.RISK_PER_TRADE) / current_price))
            
            # 3. Place option order
            if hasattr(self.execution_client, 'place_option_order'):
                success = self.execution_client.place_option_order(
                    symbol=symbol,
                    quantity=position_size,
                    order_type='call',
                    strike=current_price * 1.05,
                    expiration='2025-01-17'
                )
                
                if success:
                    logger.info(f"Trade executed: {position_size} calls on {symbol}")
            else:
                logger.info(f"Would execute trade: {position_size} calls on {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            
    def manage_existing_positions(self, market_data: dict):
        """Manage risk on existing positions"""
        try:
            current_positions = self.execution_client.get_current_positions()
            if hasattr(self.portfolio_manager, 'manage_risk'):
                exit_signals = self.portfolio_manager.manage_risk(current_positions, market_data)
                
                for signal in exit_signals:
                    if hasattr(self.execution_client, 'close_position'):
                        self.execution_client.close_position(signal['symbol'])
                        logger.info(f"Exit signal: Closed {signal['symbol']} due to {signal['reason']}")
            else:
                logger.info("Risk management not available in current mode")
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            
    def run(self):
        """Main bot loop"""
        logger.info("Starting Intelligent Trading Bot")
        
        while True:
            try:
                self.run_analysis_cycle()
                
                # Sleep until next cycle
                logger.info("Sleeping for 15 minutes until next analysis cycle...")
                time.sleep(900)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = IntelligentTradingBot()
    bot.run()

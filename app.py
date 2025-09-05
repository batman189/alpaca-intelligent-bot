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

# DIRECT ENVIRONMENT VARIABLE ACCESS
WATCHLIST = os.getenv('WATCHLIST', 'SPY,QQQ,AAPL,MSFT,NVDA,TSLA').split(',')
RESAMPLE_INTERVAL = os.getenv('RESAMPLE_INTERVAL', '15Min')
LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', '100'))
PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', '0.65'))
MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.6'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', '0.1'))
ENABLE_TRADING = os.getenv('ENABLE_TRADING', 'false').lower() == 'true'
PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
TRADING_STRATEGY = os.getenv('TRADING_STRATEGY', 'options')
OPTIONS_EXPIRATION = os.getenv('OPTIONS_EXPIRATION', '2026-01-16')
MAX_OPTIONS_POSITIONS = int(os.getenv('MAX_OPTIONS_POSITIONS', '5'))

# Import our modules with simple error handling
try:
    from data.data_client import DataClient
except ImportError as e:
    logger.error(f"DataClient import error: {e}")
    class DataClient:
        def get_multiple_historical_bars(self, symbols, timeframe='15Min', limit=100):
            return {symbol: None for symbol in symbols}
        def get_account_info(self):
            return {'equity': 10000, 'cash': 5000, 'buying_power': 10000, 'portfolio_value': 10000}
        def get_detailed_option_chain(self, symbol, expiration_date=None):
            return None

try:
    from features.feature_engineer import FeatureEngineer
except ImportError as e:
    logger.error(f"FeatureEngineer import error: {e}")
    class FeatureEngineer:
        def calculate_technical_indicators(self, df):
            return df
        def prepare_features_for_prediction(self, df):
            return df

try:
    from models.predictor import IntelligentPredictor
except ImportError as e:
    logger.error(f"IntelligentPredictor import error: {e}")
    class IntelligentPredictor:
        def load_model(self, path):
            return False
        def predict(self, features):
            return 0, 0.5

try:
    from trading.portfolio_manager import PortfolioManager
except ImportError as e:
    logger.error(f"PortfolioManager import error: {e}")
    class PortfolioManager:
        def should_enter_trade(self, symbol, confidence, positions):
            return True
        def manage_risk(self, positions, market_data):
            return []

try:
    from trading.execution_client import ExecutionClient
except ImportError as e:
    logger.error(f"ExecutionClient import error: {e}")
    class ExecutionClient:
        def get_current_positions(self):
            return {}
        def place_option_order(self, **kwargs):
            logger.info(f"Would place order: {kwargs}")
            return True
        def close_position(self, symbol):
            logger.info(f"Would close position: {symbol}")
            return True

try:
    from trading.options_strategy import OptionsStrategyEngine
except ImportError as e:
    logger.error(f"OptionsStrategyEngine import error: {e}")
    class OptionsStrategyEngine:
        def select_best_option(self, *args):
            return None
        def calculate_options_position_size(self, *args):
            return 1

try:
    from models.reinforcement_learner import ReinforcementLearner
except ImportError as e:
    logger.error(f"ReinforcementLearner import error: {e}")
    class ReinforcementLearner:
        def record_trade(self, *args): pass
        def adjust_confidence(self, symbol, confidence): return confidence
        def should_trade_symbol(self, symbol, confidence): return True

# Initialize components
data_client = DataClient()
feature_engineer = FeatureEngineer()
predictor = IntelligentPredictor()
portfolio_manager = PortfolioManager()
execution_client = ExecutionClient()
options_strategy = OptionsStrategyEngine()
reinforcement_learner = ReinforcementLearner()

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
        
        # Load pre-trained model if available
        if self.model_trained:
            logger.info("Loading trained model...")
            try:
                success = predictor.load_model('models/trained_model.pkl')
                if not success:
                    logger.warning("Failed to load trained model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning("No model available. Running in analysis-only mode.")
        
    def run_analysis_cycle(self):
        """Run one complete analysis and trading cycle"""
        logger.info("Starting analysis cycle")
        
        try:
            # 1. Get market data
            market_data = data_client.get_multiple_historical_bars(
                WATCHLIST, 
                RESAMPLE_INTERVAL, 
                LOOKBACK_WINDOW
            )
            
            # 2. Get account information
            account_info = data_client.get_account_info()
            
            # 3. Analyze each symbol
            for symbol in WATCHLIST:
                self.analyze_symbol(symbol, market_data.get(symbol), account_info)
                
            # 4. Manage existing positions
            self.manage_existing_positions(market_data)
            
            logger.info("Analysis cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            
    def analyze_symbol(self, symbol, data, account_info):
        """Analyze a single symbol and make trading decisions"""
        if data is None:
            logger.warning(f"No data for {symbol}")
            return
            
        if len(data) < 20:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
            return
            
        try:
            # 1. Feature engineering
            engineered_data = feature_engineer.calculate_technical_indicators(data)
            
            # 2. Prepare features for prediction
            prediction_features = feature_engineer.prepare_features_for_prediction(engineered_data)
            
            if prediction_features is None:
                return
                
            # 3. Get prediction from model
            prediction, confidence = predictor.predict(prediction_features)
            logger.info(f"{symbol} - Prediction: {prediction}, Confidence: {confidence:.2f}")
                
            # 4. Make trading decision
            if prediction == 1 and confidence >= PREDICTION_THRESHOLD:
                self.execute_options_trade(symbol, engineered_data, account_info, confidence, prediction)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
    def execute_options_trade(self, symbol, data, account_info, confidence, prediction):
        """Execute OPTIONS trade based on analysis"""
        if not ENABLE_TRADING:
            logger.info(f"Would place OPTIONS trade on {symbol} with confidence {confidence:.2f}")
            return
            
        try:
            # Check if we should trade this symbol based on historical performance
            if not reinforcement_learner.should_trade_symbol(symbol, confidence):
                logger.info(f"Skipping {symbol} due to poor historical performance")
                return
                
            current_price = data['close'].iloc[-1] if hasattr(data, 'iloc') else 100
            
            # Get detailed options chain
            option_chain = data_client.get_detailed_option_chain(symbol, OPTIONS_EXPIRATION)
            
            if not option_chain:
                logger.warning(f"No options chain available for {symbol}")
                return
                
            # Select best option based on prediction
            selected_option = options_strategy.select_best_option(
                symbol, prediction, confidence, option_chain, current_price
            )
            
            if not selected_option:
                logger.warning(f"No suitable option found for {symbol}")
                return
                
            # Calculate position size
            contracts = options_strategy.calculate_options_position_size(
                account_info.get('equity', 10000), selected_option['price'], confidence
            )
            
            # Store trade information for learning
            trade_data = {
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence,
                'option_type': selected_option['type'],
                'strike': selected_option['strike'],
                'contracts': contracts,
                'entry_price': selected_option['price'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Place options order
            success = execution_client.place_option_order(
                symbol=selected_option['symbol'],
                quantity=contracts,
                order_type=selected_option['type'],
                strike=selected_option['strike'],
                expiration=selected_option['expiration']
            )
                
            if success:
                logger.info(f"Options trade executed: {contracts} contracts of {selected_option['symbol']}")
                # Record trade for future learning
                reinforcement_learner.record_trade(trade_data)
                
        except Exception as e:
            logger.error(f"Error executing options trade for {symbol}: {e}")
            
    def manage_existing_positions(self, market_data):
        """Manage risk on existing positions"""
        try:
            current_positions = execution_client.get_current_positions()
            exit_signals = portfolio_manager.manage_risk(current_positions, market_data)
            
            for signal in exit_signals:
                execution_client.close_position(signal['symbol'])
                logger.info(f"Exit signal: Closed {signal['symbol']}")
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            
    def run(self):
        """Main bot loop"""
        logger.info("Starting Intelligent Trading Bot")
        
        while True:
            try:
                self.run_analysis_cycle()
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(30)

if __name__ == "__main__":
    bot = IntelligentTradingBot()
    bot.run()

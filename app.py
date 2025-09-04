import logging
import time
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)

logger = logging.getLogger(__name__)

from config.settings import settings
from data.data_client import DataClient
from features.feature_engineer import FeatureEngineer
from models.predictor import IntelligentPredictor
from trading.portfolio_manager import PortfolioManager
from trading.execution_client import ExecutionClient

class IntelligentTradingBot:
    def __init__(self):
        self.data_client = DataClient()
        self.feature_engineer = FeatureEngineer()
        self.predictor = IntelligentPredictor()
        self.portfolio_manager = PortfolioManager()
        self.execution_client = ExecutionClient()
        
        self.predictor.load_model('models/trained_model.pkl')
        
    def run_analysis_cycle(self):
        logger.info("Starting analysis cycle")
        
        try:
            market_data = self.data_client.get_multiple_historical_bars(
                settings.WATCHLIST, 
                settings.RESAMPLE_INTERVAL, 
                settings.LOOKBACK_WINDOW
            )
            
            account_info = self.data_client.get_account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return
                
            for symbol in settings.WATCHLIST:
                self.analyze_symbol(symbol, market_data.get(symbol), account_info)
                
            self.manage_existing_positions(market_data)
            
            logger.info("Analysis cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            
    def analyze_symbol(self, symbol: str, data: pd.DataFrame, account_info: dict):
        if data is None or len(data) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return
            
        try:
            engineered_data = self.feature_engineer.calculate_technical_indicators(data)
            prediction_features = self.feature_engineer.prepare_features_for_prediction(engineered_data)
            
            if prediction_features is None:
                return
                
            prediction, confidence = self.predictor.predict(prediction_features)
            
            logger.info(f"{symbol} - Prediction: {prediction}, Confidence: {confidence:.2f}")
            
            if prediction == 1 and confidence >= settings.PREDICTION_THRESHOLD:
                self.execute_trade(symbol, engineered_data, account_info, confidence)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
    def execute_trade(self, symbol: str, data: pd.DataFrame, account_info: dict, confidence: float):
        try:
            current_price = data['close'].iloc[-1]
            volatility = data['volatility'].iloc[-1] if 'volatility' in data else 0.02
            
            current_positions = self.execution_client.get_current_positions()
            if not self.portfolio_manager.should_enter_trade(symbol, confidence, current_positions):
                return
                
            stop_loss = self.portfolio_manager.determine_stop_loss(current_price, volatility, 'breakout')
            position_size = self.portfolio_manager.calculate_position_size(
                account_info['equity'], current_price, stop_loss, confidence
            )
            
            if position_size <= 0:
                return
                
            success = self.execution_client.place_option_order(
                symbol=symbol,
                quantity=position_size,
                order_type='call',
                strike=current_price * 1.05,
                expiration='2025-01-17'
            )
            
            if success:
                logger.info(f"Trade executed: {position_size} calls on {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            
    def manage_existing_positions(self, market_data: dict):
        try:
            current_positions = self.execution_client.get_current_positions()
            exit_signals = self.portfolio_manager.manage_risk(current_positions, market_data)
            
            for signal in exit_signals:
                self.execution_client.close_position(signal['symbol'])
                logger.info(f"Exit signal: Closed {signal['symbol']} due to {signal['reason']}")
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            
    def run(self):
        logger.info("Starting Intelligent Trading Bot")
        
        while True:
            try:
                self.run_analysis_cycle()
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

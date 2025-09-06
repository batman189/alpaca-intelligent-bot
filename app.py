"""
Professional Trading Bot - Main Application
Complete rebuild with real intelligence and pattern recognition
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from typing import Dict, List

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot.log')
    ]
)

# Create logs directory
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

logger = logging.getLogger(__name__)

# Flask app for monitoring and health checks
app = Flask(__name__)

# Configuration from environment variables
class Config:
    # Alpaca API settings
    APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
    APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY') 
    APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Trading settings
    WATCHLIST = os.getenv('WATCHLIST', 'SPY,QQQ,TSLA,AAPL,MSFT,NVDA,GOOGL,AMZN,META').split(',')
    ENABLE_TRADING = os.getenv('ENABLE_TRADING', 'false').lower() == 'true'
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.05'))  # 5% risk per trade
    
    # Analysis settings
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.75'))
    ANALYSIS_INTERVAL = int(os.getenv('ANALYSIS_INTERVAL', '60'))  # seconds
    
    # Multi-timeframe settings
    TIMEFRAMES = ['1Min', '5Min', '15Min', '1Hour']
    PRIMARY_TIMEFRAME = os.getenv('PRIMARY_TIMEFRAME', '15Min')
    
    # Options settings
    FOCUS_OPTIONS = os.getenv('FOCUS_OPTIONS', 'true').lower() == 'true'
    OPTIONS_EXPIRATION_DAYS = int(os.getenv('OPTIONS_EXPIRATION_DAYS', '30'))
    
    # ML Model settings
    RETRAIN_MODELS = os.getenv('RETRAIN_MODELS', 'false').lower() == 'true'
    MODEL_UPDATE_FREQUENCY = int(os.getenv('MODEL_UPDATE_FREQUENCY', '24'))  # hours

# Import components with error handling
try:
    from models.advanced_market_analyzer import AdvancedMarketAnalyzer
    from data.enhanced_data_client import EnhancedDataClient
    from trading.professional_options_engine import ProfessionalOptionsEngine
    from trading.advanced_execution_client import AdvancedExecutionClient
    from trading.intelligent_risk_manager import IntelligentRiskManager
    from models.adaptive_learning_system import AdaptiveLearningSystem
    
    COMPONENTS_LOADED = True
    logger.info("All components loaded successfully")
    
except ImportError as e:
    logger.error(f"Component import failed: {e}")
    # Fallback components for graceful degradation
    COMPONENTS_LOADED = False
    
    class MockComponent:
        def __init__(self): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None

class ProfessionalTradingBot:
    def __init__(self):
        self.config = Config()
        self.running = False
        self.last_analysis_time = datetime.now()
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Initialize components
        self.initialize_components()
        
        # Start Flask server
        self.start_web_server()
        
        # Initialize models if needed
        self.initialize_models()
    
    def initialize_components(self):
        """Initialize all trading components"""
        try:
            if COMPONENTS_LOADED:
                self.market_analyzer = AdvancedMarketAnalyzer()
                self.data_client = EnhancedDataClient()
                self.options_engine = ProfessionalOptionsEngine()
                self.execution_client = AdvancedExecutionClient()
                self.risk_manager = IntelligentRiskManager()
                self.learning_system = AdaptiveLearningSystem()
                
                logger.info("All components initialized successfully")
            else:
                # Fallback mode
                self.market_analyzer = MockComponent()
                self.data_client = MockComponent()
                self.options_engine = MockComponent()
                self.execution_client = MockComponent()
                self.risk_manager = MockComponent()
                self.learning_system = MockComponent()
                
                logger.warning("Running in fallback mode due to import errors")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def start_web_server(self):
        """Start Flask web server in background thread"""
        def run_server():
            app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info("Web server started on port 10000")
    
    def initialize_models(self):
        """Initialize or retrain ML models"""
        try:
            model_path = 'models/trained_models.pkl'
            
            # Try to load existing models
            if os.path.exists(model_path) and not self.config.RETRAIN_MODELS:
                success = self.market_analyzer.load_models(model_path)
                if success:
                    logger.info("Existing models loaded successfully")
                    return
            
            # Train new models with historical data
            logger.info("Training new models with historical data...")
            self.train_models_with_historical_data()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def train_models_with_historical_data(self):
        """Train ML models using historical data"""
        try:
            for symbol in self.config.WATCHLIST:
                logger.info(f"Training model for {symbol}...")
                
                # Get 1 year of historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                historical_data = self.data_client.get_historical_data(
                    symbol, 
                    timeframe='1Day',
                    start=start_date,
                    end=end_date,
                    limit=365
                )
                
                if historical_data is not None and len(historical_data) > 100:
                    success = self.market_analyzer.train_model(symbol, historical_data)
                    if success:
                        logger.info(f"Model trained successfully for {symbol}")
                    else:
                        logger.warning(f"Model training failed for {symbol}")
                else:
                    logger.warning(f"Insufficient historical data for {symbol}")
            
            # Save trained models
            self.market_analyzer.save_models('models/trained_models.pkl')
            logger.info("Model training completed and saved")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def run_market_analysis_cycle(self):
        """Run complete market analysis and trading cycle"""
        try:
            logger.info("Starting market analysis cycle...")
            
            # Get current account status
            account_info = self.execution_client.get_account_info()
            current_positions = self.execution_client.get_current_positions()
            
            logger.info(f"Account equity: ${account_info.get('equity', 0):,.2f}")
            logger.info(f"Current positions: {len(current_positions)}")
            
            # Analyze each symbol in watchlist
            analysis_results = []
            
            for symbol in self.config.WATCHLIST:
                result = self.analyze_single_symbol(symbol, account_info, current_positions)
                if result:
                    analysis_results.append(result)
            
            # Rank opportunities by confidence and potential
            ranked_opportunities = self.rank_trading_opportunities(analysis_results)
            
            # Execute top opportunities (if trading enabled)
            if self.config.ENABLE_TRADING and ranked_opportunities:
                self.execute_trading_decisions(ranked_opportunities, account_info)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            logger.info("Market analysis cycle completed")
            
        except Exception as e:
            logger.error(f"Error in market analysis cycle: {e}")
    
    def analyze_single_symbol(self, symbol: str, account_info: Dict, current_positions: Dict) -> Dict:
        """Analyze a single symbol across multiple timeframes"""
        try:
            logger.info(f"Analyzing {symbol}...")
            
            # Get multi-timeframe data
            timeframe_data = {}
            for tf in self.config.TIMEFRAMES:
                data = self.data_client.get_real_time_bars(symbol, timeframe=tf, limit=200)
                if data is not None and len(data) >= 50:
                    timeframe_data[tf] = data
            
            if not timeframe_data:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Perform analysis on primary timeframe
            primary_data = timeframe_data.get(self.config.PRIMARY_TIMEFRAME)
            if primary_data is None:
                return None
            
            analysis = self.market_analyzer.analyze_symbol(symbol, primary_data)
            
            # Multi-timeframe confluence check
            confluence_score = self.check_timeframe_confluence(symbol, timeframe_data)
            
            # Adjust confidence based on confluence
            analysis['confidence'] *= confluence_score
            analysis['confluence_score'] = confluence_score
            
            # Risk assessment
            risk_assessment = self.risk_manager.assess_symbol_risk(
                symbol, primary_data, current_positions, account_info
            )
            analysis['risk_assessment'] = risk_assessment
            
            # Options analysis (if focus is on options)
            if self.config.FOCUS_OPTIONS:
                options_analysis = self.options_engine.analyze_options_opportunity(
                    symbol, analysis, primary_data
                )
                analysis['options_analysis'] = options_analysis
            
            # Learning system feedback
            historical_performance = self.learning_system.get_symbol_performance(symbol)
            analysis['historical_performance'] = historical_performance
            
            # Final trade recommendation
            should_trade = self.should_execute_trade(analysis, risk_assessment)
            analysis['trade_recommendation'] = should_trade
            
            logger.info(f"{symbol}: Prediction={analysis['prediction']}, "
                       f"Confidence={analysis['confidence']:.2f}, "
                       f"Confluence={confluence_score:.2f}, "
                       f"Recommend={should_trade}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def check_timeframe_confluence(self, symbol: str, timeframe_data: Dict) -> float:
        """Check for confluence across multiple timeframes"""
        try:
            signals = []
            
            for tf, data in timeframe_data.items():
                if data is not None and len(data) >= 20:
                    tf_analysis = self.market_analyzer.analyze_symbol(symbol, data)
                    signals.append(tf_analysis['prediction'])
            
            if not signals:
                return 0.5  # Neutral if no data
            
            # Calculate confluence: percentage of timeframes agreeing
            bullish_signals = sum(signals)
            total_signals = len(signals)
            
            if bullish_signals == 0:
                return 0.3  # All bearish
            elif bullish_signals == total_signals:
                return 1.0  # All bullish
            else:
                return 0.5 + (bullish_signals / total_signals - 0.5) * 0.5
            
        except Exception as e:
            logger.error(f"Error checking confluence for {symbol}: {e}")
            return 0.5
    
    def rank_trading_opportunities(self, analysis_results: List[Dict]) -> List[Dict]:
        """Rank trading opportunities by potential and risk"""
        try:
            if not analysis_results:
                return []
            
            # Filter for tradeable opportunities
            opportunities = [
                result for result in analysis_results 
                if (result['confidence'] >= self.config.CONFIDENCE_THRESHOLD and 
                    result['trade_recommendation'])
            ]
            
            # Sort by composite score (confidence * confluence * risk adjustment)
            for opp in opportunities:
                risk_multiplier = 1.0 - opp['risk_assessment'].get('risk_score', 0)
                composite_score = (
                    opp['confidence'] * 
                    opp['confluence_score'] * 
                    risk_multiplier
                )
                opp['composite_score'] = composite_score
            
            # Sort by composite score (highest first)
            ranked = sorted(opportunities, key=lambda x: x['composite_score'], reverse=True)
            
            logger.info(f"Found {len(ranked)} trading opportunities")
            return ranked
            
        except Exception as e:
            logger.error(f"Error ranking opportunities: {e}")
            return []
    
    def should_execute_trade(self, analysis: Dict, risk_assessment: Dict) -> bool:
        """Determine if a trade should be executed"""
        try:
            # Basic checks
            if analysis['confidence'] < self.config.CONFIDENCE_THRESHOLD:
                return False
            
            if risk_assessment.get('risk_score', 1.0) > 0.7:  # Too risky
                return False
            
            # Learning system check
            symbol = analysis['symbol']
            if not self.learning_system.should_trade_symbol(symbol, analysis['confidence']):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trade decision: {e}")
            return False
    
    def execute_trading_decisions(self, opportunities: List[Dict], account_info: Dict):
        """Execute trading decisions based on analysis"""
        try:
            executed_trades = 0
            max_new_positions = min(3, self.config.MAX_POSITIONS)  # Limit new positions
            
            for opportunity in opportunities[:max_new_positions]:
                symbol = opportunity['symbol']
                
                try:
                    if self.config.FOCUS_OPTIONS:
                        success = self.execute_options_trade(opportunity, account_info)
                    else:
                        success = self.execute_stock_trade(opportunity, account_info)
                    
                    if success:
                        executed_trades += 1
                        logger.info(f"Trade executed successfully for {symbol}")
                        
                        # Record trade for learning
                        self.learning_system.record_trade_entry(opportunity)
                    
                except Exception as e:
                    logger.error(f"Error executing trade for {symbol}: {e}")
            
            logger.info(f"Executed {executed_trades} trades this cycle")
            
        except Exception as e:
            logger.error(f"Error executing trading decisions: {e}")
    
    def execute_options_trade(self, opportunity: Dict, account_info: Dict) -> bool:
        """Execute an options trade"""
        try:
            symbol = opportunity['symbol']
            options_analysis = opportunity.get('options_analysis', {})
            
            if not options_analysis:
                logger.warning(f"No options analysis for {symbol}")
                return False
            
            # Get the best option contract
            best_option = self.options_engine.select_best_option_contract(
                symbol, opportunity, account_info
            )
            
            if not best_option:
                logger.warning(f"No suitable option contract for {symbol}")
                return False
            
            # Calculate position size
            position_size = self.risk_manager.calculate_options_position_size(
                account_info, best_option, opportunity
            )
            
            # Execute the trade
            success = self.execution_client.place_options_order(
                best_option, position_size, opportunity
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing options trade: {e}")
            return False
    
    def execute_stock_trade(self, opportunity: Dict, account_info: Dict) -> bool:
        """Execute a stock trade (fallback if not focusing on options)"""
        try:
            symbol = opportunity['symbol']
            
            # Calculate position size
            position_size = self.risk_manager.calculate_stock_position_size(
                account_info, opportunity
            )
            
            # Execute the trade
            success = self.execution_client.place_stock_order(
                symbol, position_size, opportunity
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing stock trade: {e}")
            return False
    
    def update_performance_metrics(self):
        """Update bot performance metrics"""
        try:
            # Get updated account info
            account_info = self.execution_client.get_account_info()
            
            # Update metrics from learning system
            overall_performance = self.learning_system.get_overall_performance()
            
            self.performance_metrics.update({
                'current_equity': account_info.get('equity', 0),
                'total_trades': overall_performance.get('total_trades', 0),
                'profitable_trades': overall_performance.get('profitable_trades', 0),
                'total_pnl': overall_performance.get('total_pnl', 0),
                'win_rate': overall_performance.get('win_rate', 0),
                'last_updated': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def run(self):
        """Main bot execution loop"""
        logger.info("Starting Professional Trading Bot")
        logger.info(f"Watchlist: {', '.join(self.config.WATCHLIST)}")
        logger.info(f"Trading enabled: {self.config.ENABLE_TRADING}")
        logger.info(f"Focus on options: {self.config.FOCUS_OPTIONS}")
        
        self.running = True
        
        while self.running:
            try:
                # Check if markets are open (simple check)
                now = datetime.now()
                if self.is_market_hours(now):
                    self.run_market_analysis_cycle()
                else:
                    logger.info("Markets closed, waiting...")
                
                # Sleep until next analysis
                time.sleep(self.config.ANALYSIS_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def is_market_hours(self, current_time: datetime) -> bool:
        """Check if markets are currently open (simplified)"""
        # This is a basic check - you might want to use a more sophisticated market calendar
        weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        hour = current_time.hour
        
        # Monday to Friday, 9:30 AM to 4:00 PM EST (simplified)
        if weekday < 5 and 9 <= hour < 16:
            return True
        return False

# Flask routes for monitoring
@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'professional-trading-bot',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def detailed_health():
    return jsonify({
        'status': 'ok',
        'components_loaded': COMPONENTS_LOADED,
        'config': {
            'watchlist_size': len(Config.WATCHLIST),
            'trading_enabled': Config.ENABLE_TRADING,
            'focus_options': Config.FOCUS_OPTIONS
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/performance')
def get_performance():
    """Get bot performance metrics"""
    if 'bot' in globals():
        return jsonify(bot.performance_metrics)
    else:
        return jsonify({'error': 'Bot not initialized'}), 503

@app.route('/test')
def run_test():
    """Run system test"""
    try:
        # Create temporary bot instance for testing
        test_bot = ProfessionalTradingBot()
        
        # Run basic tests
        test_results = {
            'components_loaded': COMPONENTS_LOADED,
            'market_analyzer': hasattr(test_bot, 'market_analyzer'),
            'data_client': hasattr(test_bot, 'data_client'),
            'execution_client': hasattr(test_bot, 'execution_client'),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'test_completed',
            'results': test_results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'test_failed', 
            'error': str(e)
        }), 500

# Main execution
if __name__ == "__main__":
    try:
        bot = ProfessionalTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"Critical error starting bot: {e}")
        raise

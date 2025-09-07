"""
Professional Trading Bot - Main Application
Complete rebuild with real intelligence and pattern recognition
UPDATED VERSION - Fixed imports and async issues
"""

import os
import time
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

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

# Import components with error handling and fallback
COMPONENTS_LOADED = False
try:
    # Fixed import paths to match actual file structure
    from models.advanced_market_analyzer import AdvancedMarketAnalyzer
    from data.data_client import EnhancedDataClient
    from trading.options_engine import ProfessionalOptionsEngine
    from trading.execution_client import AdvancedExecutionClient
    from trading.intelligent_risk_manager import IntelligentRiskManager
    from models.adaptive_learning_system import AdaptiveLearningSystem
    
    # Import upgrade components
    from data.multi_source_data_manager import MultiSourceDataManager
    from models.signal_aggregator import MultiSourceSignalAggregator
    from models.multi_timeframe_scanner import MultiTimeframeScanner
    from models.market_regime_detector import MarketRegimeDetector
    from models.dynamic_watchlist_manager import DynamicWatchlistManager
    from monitoring.comprehensive_logger import ComprehensiveLogger
    
    COMPONENTS_LOADED = True
    logger.info("✅ All components loaded successfully")
    
except ImportError as e:
    logger.error(f"❌ Component import failed: {e}")
    COMPONENTS_LOADED = False
    
    # Fallback components for graceful degradation
    class MockComponent:
        def __init__(self, *args, **kwargs): 
            pass
        def __getattr__(self, name): 
            return lambda *args, **kwargs: None
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

class CircuitBreaker:
    """Circuit breaker pattern for resilient service calls"""
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moved to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise e

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
        
        # Circuit breakers for critical services
        self.circuit_breakers = {
            'data_manager': CircuitBreaker(),
            'signal_aggregator': CircuitBreaker(),
            'execution_client': CircuitBreaker()
        }
        
        # Initialize components
        self.initialize_components()
        
        # Start Flask server in background
        self.start_web_server()
        
        # Initialize models if needed
        asyncio.create_task(self.initialize_models())
    
    def initialize_components(self):
        """Initialize all trading components with fallback handling"""
        try:
            if COMPONENTS_LOADED:
                # Core components
                self.market_analyzer = AdvancedMarketAnalyzer()
                self.data_client = EnhancedDataClient(
                    self.config.APCA_API_KEY_ID,
                    self.config.APCA_API_SECRET_KEY,
                    self.config.APCA_API_BASE_URL
                )
                self.options_engine = ProfessionalOptionsEngine()
                self.execution_client = AdvancedExecutionClient(
                    self.config.APCA_API_KEY_ID,
                    self.config.APCA_API_SECRET_KEY,
                    self.config.APCA_API_BASE_URL
                )
                self.risk_manager = IntelligentRiskManager()
                self.learning_system = AdaptiveLearningSystem()
                
                # Upgrade components
                self.data_manager = MultiSourceDataManager(
                    self.config.APCA_API_KEY_ID,
                    self.config.APCA_API_SECRET_KEY
                )
                self.signal_aggregator = MultiSourceSignalAggregator()
                self.timeframe_scanner = MultiTimeframeScanner()
                self.regime_detector = MarketRegimeDetector()
                self.watchlist_manager = DynamicWatchlistManager(self.config.WATCHLIST)
                self.logger = ComprehensiveLogger()
                
                logger.info("✅ All components initialized successfully")
            else:
                # Fallback mode with mock components
                self.market_analyzer = MockComponent()
                self.data_client = MockComponent()
                self.options_engine = MockComponent()
                self.execution_client = MockComponent()
                self.risk_manager = MockComponent()
                self.learning_system = MockComponent()
                
                # Mock upgrade components
                self.data_manager = MockComponent()
                self.signal_aggregator = MockComponent()
                self.timeframe_scanner = MockComponent()
                self.regime_detector = MockComponent()
                self.watchlist_manager = MockComponent()
                self.logger = MockComponent()
                
                logger.warning("⚠️ Running in fallback mode due to import errors")
                
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def start_web_server(self):
        """Start Flask web server in background thread"""
        def run_server():
            app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info("🌐 Web server started on port 10000")
    
    async def initialize_models(self):
        """Initialize or retrain ML models"""
        try:
            model_path = 'models/trained_models.pkl'
            
            # Try to load existing models
            if os.path.exists(model_path) and not self.config.RETRAIN_MODELS:
                if hasattr(self.market_analyzer, 'load_models'):
                    success = self.market_analyzer.load_models(model_path)
                    if success:
                        logger.info("✅ Existing models loaded successfully")
                        return
            
            # Train new models with historical data
            logger.info("🧠 Training new models with historical data...")
            await self.train_models_with_historical_data()
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
    
    async def train_models_with_historical_data(self):
        """Train ML models using historical data"""
        try:
            for symbol in self.config.WATCHLIST:
                logger.info(f"🎯 Training model for {symbol}...")
                
                # Get 1 year of historical data using circuit breaker
                try:
                    historical_data = await self.circuit_breakers['data_manager'].call(
                        self.data_manager.get_market_data,
                        symbol, "1Day", 365
                    )
                    
                    if historical_data is not None and len(historical_data) > 100:
                        if hasattr(self.market_analyzer, 'train_model'):
                            success = self.market_analyzer.train_model(symbol, historical_data)
                            if success:
                                logger.info(f"✅ Model trained successfully for {symbol}")
                            else:
                                logger.warning(f"⚠️ Model training failed for {symbol}")
                    else:
                        logger.warning(f"⚠️ Insufficient historical data for {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Training failed for {symbol}: {e}")
            
            # Save trained models
            if hasattr(self.market_analyzer, 'save_models'):
                self.market_analyzer.save_models('models/trained_models.pkl')
                logger.info("💾 Model training completed and saved")
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
    
    async def run_market_analysis_cycle(self):
        """Run complete market analysis and trading cycle with proper async handling"""
        try:
            logger.info("🔄 Starting market analysis cycle...")
            
            # Get current account status with circuit breaker protection
            try:
                account_info = await self.circuit_breakers['execution_client'].call(
                    self.execution_client.get_account_info
                )
                current_positions = await self.circuit_breakers['execution_client'].call(
                    self.execution_client.get_current_positions
                )
            except Exception as e:
                logger.error(f"❌ Failed to get account info: {e}")
                return
            
            logger.info(f"💰 Account equity: ${account_info.get('equity', 0):,.2f}")
            logger.info(f"📊 Current positions: {len(current_positions)}")
            
            # Update dynamic watchlist
            try:
                await self.watchlist_manager.update_watchlists(self.data_manager)
                symbols = list(self.watchlist_manager.get_all_symbols())[:20]  # Limit to 20 for performance
            except Exception as e:
                logger.error(f"❌ Watchlist update failed: {e}")
                symbols = self.config.WATCHLIST
            
            # Detect market regime
            try:
                market_indices = {}
                for index in ["SPY", "QQQ", "IWM"]:
                    try:
                        market_indices[index] = await self.data_manager.get_market_data(index, "1Day", 30)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get data for {index}: {e}")
                
                if market_indices:
                    regime_analysis = await self.regime_detector.detect_market_regime(market_indices)
                    logger.info(f"🧠 Market regime: {regime_analysis.regime.value} (confidence: {regime_analysis.confidence:.1f}%)")
                else:
                    logger.warning("⚠️ No market data available for regime detection")
                    
            except Exception as e:
                logger.error(f"❌ Market regime detection failed: {e}")
            
            # Analyze symbols in batches to prevent overwhelming APIs
            batch_size = 5
            analysis_results = []
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                logger.info(f"📈 Analyzing batch {i//batch_size + 1}: {', '.join(batch)}")
                
                # Process batch in parallel
                batch_tasks = [
                    self.analyze_single_symbol(symbol, account_info, current_positions)
                    for symbol in batch
                ]
                
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for symbol, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"❌ Analysis failed for {symbol}: {result}")
                        elif result:
                            analysis_results.append(result)
                            
                except Exception as e:
                    logger.error(f"❌ Batch analysis failed: {e}")
                
                # Small delay between batches to respect rate limits
                await asyncio.sleep(1)
            
            # Rank opportunities by confidence and potential
            ranked_opportunities = self.rank_trading_opportunities(analysis_results)
            
            # Execute top opportunities (if trading enabled)
            if self.config.ENABLE_TRADING and ranked_opportunities:
                await self.execute_trading_decisions(ranked_opportunities, account_info)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            logger.info("✅ Market analysis cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Error in market analysis cycle: {e}")
    
    async def analyze_single_symbol(self, symbol: str, account_info: Dict, current_positions: Dict) -> Optional[Dict]:
        """Analyze a single symbol with proper async handling"""
        try:
            logger.debug(f"🔍 Analyzing {symbol}...")
            
            # Get market data with circuit breaker
            try:
                market_data = await self.circuit_breakers['data_manager'].call(
                    self.data_manager.get_market_data,
                    symbol, self.config.PRIMARY_TIMEFRAME, 200
                )
                
                if market_data is None or len(market_data) < 50:
                    logger.warning(f"⚠️ Insufficient data for {symbol}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Data fetch failed for {symbol}: {e}")
                return None
            
            # Perform analysis
            try:
                analysis = self.market_analyzer.analyze_symbol(symbol, market_data)
                analysis['symbol'] = symbol
                analysis['current_price'] = market_data['close'].iloc[-1]
                
            except Exception as e:
                logger.error(f"❌ Market analysis failed for {symbol}: {e}")
                return None
            
            # Multi-timeframe analysis
            try:
                timeframe_opportunities = await self.timeframe_scanner.scan_all_timeframes(
                    symbol, self.data_manager
                )
                
                # Get consensus
                consensus = self.timeframe_scanner.get_timeframe_consensus(timeframe_opportunities)
                analysis['timeframe_consensus'] = consensus
                
            except Exception as e:
                logger.warning(f"⚠️ Timeframe analysis failed for {symbol}: {e}")
                analysis['timeframe_consensus'] = {"consensus": "unknown"}
            
            # Signal aggregation with circuit breaker
            try:
                signals = await self.circuit_breakers['signal_aggregator'].call(
                    self.signal_aggregator.aggregate_signals,
                    symbol, market_data
                )
                analysis['signals'] = signals
                
            except Exception as e:
                logger.warning(f"⚠️ Signal aggregation failed for {symbol}: {e}")
                analysis['signals'] = []
            
            # Risk assessment
            try:
                risk_assessment = self.risk_manager.assess_symbol_risk(
                    symbol, market_data, current_positions, account_info
                )
                analysis['risk_assessment'] = risk_assessment
                
            except Exception as e:
                logger.warning(f"⚠️ Risk assessment failed for {symbol}: {e}")
                analysis['risk_assessment'] = {'risk_score': 0.5}
            
            # Options analysis (if enabled)
            if self.config.FOCUS_OPTIONS:
                try:
                    options_analysis = self.options_engine.analyze_options_opportunity(
                        symbol, analysis, market_data
                    )
                    analysis['options_analysis'] = options_analysis
                    
                except Exception as e:
                    logger.warning(f"⚠️ Options analysis failed for {symbol}: {e}")
                    analysis['options_analysis'] = {}
            
            # Final trade recommendation
            should_trade = self.should_execute_trade(analysis, risk_assessment)
            analysis['trade_recommendation'] = should_trade
            
            confidence = analysis.get('confidence', 0)
            prediction = analysis.get('prediction', 0)
            
            logger.info(f"📊 {symbol}: {prediction} (confidence: {confidence:.1%}, trade: {should_trade})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def rank_trading_opportunities(self, analysis_results: List[Dict]) -> List[Dict]:
        """Rank trading opportunities by potential and risk"""
        try:
            if not analysis_results:
                return []
            
            # Filter for tradeable opportunities
            opportunities = [
                result for result in analysis_results 
                if (result.get('confidence', 0) >= self.config.CONFIDENCE_THRESHOLD and 
                    result.get('trade_recommendation', False))
            ]
            
            # Sort by composite score
            for opp in opportunities:
                confidence = opp.get('confidence', 0)
                risk_score = opp.get('risk_assessment', {}).get('risk_score', 0.5)
                signals_count = len(opp.get('signals', []))
                
                # Composite scoring
                composite_score = (
                    confidence * 0.4 +
                    (1 - risk_score) * 0.3 +
                    min(signals_count / 5, 1.0) * 0.3
                )
                opp['composite_score'] = composite_score
            
            # Sort by composite score (highest first)
            ranked = sorted(opportunities, key=lambda x: x.get('composite_score', 0), reverse=True)
            
            logger.info(f"🎯 Found {len(ranked)} trading opportunities")
            return ranked
            
        except Exception as e:
            logger.error(f"❌ Error ranking opportunities: {e}")
            return []
    
    def should_execute_trade(self, analysis: Dict, risk_assessment: Dict) -> bool:
        """Determine if a trade should be executed"""
        try:
            confidence = analysis.get('confidence', 0)
            risk_score = risk_assessment.get('risk_score', 1.0)
            
            # Basic checks
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return False
            
            if risk_score > 0.7:  # Too risky
                return False
            
            # Learning system check
            symbol = analysis.get('symbol')
            if symbol and hasattr(self.learning_system, 'should_trade_symbol'):
                if not self.learning_system.should_trade_symbol(symbol, confidence):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in trade decision: {e}")
            return False
    
    async def execute_trading_decisions(self, opportunities: List[Dict], account_info: Dict):
        """Execute trading decisions based on analysis"""
        try:
            executed_trades = 0
            max_new_positions = min(3, self.config.MAX_POSITIONS)
            
            for opportunity in opportunities[:max_new_positions]:
                symbol = opportunity.get('symbol')
                if not symbol:
                    continue
                
                try:
                    if self.config.FOCUS_OPTIONS:
                        success = await self.execute_options_trade(opportunity, account_info)
                    else:
                        success = await self.execute_stock_trade(opportunity, account_info)
                    
                    if success:
                        executed_trades += 1
                        logger.info(f"✅ Trade executed successfully for {symbol}")
                        
                        # Record trade for learning
                        if hasattr(self.learning_system, 'record_trade_entry'):
                            self.learning_system.record_trade_entry(opportunity)
                        
                        # Log to comprehensive logger
                        self.logger.log_opportunity_executed(
                            symbol, 
                            opportunity.get('signals', [{}])[0].get('signal_type', 'unknown'),
                            opportunity.get('confidence', 0),
                            opportunity.get('current_price', 0),
                            1,  # position_size placeholder
                            opportunity
                        )
                    
                except Exception as e:
                    logger.error(f"❌ Error executing trade for {symbol}: {e}")
                    
                    # Log missed opportunity
                    self.logger.log_opportunity_missed(
                        symbol,
                        opportunity.get('signals', [{}])[0].get('signal_type', 'unknown'),
                        "execution_error",
                        opportunity.get('confidence', 0),
                        {"error": str(e)}
                    )
            
            logger.info(f"📈 Executed {executed_trades} trades this cycle")
            
        except Exception as e:
            logger.error(f"❌ Error executing trading decisions: {e}")
    
    async def execute_options_trade(self, opportunity: Dict, account_info: Dict) -> bool:
        """Execute an options trade"""
        try:
            symbol = opportunity.get('symbol')
            options_analysis = opportunity.get('options_analysis', {})
            
            if not options_analysis:
                logger.warning(f"⚠️ No options analysis for {symbol}")
                return False
            
            # Get the best option contract
            best_option = self.options_engine.select_best_option_contract(
                symbol, opportunity, account_info
            )
            
            if not best_option:
                logger.warning(f"⚠️ No suitable option contract for {symbol}")
                return False
            
            # Calculate position size
            position_size = self.risk_manager.calculate_options_position_size(
                account_info, best_option, opportunity
            )
            
            if position_size <= 0:
                logger.warning(f"⚠️ Invalid position size for {symbol}")
                return False
            
            # Execute the trade with circuit breaker
            success = await self.circuit_breakers['execution_client'].call(
                self.execution_client.place_options_order,
                best_option.get('symbol', symbol),
                'buy',  # Simplified
                position_size,
                'limit',
                best_option.get('ask', 1.0)
            )
            
            return success is not None
            
        except Exception as e:
            logger.error(f"❌ Error executing options trade: {e}")
            return False
    
    async def execute_stock_trade(self, opportunity: Dict, account_info: Dict) -> bool:
        """Execute a stock trade"""
        try:
            symbol = opportunity.get('symbol')
            
            # Calculate position size
            position_size = self.risk_manager.calculate_stock_position_size(
                account_info, opportunity
            )
            
            if position_size <= 0:
                logger.warning(f"⚠️ Invalid position size for {symbol}")
                return False
            
            # Determine side based on prediction
            side = 'buy' if opportunity.get('prediction', 0) > 0 else 'sell'
            
            # Execute the trade with circuit breaker
            success = await self.circuit_breakers['execution_client'].call(
                self.execution_client.place_order,
                symbol,
                side,
                position_size,
                'market'  # Simplified order type
            )
            
            return success is not None
            
        except Exception as e:
            logger.error(f"❌ Error executing stock trade: {e}")
            return False
    
    def update_performance_metrics(self):
        """Update bot performance metrics"""
        try:
            # This would get real performance data in production
            # For now, just update timestamp
            self.performance_metrics['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def run(self):
        """Main bot execution loop with proper async handling"""
        logger.info("🚀 Starting Professional Trading Bot")
        logger.info(f"📋 Watchlist: {', '.join(self.config.WATCHLIST)}")
        logger.info(f"💰 Trading enabled: {self.config.ENABLE_TRADING}")
        logger.info(f"📊 Focus on options: {self.config.FOCUS_OPTIONS}")
        
        self.running = True
        
        # Start background services
        if hasattr(self.execution_client, 'start'):
            await self.execution_client.start()
        if hasattr(self.data_client, 'start'):
            await self.data_client.start(self.config.WATCHLIST)
        
        try:
            while self.running:
                try:
                    # Check if markets are open (simple check)
                    if self.is_market_hours(datetime.now()):
                        await self.run_market_analysis_cycle()
                    else:
                        logger.info("🌙 Markets closed, waiting...")
                        await asyncio.sleep(300)  # Wait 5 minutes when markets closed
                    
                    # Sleep until next analysis
                    await asyncio.sleep(self.config.ANALYSIS_INTERVAL)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Bot stopped by user")
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"❌ Unexpected error in main loop: {e}")
                    await asyncio.sleep(60)  # Wait a minute before retrying
                    
        finally:
            # Cleanup
            if hasattr(self.execution_client, 'stop'):
                await self.execution_client.stop()
            if hasattr(self.data_client, 'stop'):
                await self.data_client.stop()
    
    def is_market_hours(self, current_time: datetime) -> bool:
        """Check if markets are currently open (simplified)"""
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
        'timestamp': datetime.now().isoformat(),
        'components_loaded': COMPONENTS_LOADED
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
    try:
        if 'bot' in globals():
            return jsonify(bot.performance_metrics)
        else:
            return jsonify({'error': 'Bot not initialized'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def run_test():
    """Run system test"""
    try:
        # Create temporary bot instance for testing
        test_results = {
            'components_loaded': COMPONENTS_LOADED,
            'config_valid': bool(Config.APCA_API_KEY_ID),
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
async def main():
    """Main async entry point"""
    try:
        bot = ProfessionalTradingBot()
        await bot.run()
    except Exception as e:
        logger.error(f"❌

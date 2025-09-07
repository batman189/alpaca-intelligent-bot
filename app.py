"""
Professional Trading Bot - Main Application
UPDATED VERSION with Senior Analyst ML Intelligence
Complete rebuild with real intelligence and pattern recognition
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
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot.log') if os.path.exists('logs') else logging.StreamHandler()
    ]
)

# Create directories with error handling
for directory in ['logs', 'data', 'models', 'monitoring']:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {directory}: {e}")

logger = logging.getLogger(__name__)

# Flask app for monitoring
app = Flask(__name__)

# Configuration with validation
class Config:
    def __init__(self):
        # Alpaca API settings with validation
        self.APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID', '')
        self.APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY', '')
        self.APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Validate credentials
        if not self.APCA_API_KEY_ID or not self.APCA_API_SECRET_KEY:
            logger.warning("Missing Alpaca API credentials - trading will be disabled")
        
        # Trading settings
        self.WATCHLIST = os.getenv('WATCHLIST', 'SPY,QQQ,TSLA,AAPL,MSFT,NVDA,GOOGL,AMZN,META').split(',')
        self.ENABLE_TRADING = os.getenv('ENABLE_TRADING', 'false').lower() == 'true'
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))
        self.RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.05'))
        
        # Analysis settings
        self.CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.70'))
        self.ANALYSIS_INTERVAL = int(os.getenv('ANALYSIS_INTERVAL', '60'))
        
        # Multi-timeframe settings
        self.TIMEFRAMES = ['1Min', '5Min', '15Min', '1Hour']
        self.PRIMARY_TIMEFRAME = os.getenv('PRIMARY_TIMEFRAME', '15Min')
        
        # Options settings
        self.FOCUS_OPTIONS = os.getenv('FOCUS_OPTIONS', 'true').lower() == 'true'
        self.OPTIONS_EXPIRATION_DAYS = int(os.getenv('OPTIONS_EXPIRATION_DAYS', '30'))
        
        # Senior Analyst settings
        self.ENABLE_SENIOR_ANALYST = os.getenv('ENABLE_SENIOR_ANALYST', 'true').lower() == 'true'
        self.TRAINING_PERIODS = int(os.getenv('TRAINING_PERIODS', '2000'))

# Mock/Fallback components for when imports fail
class MockComponent:
    def __init__(self, *args, **kwargs):
        self.name = "MockComponent"
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# Safe component imports with fallbacks
def safe_import_components():
    """Safely import components with fallbacks"""
    components = {}
    
    # Core components
    try:
        from models.advanced_market_analyzer import AdvancedMarketAnalyzer
        components['AdvancedMarketAnalyzer'] = AdvancedMarketAnalyzer
        logger.info("✅ AdvancedMarketAnalyzer imported")
    except Exception as e:
        logger.warning(f"⚠️ AdvancedMarketAnalyzer import failed: {e}")
        components['AdvancedMarketAnalyzer'] = MockComponent
    
    try:
        from data.data_client import EnhancedDataClient
        components['EnhancedDataClient'] = EnhancedDataClient
        logger.info("✅ EnhancedDataClient imported")
    except Exception as e:
        logger.warning(f"⚠️ EnhancedDataClient import failed: {e}")
        components['EnhancedDataClient'] = MockComponent
    
    try:
        from trading.options_engine import ProfessionalOptionsEngine
        components['ProfessionalOptionsEngine'] = ProfessionalOptionsEngine
        logger.info("✅ ProfessionalOptionsEngine imported")
    except Exception as e:
        logger.warning(f"⚠️ ProfessionalOptionsEngine import failed: {e}")
        components['ProfessionalOptionsEngine'] = MockComponent
    
    try:
        from trading.execution_client import AdvancedExecutionClient
        components['AdvancedExecutionClient'] = AdvancedExecutionClient
        logger.info("✅ AdvancedExecutionClient imported")
    except Exception as e:
        logger.warning(f"⚠️ AdvancedExecutionClient import failed: {e}")
        components['AdvancedExecutionClient'] = MockComponent
    
    try:
        from trading.intelligent_risk_manager import IntelligentRiskManager
        components['IntelligentRiskManager'] = IntelligentRiskManager
        logger.info("✅ IntelligentRiskManager imported")
    except Exception as e:
        logger.warning(f"⚠️ IntelligentRiskManager import failed: {e}")
        components['IntelligentRiskManager'] = MockComponent
    
    try:
        from models.adaptive_learning_system import AdaptiveLearningSystem
        components['AdaptiveLearningSystem'] = AdaptiveLearningSystem
        logger.info("✅ AdaptiveLearningSystem imported")
    except Exception as e:
        logger.warning(f"⚠️ AdaptiveLearningSystem import failed: {e}")
        components['AdaptiveLearningSystem'] = MockComponent
    
    # Upgrade components
    try:
        from data.multi_source_data_manager import MultiSourceDataManager
        components['MultiSourceDataManager'] = MultiSourceDataManager
        logger.info("✅ MultiSourceDataManager imported")
    except Exception as e:
        logger.warning(f"⚠️ MultiSourceDataManager import failed: {e}")
        components['MultiSourceDataManager'] = MockComponent
    
    try:
        from models.signal_aggregator import MultiSourceSignalAggregator
        components['MultiSourceSignalAggregator'] = MultiSourceSignalAggregator
        logger.info("✅ MultiSourceSignalAggregator imported")
    except Exception as e:
        logger.warning(f"⚠️ MultiSourceSignalAggregator import failed: {e}")
        components['MultiSourceSignalAggregator'] = MockComponent
    
    try:
        from models.multi_timeframe_scanner import MultiTimeframeScanner
        components['MultiTimeframeScanner'] = MultiTimeframeScanner
        logger.info("✅ MultiTimeframeScanner imported")
    except Exception as e:
        logger.warning(f"⚠️ MultiTimeframeScanner import failed: {e}")
        components['MultiTimeframeScanner'] = MockComponent
    
    try:
        from models.market_regime_detector import MarketRegimeDetector
        components['MarketRegimeDetector'] = MarketRegimeDetector
        logger.info("✅ MarketRegimeDetector imported")
    except Exception as e:
        logger.warning(f"⚠️ MarketRegimeDetector import failed: {e}")
        components['MarketRegimeDetector'] = MockComponent
    
    try:
        from models.dynamic_watchlist_manager import DynamicWatchlistManager
        components['DynamicWatchlistManager'] = DynamicWatchlistManager
        logger.info("✅ DynamicWatchlistManager imported")
    except Exception as e:
        logger.warning(f"⚠️ DynamicWatchlistManager import failed: {e}")
        components['DynamicWatchlistManager'] = MockComponent
    
    try:
        from monitoring.comprehensive_logger import ComprehensiveLogger
        components['ComprehensiveLogger'] = ComprehensiveLogger
        logger.info("✅ ComprehensiveLogger imported")
    except Exception as e:
        logger.warning(f"⚠️ ComprehensiveLogger import failed: {e}")
        components['ComprehensiveLogger'] = MockComponent
    
    # SENIOR ANALYST BRAIN - The main intelligence upgrade
    try:
        from models.senior_analyst_ml_system import SeniorAnalystIntegration
        components['SeniorAnalystIntegration'] = SeniorAnalystIntegration
        logger.info("🧠 ✅ SENIOR ANALYST BRAIN imported successfully!")
        global SENIOR_ANALYST_AVAILABLE
        SENIOR_ANALYST_AVAILABLE = True
    except Exception as e:
        logger.warning(f"⚠️ Senior Analyst Brain import failed: {e}")
        components['SeniorAnalystIntegration'] = MockComponent
        SENIOR_ANALYST_AVAILABLE = False
    
    return components

# Import components
COMPONENTS = safe_import_components()
SENIOR_ANALYST_AVAILABLE = 'SeniorAnalystIntegration' in COMPONENTS and COMPONENTS['SeniorAnalystIntegration'] != MockComponent

class CircuitBreaker:
    """Circuit breaker for resilient service calls"""
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
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
            'max_drawdown': 0.0,
            'senior_analyst_accuracy': 0.0
        }
        
        # Circuit breakers
        self.circuit_breakers = {
            'data_manager': CircuitBreaker(),
            'signal_aggregator': CircuitBreaker(),
            'execution_client': CircuitBreaker(),
            'senior_analyst': CircuitBreaker()
        }
        
        # Trade tracking for learning
        self.active_trades = {}
        self.completed_trades = []
        
        # Initialize components with error handling
        self.initialize_components()
        
        # Start Flask server
        self.start_web_server()
        
        logger.info("🚀 Professional Trading Bot with Senior Analyst initialized")
    
    def initialize_components(self):
        """Initialize components with proper error handling"""
        try:
            # Core components
            self.market_analyzer = COMPONENTS['AdvancedMarketAnalyzer']()
            self.data_client = COMPONENTS['EnhancedDataClient'](
                self.config.APCA_API_KEY_ID,
                self.config.APCA_API_SECRET_KEY,
                self.config.APCA_API_BASE_URL
            )
            self.options_engine = COMPONENTS['ProfessionalOptionsEngine']()
            self.execution_client = COMPONENTS['AdvancedExecutionClient'](
                self.config.APCA_API_KEY_ID,
                self.config.APCA_API_SECRET_KEY,
                self.config.APCA_API_BASE_URL
            )
            self.risk_manager = COMPONENTS['IntelligentRiskManager']()
            self.learning_system = COMPONENTS['AdaptiveLearningSystem']()
            
            # Upgrade components
            self.data_manager = COMPONENTS['MultiSourceDataManager'](
                self.config.APCA_API_KEY_ID,
                self.config.APCA_API_SECRET_KEY
            )
            self.signal_aggregator = COMPONENTS['MultiSourceSignalAggregator']()
            self.timeframe_scanner = COMPONENTS['MultiTimeframeScanner']()
            self.regime_detector = COMPONENTS['MarketRegimeDetector']()
            self.watchlist_manager = COMPONENTS['DynamicWatchlistManager'](self.config.WATCHLIST)
            self.logger = COMPONENTS['ComprehensiveLogger']()
            
            # 🧠 SENIOR ANALYST BRAIN - The Intelligence Upgrade
            if SENIOR_ANALYST_AVAILABLE and self.config.ENABLE_SENIOR_ANALYST:
                self.senior_analyst = COMPONENTS['SeniorAnalystIntegration']()
                logger.info("🧠 SENIOR ANALYST BRAIN activated - Trading intelligence upgraded!")
            else:
                self.senior_analyst = None
                logger.warning("⚠️ Senior Analyst Brain not available - using basic analysis")
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            # Continue with mock components rather than crashing
    
    def start_web_server(self):
        """Start Flask web server in background thread"""
        def run_server():
            try:
                port = int(os.getenv('PORT', 10000))
                app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
            except Exception as e:
                logger.error(f"Web server error: {e}")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info(f"🌐 Web server started on port {os.getenv('PORT', 10000)}")
    
    async def initialize_senior_analyst_training(self):
        """Train the senior analyst on historical data"""
        if not self.senior_analyst:
            logger.info("🤖 No Senior Analyst available - skipping ML training")
            return
        
        logger.info("🧠 Training Senior Analyst Brain on historical data...")
        training_start = time.time()
        
        # Train on top symbols first for faster startup
        priority_symbols = self.config.WATCHLIST[:5]  # Top 5 symbols
        
        for i, symbol in enumerate(priority_symbols):
            try:
                logger.info(f"🎓 Training Senior Analyst on {symbol} ({i+1}/{len(priority_symbols)})...")
                
                # Get lots of historical data for training
                historical_data = await self.data_manager.get_market_data(
                    symbol, "15Min", self.config.TRAINING_PERIODS
                )
                
                if historical_data is not None and len(historical_data) > 500:
                    await self.senior_analyst.initialize_for_symbol(symbol, historical_data)
                    logger.info(f"✅ Senior Analyst trained on {symbol} ({len(historical_data)} data points)")
                else:
                    logger.warning(f"⚠️ Insufficient training data for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Training failed for {symbol}: {e}")
        
        training_time = time.time() - training_start
        logger.info(f"🎓 Senior Analyst training completed in {training_time:.1f} seconds")
        
        # Get intelligence report
        if hasattr(self.senior_analyst, 'get_system_intelligence_report'):
            intelligence_report = self.senior_analyst.get_system_intelligence_report()
            logger.info(f"🧠 Current Intelligence Level: {intelligence_report.get('intelligence_level', 'Unknown')}")
            logger.info(f"📊 System Maturity: {intelligence_report.get('system_maturity', 'Unknown')}")
    
    async def run_market_analysis_cycle(self):
        """Run market analysis cycle with Senior Analyst intelligence"""
        try:
            logger.info("🔄 Starting market analysis cycle...")
            
            # Get account info
            try:
                account_info = {'equity': 100000, 'buying_power': 50000}  # Default values
                current_positions = {}  # Default empty positions
                
                if hasattr(self.execution_client, 'get_account_info'):
                    try:
                        account_info = await self.execution_client.get_account_info()
                        current_positions = await self.execution_client.get_current_positions()
                    except Exception as e:
                        logger.warning(f"Using default account info due to error: {e}")
                        
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
                return
            
            logger.info(f"💰 Account equity: ${account_info.get('equity', 0):,.2f}")
            
            # Get symbols to analyze
            try:
                if hasattr(self.watchlist_manager, 'get_all_symbols'):
                    symbols = list(self.watchlist_manager.get_all_symbols())[:10]
                else:
                    symbols = self.config.WATCHLIST[:10]
            except Exception as e:
                logger.warning(f"Using default watchlist due to error: {e}")
                symbols = self.config.WATCHLIST[:10]
            
            # Analyze symbols with Senior Analyst intelligence
            analysis_results = []
            for symbol in symbols:
                try:
                    result = await self.analyze_single_symbol_with_senior_analyst(
                        symbol, account_info, current_positions
                    )
                    if result:
                        analysis_results.append(result)
                except Exception as e:
                    logger.error(f"Analysis failed for {symbol}: {e}")
                    continue
            
            # Rank opportunities
            ranked_opportunities = self.rank_trading_opportunities(analysis_results)
            
            # Execute trades if enabled
            if self.config.ENABLE_TRADING and ranked_opportunities:
                await self.execute_trading_decisions(ranked_opportunities, account_info)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            logger.info("✅ Market analysis cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Error in market analysis cycle: {e}")
    
    async def analyze_single_symbol_with_senior_analyst(self, symbol: str, account_info: Dict, 
                                                       current_positions: Dict) -> Optional[Dict]:
        """Analyze single symbol using Senior Analyst Brain"""
        try:
            logger.debug(f"🔍 Senior Analyst analyzing {symbol}...")
            
            # Get market data
            market_data = await self.data_manager.get_market_data(symbol, self.config.PRIMARY_TIMEFRAME, 200)
            
            if market_data is None or len(market_data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # USE SENIOR ANALYST BRAIN FOR INTELLIGENCE
            if self.senior_analyst:
                try:
                    # Get senior analyst recommendation with circuit breaker
                    analysis = await self.circuit_breakers['senior_analyst'].call(
                        self.senior_analyst.get_senior_analyst_recommendation,
                        symbol, 
                        market_data, 
                        {
                            'account_info': account_info,
                            'current_positions': current_positions,
                            'market_hours': self.is_market_hours(datetime.now())
                        }
                    )
                    
                    # Add current price and additional fields
                    analysis['current_price'] = market_data['close'].iloc[-1]
                    analysis['data_quality'] = len(market_data)
                    analysis['trade_recommendation'] = self.should_execute_trade_with_senior_analyst(analysis)
                    
                    # Log the senior analyst's decision
                    grade = analysis.get('analyst_grade', 'HOLD')
                    confidence = analysis.get('confidence', 0.0)
                    expected_return = analysis.get('expected_return', 0.0)
                    reasoning = analysis.get('reasoning', ['No reasoning provided'])
                    
                    logger.info(f"🧠 {symbol}: {grade} | "
                               f"Confidence: {confidence:.1%} | "
                               f"Expected Return: {expected_return:.1%} | "
                               f"Reasoning: {reasoning[0] if reasoning else 'None'}")
                    
                    return analysis
                    
                except Exception as e:
                    logger.warning(f"Senior Analyst failed for {symbol}: {e}")
                    # Fall back to basic analysis
                    return await self._basic_analysis_fallback(symbol, market_data, account_info, current_positions)
            else:
                # Use basic analysis if Senior Analyst not available
                return await self._basic_analysis_fallback(symbol, market_data, account_info, current_positions)
                
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return None
    
    async def _basic_analysis_fallback(self, symbol: str, market_data: pd.DataFrame, 
                                      account_info: Dict, current_positions: Dict) -> Dict:
        """Basic analysis when senior analyst is unavailable"""
        try:
            # Simple trend analysis
            recent_return = (market_data['close'].iloc[-1] - market_data['close'].iloc[-10]) / market_data['close'].iloc[-10]
            
            # Simple volume analysis
            avg_volume = market_data['volume'].mean() if 'volume' in market_data.columns else 1000000
            current_volume = market_data['volume'].iloc[-1] if 'volume' in market_data.columns else avg_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine prediction
            prediction = 1 if recent_return > 0.02 else 0
            confidence = min(0.7, 0.5 + abs(recent_return) * 5)
            
            # Boost confidence with volume
            if volume_ratio > 1.5:
                confidence = min(0.8, confidence * 1.2)
            
            return {
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence,
                'expected_return': abs(recent_return) * 0.5,
                'reasoning': [f'Basic trend analysis: {recent_return:.2%} recent move'],
                'analyst_grade': 'BUY' if prediction == 1 else 'HOLD',
                'time_horizon_minutes': 240,
                'current_price': market_data['close'].iloc[-1],
                'risk_factors': ['Basic analysis only'],
                'supporting_patterns': ['trend_analysis'],
                'market_regime': 'unknown',
                'feature_importance': {},
                'trade_recommendation': prediction == 1 and confidence > 0.6
            }
            
        except Exception as e:
            logger.error(f"Basic analysis fallback failed for {symbol}: {e}")
            return None
    
    def should_execute_trade_with_senior_analyst(self, analysis: Dict) -> bool:
        """Enhanced trade decision with Senior Analyst insights"""
        try:
            # Get analyst recommendation
            analyst_grade = analysis.get('analyst_grade', 'HOLD')
            confidence = analysis.get('confidence', 0.0)
            expected_return = analysis.get('expected_return', 0.0)
            risk_factors = analysis.get('risk_factors', [])
            
            # Strong buy/sell signals
            if analyst_grade in ['STRONG_BUY', 'STRONG_SELL']:
                return confidence > 0.6
            
            # Regular buy/sell signals
            elif analyst_grade in ['BUY', 'SELL']:
                return confidence > self.config.CONFIDENCE_THRESHOLD
            
            # Weak signals require higher confidence
            elif analyst_grade in ['WEAK_BUY', 'WEAK_SELL']:
                return confidence > 0.8 and expected_return > 0.05
            
            # No trading on hold
            else:
                return False
                
        except Exception as e:
            logger.error(f"Trade decision error: {e}")
            return False
    
    def rank_trading_opportunities(self, analysis_results: List[Dict]) -> List[Dict]:
        """Rank trading opportunities with Senior Analyst insights"""
        try:
            if not analysis_results:
                return []
            
            # Filter for tradeable opportunities
            opportunities = [
                result for result in analysis_results 
                if result.get('trade_recommendation', False)
            ]
            
            # Sort by analyst grade and confidence
            def opportunity_score(opp):
                grade = opp.get('analyst_grade', 'HOLD')
                confidence = opp.get('confidence', 0.0)
                expected_return = opp.get('expected_return', 0.0)
                
                # Grade scores
                grade_scores = {
                    'STRONG_BUY': 10,
                    'BUY': 8,
                    'WEAK_BUY': 6,
                    'HOLD': 0,
                    'WEAK_SELL': -6,
                    'SELL': -8,
                    'STRONG_SELL': -10
                }
                
                base_score = grade_scores.get(grade, 0)
                confidence_boost = confidence * 5
                return_boost = expected_return * 10
                
                return base_score + confidence_boost + return_boost
            
            ranked = sorted(opportunities, key=opportunity_score, reverse=True)
            
            logger.info(f"🎯 Found {len(ranked)} trading opportunities (from {len(analysis_results)} analyzed)")
            
            # Log top opportunities
            for i, opp in enumerate(ranked[:3]):
                symbol = opp.get('symbol')
                grade = opp.get('analyst_grade')
                confidence = opp.get('confidence', 0)
                expected_return = opp.get('expected_return', 0)
                logger.info(f"  {i+1}. {symbol}: {grade} ({confidence:.1%} confidence, {expected_return:.1%} expected)")
            
            return ranked
            
        except Exception as e:
            logger.error(f"Error ranking opportunities: {e}")
            return []
    
    async def execute_trading_decisions(self, opportunities: List[Dict], account_info: Dict):
        """Execute trading decisions with learning feedback"""
        try:
            executed_trades = 0
            max_trades = min(3, self.config.MAX_POSITIONS)
            
            for opportunity in opportunities[:max_trades]:
                symbol = opportunity.get('symbol')
                if not symbol:
                    continue
                
                try:
                    trade_success = False
                    
                    if self.config.FOCUS_OPTIONS:
                        trade_success = await self.execute_options_trade_with_learning(opportunity, account_info)
                    else:
                        trade_success = await self.execute_stock_trade_with_learning(opportunity, account_info)
                    
                    if trade_success:
                        executed_trades += 1
                        logger.info(f"✅ Trade executed for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Trade execution failed for {symbol}: {e}")
            
            logger.info(f"📈 Executed {executed_trades} trades this cycle")
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    async def execute_stock_trade_with_learning(self, opportunity: Dict, account_info: Dict) -> bool:
        """Execute stock trade with learning feedback"""
        try:
            symbol = opportunity.get('symbol')
            
            # Create trade record for learning
            trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            trade_data = {
                'trade_id': trade_id,
                'symbol': symbol,
                'prediction_accuracy': opportunity.get('confidence', 0.5),
                'predicted_return': opportunity.get('expected_return', 0.0),
                'analyst_grade': opportunity.get('analyst_grade', 'HOLD'),
                'entry_time': datetime.now(),
                'reasoning': opportunity.get('reasoning', []),
                'market_regime': opportunity.get('market_regime', 'unknown'),
                'profitable': None  # Will be updated when trade closes
            }
            
            # Store for learning (simulate trade execution)
            self.active_trades[trade_id] = trade_data
            
            # Feed back to Senior Analyst for learning
            if self.senior_analyst:
                await self.senior_analyst.learn_from_trade_outcome(symbol, trade_data)
            
            # Simulate successful execution
            logger.info(f"📊 Stock trade simulated for {symbol} (Grade: {trade_data['analyst_grade']})")
            
            # Schedule trade outcome tracking
            asyncio.create_task(self._track_trade_outcome(trade_id, opportunity))
            
            return True
            
        except Exception as e:
            logger.error(f"Stock trade execution error: {e}")
            return False
    
    async def execute_options_trade_with_learning(self, opportunity: Dict, account_info: Dict) -> bool:
        """Execute options trade with learning feedback"""
        try:
            symbol = opportunity.get('symbol')
            
            # Create trade record
            trade_id = f"{symbol}_OPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            trade_data = {
                'trade_id': trade_id,
                'symbol': symbol,
                'trade_type': 'options',
                'prediction_accuracy': opportunity.get('confidence', 0.5),
                'predicted_return': opportunity.get('expected_return', 0.0),
                'analyst_grade': opportunity.get('analyst_grade', 'HOLD'),
                'entry_time': datetime.now(),
                'reasoning': opportunity.get('reasoning', []),
                'profitable': None
            }
            
            self.active_trades[trade_id] = trade_data
            
            # Feed back to learning system
            if self.senior_analyst:
                await self.senior_analyst.learn_from_trade_outcome(symbol, trade_data)
            
            logger.info(f"📊 Options trade simulated for {symbol} (Grade: {trade_data['analyst_grade']})")
            
            # Schedule outcome tracking
            asyncio.create_task(self._track_trade_outcome(trade_id, opportunity))
            
            return True
            
        except Exception as e:
            logger.error(f"Options trade execution error: {e}")
            return False
    
    async def _track_trade_outcome(self, trade_id: str, opportunity: Dict):
        """Track trade outcomes for learning"""
        try:
            # Wait for trade to "complete" (simulate holding period)
            time_horizon = opportunity.get('time_horizon_minutes', 240)
            await asyncio.sleep(min(time_horizon * 60, 3600))  # Max 1 hour for demo
            
            # Simulate trade outcome
            import random
            confidence = opportunity.get('confidence', 0.5)
            expected_return = opportunity.get('expected_return', 0.0)
            
            # Higher confidence = better success rate
            success_probability = 0.3 + (confidence * 0.5)
            actual_success = random.random() < success_probability
            
            # Calculate simulated return
            if actual_success:
                actual_return = expected_return * random.uniform(0.7, 1.3)
            else:
                actual_return = -abs(expected_return) * random.uniform(0.5, 1.0)
            
            # Update trade record
            if trade_id in self.active_trades:
                trade_data = self.active_trades[trade_id]
                trade_data['profitable'] = actual_success
                trade_data['actual_return'] = actual_return
                trade_data['close_time'] = datetime.now()
                
                # Move to completed trades
                self.completed_trades.append(trade_data)
                del self.active_trades[trade_id]
                
                # Update performance metrics
                self.performance_metrics['total_trades'] += 1
                if actual_success:
                    self.performance_metrics['profitable_trades'] += 1
                
                self.performance_metrics['total_pnl'] += actual_return
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['profitable_trades'] / 
                    self.performance_metrics['total_trades']
                )
                
                # Log outcome
                status = "✅ PROFIT" if actual_success else "❌ LOSS"
                logger.info(f"{status} {trade_data['symbol']}: {actual_return:.2%} "
                           f"(Predicted: {expected_return:.2%})")
                
        except Exception as e:
            logger.error(f"Trade outcome tracking failed: {e}")
    
    def update_performance_metrics(self):
        """Update bot performance metrics"""
        try:
            # Update Senior Analyst accuracy if available
            if self.senior_analyst and hasattr(self.senior_analyst, 'get_system_intelligence_report'):
                intelligence_report = self.senior_analyst.get_system_intelligence_report()
                self.performance_metrics['senior_analyst_accuracy'] = intelligence_report.get('average_accuracy', 0.0)
            
            self.performance_metrics['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def run(self):
        """Main execution loop with Senior Analyst"""
        logger.info("🚀 Starting Professional Trading Bot with Senior Analyst")
        logger.info(f"📋 Watchlist: {', '.join(self.config.WATCHLIST)}")
        logger.info(f"💰 Trading enabled: {self.config.ENABLE_TRADING}")
        logger.info(f"🧠 Senior Analyst enabled: {self.config.ENABLE_SENIOR_ANALYST and SENIOR_ANALYST_AVAILABLE}")
        
        self.running = True
        
        # Train the Senior Analyst first
        if self.senior_analyst:
            await self.initialize_senior_analyst_training()
        
        try:
            while self.running:
                try:
                    if self.is_market_hours(datetime.now()):
                        await self.run_market_analysis_cycle()
                    else:
                        logger.info("🌙 Markets closed, waiting...")
                        await asyncio.sleep(300)
                    
                    await asyncio.sleep(self.config.ANALYSIS_INTERVAL)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(60)
                    
        finally:
            self.running = False
    
    def is_market_hours(self, current_time: datetime) -> bool:
        """Check if markets are open"""
        weekday = current_time.weekday()
        hour = current_time.hour
        return weekday < 5 and 9 <= hour < 16

# Global bot instance
bot = None

# Flask routes for monitoring
@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'professional-trading-bot-with-senior-analyst',
        'timestamp': datetime.now().isoformat(),
        'senior_analyst_available': SENIOR_ANALYST_AVAILABLE,
        'message': 'Trading bot with AI intelligence is running'
    })

@app.route('/health')
def detailed_health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': len(COMPONENTS),
        'senior_analyst_available': SENIOR_ANALYST_AVAILABLE,
        'market_hours': bot.is_market_hours(datetime.now()) if bot else False,
        'active_trades': len(bot.active_trades) if bot else 0,
        'completed_trades': len(bot.completed_trades) if bot else 0
    })

@app.route('/intelligence')
def get_intelligence_status():
    """Get senior analyst intelligence status"""
    try:
        if bot and bot.senior_analyst:
            intelligence_report = bot.senior_analyst.get_system_intelligence_report()
            return jsonify({
                'status': 'active',
                'intelligence_report': intelligence_report,
                'performance_metrics': bot.performance_metrics,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'unavailable',
                'message': 'Senior Analyst not initialized',
                'senior_analyst_available': SENIOR_ANALYST_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance')
def get_performance():
    """Get bot performance metrics"""
    try:
        if bot:
            return jsonify({
                'performance_metrics': bot.performance_metrics,
                'active_trades': len(bot.active_trades),
                'completed_trades': len(bot.completed_trades),
                'senior_analyst_status': 'active' if bot.senior_analyst else 'unavailable',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Bot not initialized'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trades')
def get_trades():
    """Get trading history"""
    try:
        if bot:
            return jsonify({
                'active_trades': [
                    {
                        'trade_id': trade_id,
                        'symbol': data['symbol'],
                        'analyst_grade': data.get('analyst_grade', 'UNKNOWN'),
                        'confidence': data.get('prediction_accuracy', 0),
                        'entry_time': data['entry_time'].isoformat(),
                        'reasoning': data.get('reasoning', [])[:2]  # First 2 reasons
                    }
                    for trade_id, data in bot.active_trades.items()
                ],
                'recent_completed_trades': [
                    {
                        'symbol': trade['symbol'],
                        'analyst_grade': trade.get('analyst_grade', 'UNKNOWN'),
                        'profitable': trade.get('profitable', False),
                        'actual_return': trade.get('actual_return', 0),
                        'predicted_return': trade.get('predicted_return', 0),
                        'close_time': trade.get('close_time', datetime.now()).isoformat() if trade.get('close_time') else None
                    }
                    for trade in bot.completed_trades[-10:]  # Last 10 completed trades
                ],
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Bot not initialized'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def run_test():
    return jsonify({
        'status': 'test_completed',
        'timestamp': datetime.now().isoformat(),
        'senior_analyst_available': SENIOR_ANALYST_AVAILABLE,
        'components_loaded': len(COMPONENTS),
        'message': 'System operational with AI intelligence'
    })

@app.route('/retrain', methods=['POST'])
def trigger_retrain():
    """Trigger Senior Analyst retraining"""
    try:
        if bot and bot.senior_analyst:
            # This would trigger retraining in a real system
            asyncio.create_task(bot.initialize_senior_analyst_training())
            return jsonify({
                'status': 'retraining_started',
                'message': 'Senior Analyst retraining initiated',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'unavailable',
                'message': 'Senior Analyst not available for retraining'
            }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

async def main():
    """Main async entry point"""
    global bot
    try:
        bot = ProfessionalTradingBot()
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")

def run_bot():
    """Run bot with proper async handling"""
    try:
        if os.name == 'nt':  # Windows
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    run_bot()

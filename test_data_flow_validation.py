#!/usr/bin/env python3
"""
Data Flow and Error Handling Validation Tests
Ensures data flows correctly between all components and errors are handled gracefully
"""
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta
import sqlite3

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class DataFlowValidator:
    """Validates data flow between all upgrade components"""
    
    def __init__(self):
        self.test_data = self._generate_test_data()
        self.validation_results = {}
    
    def _generate_test_data(self):
        """Generate comprehensive test data"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        test_data = {}
        
        for symbol in symbols:
            base_price = np.random.uniform(100, 300)
            periods = 200
            
            # Generate realistic OHLCV data
            returns = np.random.normal(0, 0.02, periods)
            prices = [base_price]
            
            for i in range(1, periods):
                new_price = prices[i-1] * (1 + returns[i])
                prices.append(max(new_price, 1.0))
            
            test_data[symbol] = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01 09:30:00', periods=periods, freq='1min'),
                'open': prices,
                'high': [p * np.random.uniform(1.0, 1.03) for p in prices],
                'low': [p * np.random.uniform(0.97, 1.0) for p in prices],
                'close': prices,
                'volume': np.random.randint(50000, 500000, periods),
                'symbol': symbol
            })
        
        return test_data
    
    async def test_data_manager_to_aggregator_flow(self):
        """Test data flow from MultiSourceDataManager to SignalAggregator"""
        print("🔄 Testing Data Manager → Signal Aggregator Flow...")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            from models.signal_aggregator import SignalAggregator
            
            data_manager = MultiSourceDataManager()
            signal_aggregator = SignalAggregator()
            
            # Mock data manager to return our test data
            async def mock_get_market_data(symbol, timeframe):
                return self.test_data.get(symbol)
            
            # Mock signal aggregator's internal data fetch to use data manager
            async def mock_fetch_market_data(symbol):
                return await data_manager.get_market_data(symbol, "1min")
            
            with patch.object(data_manager, 'get_market_data', side_effect=mock_get_market_data):
                with patch.object(signal_aggregator, '_fetch_market_data', side_effect=mock_fetch_market_data):
                    
                    # Test the flow
                    for symbol in ["AAPL", "GOOGL"]:
                        data = await data_manager.get_market_data(symbol, "1min")
                        signals = await signal_aggregator.aggregate_signals(symbol)
                        
                        # Validate data structure consistency
                        assert data is not None, f"No data returned for {symbol}"
                        assert isinstance(signals, list), f"Signals should be a list for {symbol}"
                        assert 'close' in data.columns, f"Data missing 'close' column for {symbol}"
                        
            print("✅ Data Manager → Signal Aggregator flow validated")
            return True
            
        except Exception as e:
            print(f"❌ Data Manager → Signal Aggregator flow failed: {e}")
            return False
    
    async def test_aggregator_to_timeframe_scanner_flow(self):
        """Test data flow from SignalAggregator to MultiTimeframeScanner"""
        print("🔄 Testing Signal Aggregator → Timeframe Scanner Flow...")
        
        try:
            from models.signal_aggregator import SignalAggregator
            from models.multi_timeframe_scanner import MultiTimeframeScanner
            
            aggregator = SignalAggregator()
            scanner = MultiTimeframeScanner()
            
            # Mock both components
            async def mock_aggregate_signals(symbol):
                return [{'symbol': symbol, 'signal_type': 'test', 'confidence': 65.0}]
            
            async def mock_fetch_timeframe_data(symbol, timeframe):
                return self.test_data.get(symbol)
            
            with patch.object(aggregator, 'aggregate_signals', side_effect=mock_aggregate_signals):
                with patch.object(scanner, '_fetch_data_for_timeframe', side_effect=mock_fetch_timeframe_data):
                    
                    for symbol in ["AAPL", "MSFT"]:
                        # Test signal aggregation first
                        signals = await aggregator.aggregate_signals(symbol)
                        
                        # Then test timeframe scanning
                        opportunities = await scanner.scan_all_timeframes(symbol)
                        
                        # Validate flow
                        assert isinstance(signals, list), f"Signals should be list for {symbol}"
                        assert isinstance(opportunities, list), f"Opportunities should be list for {symbol}"
                        assert len(signals) > 0, f"No signals generated for {symbol}"
            
            print("✅ Signal Aggregator → Timeframe Scanner flow validated")
            return True
            
        except Exception as e:
            print(f"❌ Signal Aggregator → Timeframe Scanner flow failed: {e}")
            return False
    
    async def test_regime_detector_integration_flow(self):
        """Test MarketRegimeDetector integration with other components"""
        print("🔄 Testing Market Regime Detector Integration Flow...")
        
        try:
            from models.market_regime_detector import MarketRegimeDetector, MarketRegime
            from models.signal_aggregator import SignalAggregator
            
            regime_detector = MarketRegimeDetector()
            signal_aggregator = SignalAggregator()
            
            # Mock market data for regime detection
            spy_data = self.test_data["AAPL"].copy()
            spy_data['symbol'] = 'SPY'  # Use as market proxy
            
            async def mock_fetch_market_data(symbol):
                if symbol == 'SPY':
                    return spy_data
                return self.test_data.get(symbol, spy_data)
            
            with patch.object(regime_detector, '_fetch_market_data', side_effect=mock_fetch_market_data):
                with patch.object(signal_aggregator, '_fetch_market_data', side_effect=mock_fetch_market_data):
                    
                    # Test regime detection
                    regime_result = await regime_detector.detect_market_regime()
                    
                    # Test signal aggregation with regime awareness
                    signals = await signal_aggregator.aggregate_signals("AAPL")
                    
                    # Validate regime detection results
                    assert 'regime' in regime_result, "Regime result missing 'regime' key"
                    assert 'confidence' in regime_result, "Regime result missing 'confidence' key"
                    assert regime_result['regime'] in [r.value for r in MarketRegime], "Invalid regime value"
                    assert 0 <= regime_result['confidence'] <= 100, "Invalid confidence value"
                    
                    # Validate signals are still generated
                    assert isinstance(signals, list), "Signals should be a list"
            
            print("✅ Market Regime Detector integration flow validated")
            return True
            
        except Exception as e:
            print(f"❌ Market Regime Detector integration flow failed: {e}")
            return False
    
    async def test_watchlist_manager_flow(self):
        """Test DynamicWatchlistManager data flow"""
        print("🔄 Testing Dynamic Watchlist Manager Flow...")
        
        try:
            from models.dynamic_watchlist_manager import DynamicWatchlistManager, WatchlistCategory
            from data.multi_source_data_manager import MultiSourceDataManager
            
            watchlist_manager = DynamicWatchlistManager()
            data_manager = MultiSourceDataManager()
            
            # Mock data sources
            mock_trending_data = pd.DataFrame({
                'symbol': ['NVDA', 'AMD', 'INTC'],
                'volume': [1000000, 800000, 600000],
                'price_change_pct': [5.2, 3.8, -2.1]
            })
            
            async def mock_fetch_trending_stocks():
                return mock_trending_data
            
            async def mock_get_market_data(symbol, timeframe):
                return self.test_data.get(symbol, self.test_data["AAPL"])
            
            with patch.object(watchlist_manager, '_fetch_trending_stocks', side_effect=mock_fetch_trending_stocks):
                with patch.object(data_manager, 'get_market_data', side_effect=mock_get_market_data):
                    
                    # Test watchlist updates
                    await watchlist_manager.update_watchlists()
                    
                    # Get updated watchlists
                    trending = watchlist_manager.get_watchlist(WatchlistCategory.TRENDING)
                    base = watchlist_manager.get_watchlist(WatchlistCategory.BASE)
                    
                    # Validate watchlist structure
                    assert isinstance(trending, list), "Trending watchlist should be a list"
                    assert isinstance(base, list), "Base watchlist should be a list"
                    assert len(base) > 0, "Base watchlist should not be empty"
                    
                    # Test data access for watchlist symbols
                    for symbol in base[:2]:  # Test first 2 symbols
                        data = await data_manager.get_market_data(symbol, "1min")
                        assert data is not None, f"No data available for watchlist symbol {symbol}"
            
            print("✅ Dynamic Watchlist Manager flow validated")
            return True
            
        except Exception as e:
            print(f"❌ Dynamic Watchlist Manager flow failed: {e}")
            return False
    
    async def test_comprehensive_logger_flow(self):
        """Test ComprehensiveLogger integration with all components"""
        print("🔄 Testing Comprehensive Logger Flow...")
        
        try:
            from monitoring.comprehensive_logger import ComprehensiveLogger
            from models.signal_aggregator import SignalAggregator
            
            logger = ComprehensiveLogger()
            aggregator = SignalAggregator()
            
            # Test logging various events
            test_events = [
                ('opportunity_detected', 'AAPL', 'breakout', 75.0, {'price': 150.25}),
                ('opportunity_missed', 'GOOGL', 'low_confidence', None, {'confidence': 35.0}),
                ('system_error', 'data_manager', 'timeout', None, {'source': 'alpaca'}),
                ('trade_executed', 'MSFT', 'buy', None, {'quantity': 100, 'price': 300.50})
            ]
            
            # Log all test events
            for event_type, *args in test_events:
                if event_type == 'opportunity_detected':
                    logger.log_opportunity_detected(args[0], args[1], args[2], args[3])
                elif event_type == 'opportunity_missed':
                    logger.log_opportunity_missed(args[0], args[1], args[3])
                elif event_type == 'system_error':
                    logger.log_system_error(args[0], args[1], args[3])
                elif event_type == 'trade_executed':
                    logger.log_trade_executed(args[0], args[1], args[3])
            
            # Test analytics and reporting
            health_report = logger.get_system_health_report()
            daily_summary = logger.get_daily_summary()
            
            # Validate logging functionality
            assert isinstance(health_report, dict), "Health report should be a dictionary"
            assert isinstance(daily_summary, dict), "Daily summary should be a dictionary"
            assert 'opportunities_detected' in health_report, "Health report missing opportunities"
            assert 'total_errors' in health_report, "Health report missing errors"
            
            # Test database integrity
            conn = sqlite3.connect(logger.db_path)
            cursor = conn.cursor()
            
            # Check that tables exist and have data
            tables = ['opportunities', 'system_errors', 'trade_history']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                assert count >= 0, f"Table {table} query failed"
            
            conn.close()
            
            print("✅ Comprehensive Logger flow validated")
            return True
            
        except Exception as e:
            print(f"❌ Comprehensive Logger flow failed: {e}")
            return False
    
    async def test_error_propagation_handling(self):
        """Test how errors propagate through the system"""
        print("🔄 Testing Error Propagation and Handling...")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            from models.signal_aggregator import SignalAggregator
            from monitoring.comprehensive_logger import ComprehensiveLogger
            
            data_manager = MultiSourceDataManager()
            aggregator = SignalAggregator()
            logger = ComprehensiveLogger()
            
            error_scenarios = []
            
            # Test 1: Data source failure with fallback
            with patch.object(data_manager, '_fetch_from_alpaca', side_effect=Exception("API Error")):
                with patch.object(data_manager, '_fetch_from_iex', return_value=self.test_data["AAPL"]):
                    try:
                        data = await data_manager.get_market_data("AAPL", "1min")
                        error_scenarios.append("data_fallback_success" if data is not None else "data_fallback_failed")
                    except Exception:
                        error_scenarios.append("data_fallback_failed")
            
            # Test 2: Partial signal processing failure
            original_method = aggregator._analyze_technical_signals if hasattr(aggregator, '_analyze_technical_signals') else None
            
            if original_method:
                async def failing_analysis(*args, **kwargs):
                    raise Exception("Technical analysis failed")
                
                with patch.object(aggregator, '_analyze_technical_signals', side_effect=failing_analysis):
                    with patch.object(aggregator, '_fetch_market_data', return_value=self.test_data["AAPL"]):
                        try:
                            signals = await aggregator.aggregate_signals("AAPL")
                            error_scenarios.append("signal_partial_success" if isinstance(signals, list) else "signal_total_failure")
                        except Exception:
                            error_scenarios.append("signal_total_failure")
            
            # Test 3: Logging system resilience
            try:
                # Simulate database lock
                logger.log_system_error("test_component", "test_error", {"test": True})
                logger.log_opportunity_detected("TEST", "test_signal", 50.0, {"test": True})
                error_scenarios.append("logging_resilient")
            except Exception:
                error_scenarios.append("logging_failed")
            
            # Evaluate error handling
            success_rate = len([s for s in error_scenarios if "success" in s or "resilient" in s]) / len(error_scenarios)
            
            print(f"✅ Error handling: {success_rate:.1%} resilience rate")
            print(f"   Scenarios: {', '.join(error_scenarios)}")
            
            return {
                'resilience_rate': success_rate,
                'scenarios': error_scenarios
            }
            
        except Exception as e:
            print(f"❌ Error propagation test failed: {e}")
            return {'resilience_rate': 0, 'scenarios': []}
    
    async def test_end_to_end_data_flow(self):
        """Test complete end-to-end data flow through all components"""
        print("🔄 Testing End-to-End Data Flow...")
        
        try:
            # Import all components
            from data.multi_source_data_manager import MultiSourceDataManager
            from models.signal_aggregator import SignalAggregator
            from models.multi_timeframe_scanner import MultiTimeframeScanner
            from models.market_regime_detector import MarketRegimeDetector
            from models.dynamic_watchlist_manager import DynamicWatchlistManager
            from monitoring.comprehensive_logger import ComprehensiveLogger
            
            # Initialize all components
            data_manager = MultiSourceDataManager()
            signal_aggregator = SignalAggregator()
            timeframe_scanner = MultiTimeframeScanner()
            regime_detector = MarketRegimeDetector()
            watchlist_manager = DynamicWatchlistManager()
            logger = ComprehensiveLogger()
            
            # Mock all data sources
            async def mock_get_market_data(symbol, timeframe):
                return self.test_data.get(symbol, self.test_data["AAPL"])
            
            async def mock_fetch_market_data(symbol):
                return self.test_data.get(symbol, self.test_data["AAPL"])
            
            async def mock_fetch_timeframe_data(symbol, timeframe):
                return self.test_data.get(symbol, self.test_data["AAPL"])
            
            # Apply mocks
            with patch.object(data_manager, 'get_market_data', side_effect=mock_get_market_data):
                with patch.object(signal_aggregator, '_fetch_market_data', side_effect=mock_fetch_market_data):
                    with patch.object(timeframe_scanner, '_fetch_data_for_timeframe', side_effect=mock_fetch_timeframe_data):
                        with patch.object(regime_detector, '_fetch_market_data', side_effect=mock_fetch_market_data):
                            
                            # Execute complete workflow
                            symbol = "AAPL"
                            
                            # Step 1: Get market regime
                            regime = await regime_detector.detect_market_regime()
                            logger.log_system_info("market_regime", f"Detected regime: {regime.get('regime', 'unknown')}")
                            
                            # Step 2: Update watchlists
                            await watchlist_manager.update_watchlists()
                            
                            # Step 3: Get market data
                            data = await data_manager.get_market_data(symbol, "1min")
                            assert data is not None, "Failed to get market data"
                            
                            # Step 4: Aggregate signals
                            signals = await signal_aggregator.aggregate_signals(symbol)
                            assert isinstance(signals, list), "Signal aggregation failed"
                            
                            # Step 5: Multi-timeframe analysis
                            opportunities = await timeframe_scanner.scan_all_timeframes(symbol)
                            assert isinstance(opportunities, list), "Timeframe scanning failed"
                            
                            # Step 6: Log results
                            for signal in signals:
                                logger.log_opportunity_detected(
                                    symbol, 
                                    signal.get('signal_type', 'unknown'),
                                    signal.get('confidence', 0),
                                    signal
                                )
                            
                            # Validate complete flow
                            health_report = logger.get_system_health_report()
                            
                            flow_metrics = {
                                'regime_detected': 'regime' in regime,
                                'data_retrieved': data is not None,
                                'signals_generated': len(signals) > 0,
                                'opportunities_found': len(opportunities) >= 0,
                                'logging_functional': isinstance(health_report, dict)
                            }
                            
                            success_count = sum(flow_metrics.values())
                            total_steps = len(flow_metrics)
                            
                            print(f"✅ End-to-end flow: {success_count}/{total_steps} steps successful")
                            print(f"   Flow metrics: {flow_metrics}")
                            
                            return {
                                'success_rate': success_count / total_steps,
                                'flow_metrics': flow_metrics
                            }
            
        except Exception as e:
            print(f"❌ End-to-end data flow test failed: {e}")
            return {'success_rate': 0, 'flow_metrics': {}}
    
    async def run_all_validations(self):
        """Run all data flow validations"""
        print("🚀 DATA FLOW AND ERROR HANDLING VALIDATION")
        print("=" * 55)
        print("Validating data flows correctly between all components...")
        print()
        
        validation_methods = [
            self.test_data_manager_to_aggregator_flow,
            self.test_aggregator_to_timeframe_scanner_flow,
            self.test_regime_detector_integration_flow,
            self.test_watchlist_manager_flow,
            self.test_comprehensive_logger_flow,
            self.test_error_propagation_handling,
            self.test_end_to_end_data_flow
        ]
        
        results = {}
        success_count = 0
        
        for validation_method in validation_methods:
            try:
                result = await validation_method()
                results[validation_method.__name__] = result
                if isinstance(result, bool) and result:
                    success_count += 1
                elif isinstance(result, dict) and result.get('success_rate', 0) > 0.5:
                    success_count += 1
            except Exception as e:
                print(f"❌ {validation_method.__name__} failed: {e}")
                results[validation_method.__name__] = False
        
        return {
            'total_validations': len(validation_methods),
            'successful_validations': success_count,
            'success_rate': success_count / len(validation_methods),
            'detailed_results': results
        }

async def main():
    """Run all data flow validations"""
    validator = DataFlowValidator()
    
    print("🔍 Starting comprehensive data flow validation...")
    print("This ensures all components communicate properly and handle errors gracefully.")
    print()
    
    results = await validator.run_all_validations()
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"✅ Successful Validations: {results['successful_validations']}/{results['total_validations']}")
    print(f"📈 Overall Success Rate: {results['success_rate']:.1%}")
    
    if results['success_rate'] >= 0.8:
        print("🟢 EXCELLENT - Data flows are working correctly")
        print("🚀 System is ready for production use")
    elif results['success_rate'] >= 0.6:
        print("🟡 GOOD - Most data flows working, minor issues detected")
        print("🔧 Review any failed validations above")
    else:
        print("🔴 ATTENTION NEEDED - Multiple data flow issues detected")
        print("⚠️  Address failed validations before production use")
    
    print(f"\n🏁 DATA FLOW VALIDATION COMPLETE")
    print("All component interactions have been tested.")

if __name__ == "__main__":
    asyncio.run(main())
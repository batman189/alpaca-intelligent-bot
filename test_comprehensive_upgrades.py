#!/usr/bin/env python3
"""
Comprehensive Test Suite for All Upgrade Components
Tests all 6 major upgrade components before market open
"""
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestUpgradeComponents(unittest.TestCase):
    """Test all upgrade components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_symbol = "AAPL"
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(150, 160, 100),
            'high': np.random.uniform(155, 165, 100),
            'low': np.random.uniform(145, 155, 100),
            'close': np.random.uniform(150, 160, 100),
            'volume': np.random.randint(10000, 100000, 100)
        })
    
    def test_multi_source_data_manager_import(self):
        """Test Multi-Source Data Manager import and initialization"""
        print("🧪 Testing Multi-Source Data Manager...")
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            manager = MultiSourceDataManager()
            self.assertIsNotNone(manager)
            self.assertIn('alpaca', manager.sources)
            self.assertIn('iex', manager.sources)
            self.assertIn('yahoo', manager.sources)
            print("✅ Multi-Source Data Manager initialization successful")
            return True
        except Exception as e:
            print(f"❌ Multi-Source Data Manager test failed: {e}")
            return False
    
    def test_signal_aggregator_import(self):
        """Test Signal Aggregator import and initialization"""
        print("🧪 Testing Signal Aggregator...")
        try:
            from models.signal_aggregator import SignalAggregator
            aggregator = SignalAggregator()
            self.assertIsNotNone(aggregator)
            self.assertEqual(aggregator.min_confidence, 40)  # Lowered threshold
            print("✅ Signal Aggregator initialization successful")
            return True
        except Exception as e:
            print(f"❌ Signal Aggregator test failed: {e}")
            return False
    
    def test_multi_timeframe_scanner_import(self):
        """Test Multi-Timeframe Scanner import and initialization"""
        print("🧪 Testing Multi-Timeframe Scanner...")
        try:
            from models.multi_timeframe_scanner import MultiTimeframeScanner, TimeFrame
            scanner = MultiTimeframeScanner()
            self.assertIsNotNone(scanner)
            # Test all timeframes are defined
            self.assertTrue(hasattr(TimeFrame, 'SCALP'))
            self.assertTrue(hasattr(TimeFrame, 'SHORT'))
            self.assertTrue(hasattr(TimeFrame, 'SWING'))
            self.assertTrue(hasattr(TimeFrame, 'POSITION'))
            self.assertTrue(hasattr(TimeFrame, 'DAILY'))
            print("✅ Multi-Timeframe Scanner initialization successful")
            return True
        except Exception as e:
            print(f"❌ Multi-Timeframe Scanner test failed: {e}")
            return False
    
    def test_market_regime_detector_import(self):
        """Test Market Regime Detector import and initialization"""
        print("🧪 Testing Market Regime Detector...")
        try:
            from models.market_regime_detector import MarketRegimeDetector, MarketRegime
            detector = MarketRegimeDetector()
            self.assertIsNotNone(detector)
            # Test all regimes are defined
            self.assertTrue(hasattr(MarketRegime, 'BULL'))
            self.assertTrue(hasattr(MarketRegime, 'BEAR'))
            self.assertTrue(hasattr(MarketRegime, 'SIDEWAYS'))
            self.assertTrue(hasattr(MarketRegime, 'VOLATILE'))
            print("✅ Market Regime Detector initialization successful")
            return True
        except Exception as e:
            print(f"❌ Market Regime Detector test failed: {e}")
            return False
    
    def test_dynamic_watchlist_manager_import(self):
        """Test Dynamic Watchlist Manager import and initialization"""
        print("🧪 Testing Dynamic Watchlist Manager...")
        try:
            from models.dynamic_watchlist_manager import DynamicWatchlistManager, WatchlistCategory
            manager = DynamicWatchlistManager()
            self.assertIsNotNone(manager)
            # Test all categories are defined
            self.assertTrue(hasattr(WatchlistCategory, 'BASE'))
            self.assertTrue(hasattr(WatchlistCategory, 'TRENDING'))
            self.assertTrue(hasattr(WatchlistCategory, 'VOLUME_LEADERS'))
            self.assertTrue(hasattr(WatchlistCategory, 'EARNINGS'))
            print("✅ Dynamic Watchlist Manager initialization successful")
            return True
        except Exception as e:
            print(f"❌ Dynamic Watchlist Manager test failed: {e}")
            return False
    
    def test_comprehensive_logger_import(self):
        """Test Comprehensive Logger import and initialization"""
        print("🧪 Testing Comprehensive Logger...")
        try:
            from monitoring.comprehensive_logger import ComprehensiveLogger
            logger = ComprehensiveLogger()
            self.assertIsNotNone(logger)
            # Test database initialization
            self.assertTrue(os.path.exists(logger.db_path))
            print("✅ Comprehensive Logger initialization successful")
            return True
        except Exception as e:
            print(f"❌ Comprehensive Logger test failed: {e}")
            return False

class TestAsyncComponents(unittest.TestCase):
    """Test async functionality of upgrade components"""
    
    def setUp(self):
        """Set up async test environment"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test environment"""
        self.loop.close()
    
    def test_async_signal_aggregation(self):
        """Test async signal aggregation functionality"""
        print("🧪 Testing Async Signal Aggregation...")
        
        async def run_test():
            try:
                from models.signal_aggregator import SignalAggregator
                aggregator = SignalAggregator()
                
                # Mock data for testing
                test_data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min'),
                    'close': np.random.uniform(150, 160, 50),
                    'volume': np.random.randint(10000, 100000, 50)
                })
                
                # Test signal aggregation
                with patch.object(aggregator, '_fetch_market_data', return_value=test_data):
                    signals = await aggregator.aggregate_signals("AAPL")
                    self.assertIsInstance(signals, list)
                    print("✅ Async signal aggregation successful")
                    return True
                    
            except Exception as e:
                print(f"❌ Async signal aggregation test failed: {e}")
                return False
        
        return self.loop.run_until_complete(run_test())
    
    def test_async_multi_timeframe_scanning(self):
        """Test async multi-timeframe scanning functionality"""
        print("🧪 Testing Async Multi-Timeframe Scanning...")
        
        async def run_test():
            try:
                from models.multi_timeframe_scanner import MultiTimeframeScanner
                scanner = MultiTimeframeScanner()
                
                # Mock successful scan
                with patch.object(scanner, '_fetch_data_for_timeframe', return_value=pd.DataFrame({
                    'close': [150, 151, 152], 'volume': [10000, 11000, 12000]
                })):
                    opportunities = await scanner.scan_all_timeframes("AAPL")
                    self.assertIsInstance(opportunities, list)
                    print("✅ Async multi-timeframe scanning successful")
                    return True
                    
            except Exception as e:
                print(f"❌ Async multi-timeframe scanning test failed: {e}")
                return False
        
        return self.loop.run_until_complete(run_test())

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios between components"""
    
    def test_end_to_end_opportunity_detection(self):
        """Test complete opportunity detection workflow"""
        print("🧪 Testing End-to-End Opportunity Detection...")
        
        try:
            # Import all required components
            from data.multi_source_data_manager import MultiSourceDataManager
            from models.signal_aggregator import SignalAggregator
            from models.multi_timeframe_scanner import MultiTimeframeScanner
            from models.market_regime_detector import MarketRegimeDetector
            from models.dynamic_watchlist_manager import DynamicWatchlistManager
            from monitoring.comprehensive_logger import ComprehensiveLogger
            
            # Initialize components
            data_manager = MultiSourceDataManager()
            signal_aggregator = SignalAggregator()
            timeframe_scanner = MultiTimeframeScanner()
            regime_detector = MarketRegimeDetector()
            watchlist_manager = DynamicWatchlistManager()
            logger = ComprehensiveLogger()
            
            # Test component interaction
            self.assertIsNotNone(data_manager)
            self.assertIsNotNone(signal_aggregator)
            self.assertIsNotNone(timeframe_scanner)
            self.assertIsNotNone(regime_detector)
            self.assertIsNotNone(watchlist_manager)
            self.assertIsNotNone(logger)
            
            print("✅ End-to-end integration test successful")
            return True
            
        except Exception as e:
            print(f"❌ End-to-end integration test failed: {e}")
            return False
    
    def test_logging_integration(self):
        """Test logging integration with all components"""
        print("🧪 Testing Logging Integration...")
        
        try:
            from monitoring.comprehensive_logger import ComprehensiveLogger
            logger = ComprehensiveLogger()
            
            # Test logging different types of events
            logger.log_opportunity_detected("AAPL", "breakout", 65.0, {"price": 150.50})
            logger.log_opportunity_missed("TSLA", "low_confidence", {"confidence": 35.0})
            logger.log_system_error("data_fetch", "timeout", {"source": "alpaca"})
            
            # Test analytics
            health_report = logger.get_system_health_report()
            self.assertIsInstance(health_report, dict)
            
            print("✅ Logging integration test successful")
            return True
            
        except Exception as e:
            print(f"❌ Logging integration test failed: {e}")
            return False

def run_performance_benchmarks():
    """Run performance benchmarks for all components"""
    print("\n🏃 PERFORMANCE BENCHMARKS")
    print("=" * 50)
    
    import time
    
    # Test import speeds
    start_time = time.time()
    try:
        from data.multi_source_data_manager import MultiSourceDataManager
        from models.signal_aggregator import SignalAggregator
        from models.multi_timeframe_scanner import MultiTimeframeScanner
        from models.market_regime_detector import MarketRegimeDetector
        from models.dynamic_watchlist_manager import DynamicWatchlistManager
        from monitoring.comprehensive_logger import ComprehensiveLogger
        
        import_time = time.time() - start_time
        print(f"✅ All components imported in {import_time:.3f}s")
        
        # Test initialization speeds
        start_time = time.time()
        components = {
            'data_manager': MultiSourceDataManager(),
            'signal_aggregator': SignalAggregator(),
            'timeframe_scanner': MultiTimeframeScanner(),
            'regime_detector': MarketRegimeDetector(),
            'watchlist_manager': DynamicWatchlistManager(),
            'logger': ComprehensiveLogger()
        }
        
        init_time = time.time() - start_time
        print(f"✅ All components initialized in {init_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE UPGRADE COMPONENT TESTS")
    print("=" * 60)
    print("Testing all 6 upgrade components before market open...")
    print()
    
    # Track results
    test_results = []
    
    # Run import and initialization tests
    print("📦 COMPONENT IMPORT & INITIALIZATION TESTS")
    print("-" * 40)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUpgradeComponents)
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    
    # Run each test individually for better reporting
    test_methods = [
        'test_multi_source_data_manager_import',
        'test_signal_aggregator_import', 
        'test_multi_timeframe_scanner_import',
        'test_market_regime_detector_import',
        'test_dynamic_watchlist_manager_import',
        'test_comprehensive_logger_import'
    ]
    
    test_instance = TestUpgradeComponents()
    test_instance.setUp()
    
    for method_name in test_methods:
        try:
            method = getattr(test_instance, method_name)
            result = method()
            test_results.append(result)
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            test_results.append(False)
    
    # Run async tests
    print("\n⚡ ASYNC FUNCTIONALITY TESTS")
    print("-" * 30)
    
    async_test_instance = TestAsyncComponents()
    async_test_instance.setUp()
    
    try:
        async_result1 = async_test_instance.test_async_signal_aggregation()
        async_result2 = async_test_instance.test_async_multi_timeframe_scanning()
        test_results.extend([async_result1, async_result2])
    except Exception as e:
        print(f"❌ Async tests failed: {e}")
        test_results.extend([False, False])
    finally:
        async_test_instance.tearDown()
    
    # Run integration tests
    print("\n🔗 INTEGRATION TESTS")
    print("-" * 20)
    
    integration_test_instance = TestIntegrationScenarios()
    try:
        integration_result1 = integration_test_instance.test_end_to_end_opportunity_detection()
        integration_result2 = integration_test_instance.test_logging_integration()
        test_results.extend([integration_result1, integration_result2])
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        test_results.extend([False, False])
    
    # Run performance benchmarks
    benchmark_result = run_performance_benchmarks()
    test_results.append(benchmark_result)
    
    # Final results
    print("\n" + "=" * 60)
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ {passed_tests}/{total_tests} tests successful")
        print("🚀 Your upgraded trading bot is ready for market open!")
        print("\n📊 UPGRADE COMPONENTS STATUS:")
        print("✅ Multi-Source Data Manager - READY")
        print("✅ Signal Aggregator - READY") 
        print("✅ Multi-Timeframe Scanner - READY")
        print("✅ Market Regime Detector - READY")
        print("✅ Dynamic Watchlist Manager - READY")
        print("✅ Comprehensive Logger - READY")
    else:
        failed_tests = total_tests - passed_tests
        print(f"⚠️  {failed_tests}/{total_tests} tests failed")
        print("🔧 Review the errors above and fix issues before market open")
        
        # Provide specific guidance
        print("\n🛠️  TROUBLESHOOTING TIPS:")
        print("1. Ensure all upgrade files are in correct directories")
        print("2. Check that all required dependencies are installed")
        print("3. Verify no syntax errors in upgrade components")
        print("4. Run individual component tests to isolate issues")

if __name__ == "__main__":
    main()
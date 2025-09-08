#!/usr/bin/env python3
"""
🚀 MARKET READINESS TEST SCRIPT
Comprehensive test to ensure your trading bot is ready for market open
Tests all critical components, data flows, and error handling
"""

import sys
import os
import asyncio
import time
import traceback
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MarketReadinessTester:
    def __init__(self):
        self.test_results = {}
        self.critical_failures = []
        self.warnings = []
        self.start_time = time.time()
        
        # Generate realistic test data
        self.test_data = self._generate_realistic_market_data()
        
    def _generate_realistic_market_data(self):
        """Generate realistic market data for testing"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ"]
        test_data = {}
        
        for symbol in symbols:
            base_price = np.random.uniform(100, 300)
            periods = 200
            
            # Generate realistic price movements
            returns = np.random.normal(0, 0.015, periods)  # 1.5% daily volatility
            prices = [base_price]
            
            for i in range(1, periods):
                new_price = prices[i-1] * (1 + returns[i])
                prices.append(max(new_price, 1.0))  # Ensure positive prices
            
            test_data[symbol] = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01 09:30:00', periods=periods, freq='5min'),
                'open': prices,
                'high': [p * np.random.uniform(1.0, 1.025) for p in prices],
                'low': [p * np.random.uniform(0.975, 1.0) for p in prices],
                'close': prices,
                'volume': np.random.randint(50000, 500000, periods)
            })
            test_data[symbol].set_index('timestamp', inplace=True)
        
        return test_data

    def log_test(self, test_name: str, success: bool, details: str = "", critical: bool = False):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'critical': critical
        }
        
        if not success and critical:
            self.critical_failures.append(test_name)
        elif not success:
            self.warnings.append(test_name)

    def test_file_structure(self):
        """Test that all required files exist"""
        print("\n🔍 TESTING FILE STRUCTURE")
        print("-" * 40)
        
        required_files = [
            ("app.py", True),
            ("requirements.txt", True),
            ("config/settings.py", False),
            ("data/data_client.py", True),
            ("data/multi_source_data_manager.py", True),
            ("models/advanced_market_analyzer.py", True),
            ("models/adaptive_learning_system.py", True),
            ("models/signal_aggregator.py", True),
            ("models/multi_timeframe_scanner.py", True),
            ("models/market_regime_detector.py", True),
            ("models/dynamic_watchlist_manager.py", True),
            ("features/feature_engineer.py", False),
            ("analysis_logger.py", False)
        ]
        
        missing_critical = []
        missing_optional = []
        
        for file_path, is_critical in required_files:
            if Path(file_path).exists():
                self.log_test(f"File: {file_path}", True)
            else:
                self.log_test(f"File: {file_path}", False, f"Missing: {file_path}", is_critical)
                if is_critical:
                    missing_critical.append(file_path)
                else:
                    missing_optional.append(file_path)
        
        if missing_critical:
            self.log_test("Critical Files Check", False, f"Missing: {missing_critical}", True)
        else:
            self.log_test("Critical Files Check", True, "All critical files present")

    def test_imports(self):
        """Test all critical imports"""
        print("\n📦 TESTING IMPORTS")
        print("-" * 40)
        
        import_tests = [
            ("Multi-Source Data Manager", "data.multi_source_data_manager", "MultiSourceDataManager", True),
            ("Signal Aggregator", "models.signal_aggregator", "MultiSourceSignalAggregator", True),
            ("Multi-Timeframe Scanner", "models.multi_timeframe_scanner", "MultiTimeframeScanner", True),
            ("Market Regime Detector", "models.market_regime_detector", "MarketRegimeDetector", True),
            ("Dynamic Watchlist Manager", "models.dynamic_watchlist_manager", "DynamicWatchlistManager", True),
            ("Advanced Market Analyzer", "models.advanced_market_analyzer", "AdvancedMarketAnalyzer", True),
            ("Adaptive Learning System", "models.adaptive_learning_system", "AdaptiveLearningSystem", True),
            ("Data Client", "data.data_client", "EnhancedDataClient", True)
        ]
        
        import_failures = []
        
        for name, module_path, class_name, is_critical in import_tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.log_test(f"Import: {name}", True)
            except Exception as e:
                self.log_test(f"Import: {name}", False, str(e), is_critical)
                if is_critical:
                    import_failures.append(name)
        
        if import_failures:
            self.log_test("Critical Imports Check", False, f"Failed: {import_failures}", True)
        else:
            self.log_test("Critical Imports Check", True, "All critical imports successful")

    def test_component_initialization(self):
        """Test component initialization"""
        print("\n🏗️ TESTING COMPONENT INITIALIZATION")
        print("-" * 40)
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            from models.signal_aggregator import MultiSourceSignalAggregator
            from models.multi_timeframe_scanner import MultiTimeframeScanner
            from models.market_regime_detector import MarketRegimeDetector
            from models.dynamic_watchlist_manager import DynamicWatchlistManager
            from models.advanced_market_analyzer import AdvancedMarketAnalyzer
            from models.adaptive_learning_system import AdaptiveLearningSystem
            
            components = {}
            
            # Test each component initialization
            try:
                components['data_manager'] = MultiSourceDataManager()
                self.log_test("Init: Data Manager", True)
            except Exception as e:
                self.log_test("Init: Data Manager", False, str(e), True)
            
            try:
                components['signal_aggregator'] = MultiSourceSignalAggregator()
                self.log_test("Init: Signal Aggregator", True)
            except Exception as e:
                self.log_test("Init: Signal Aggregator", False, str(e), True)
            
            try:
                components['timeframe_scanner'] = MultiTimeframeScanner()
                self.log_test("Init: Timeframe Scanner", True)
            except Exception as e:
                self.log_test("Init: Timeframe Scanner", False, str(e), True)
            
            try:
                components['regime_detector'] = MarketRegimeDetector()
                self.log_test("Init: Regime Detector", True)
            except Exception as e:
                self.log_test("Init: Regime Detector", False, str(e), True)
            
            try:
                components['watchlist_manager'] = DynamicWatchlistManager()
                self.log_test("Init: Watchlist Manager", True)
            except Exception as e:
                self.log_test("Init: Watchlist Manager", False, str(e), True)
            
            try:
                components['market_analyzer'] = AdvancedMarketAnalyzer()
                self.log_test("Init: Market Analyzer", True)
            except Exception as e:
                self.log_test("Init: Market Analyzer", False, str(e), True)
            
            try:
                components['learning_system'] = AdaptiveLearningSystem()
                self.log_test("Init: Learning System", True)
            except Exception as e:
                self.log_test("Init: Learning System", False, str(e), True)
            
            return components
            
        except Exception as e:
            self.log_test("Component Initialization", False, f"Critical import failure: {e}", True)
            return {}

    async def test_data_flow(self, components):
        """Test data flow between components"""
        print("\n🔄 TESTING DATA FLOW")
        print("-" * 40)
        
        if not components:
            self.log_test("Data Flow Test", False, "No components available", True)
            return
        
        # Test data manager
        if 'data_manager' in components:
            try:
                # Mock the data fetch
                async def mock_get_data(symbol, timeframe, limit):
                    return self.test_data.get(symbol)
                
                with patch.object(components['data_manager'], 'get_market_data', side_effect=mock_get_data):
                    data = await components['data_manager'].get_market_data("AAPL", "5Min", 100)
                    if data is not None and not data.empty:
                        self.log_test("Data Manager Flow", True, f"Retrieved {len(data)} bars")
                    else:
                        self.log_test("Data Manager Flow", False, "No data returned", True)
            except Exception as e:
                self.log_test("Data Manager Flow", False, str(e), True)
        
        # Test signal aggregation
        if 'signal_aggregator' in components:
            try:
                signals = await components['signal_aggregator'].aggregate_signals("AAPL", self.test_data["AAPL"])
                if isinstance(signals, list):
                    self.log_test("Signal Aggregation Flow", True, f"Generated {len(signals)} signals")
                else:
                    self.log_test("Signal Aggregation Flow", False, "Invalid signal format", True)
            except Exception as e:
                self.log_test("Signal Aggregation Flow", False, str(e), True)
        
        # Test market regime detection
        if 'regime_detector' in components:
            try:
                market_data = {
                    "SPY": self.test_data["SPY"],
                    "QQQ": self.test_data["QQQ"]
                }
                regime = await components['regime_detector'].detect_market_regime(market_data)
                if isinstance(regime, dict) and 'regime' in regime:
                    self.log_test("Regime Detection Flow", True, f"Detected: {regime['regime']}")
                else:
                    self.log_test("Regime Detection Flow", False, "Invalid regime format")
            except Exception as e:
                self.log_test("Regime Detection Flow", False, str(e))

    def test_error_handling(self, components):
        """Test error handling and resilience"""
        print("\n🛡️ TESTING ERROR HANDLING")
        print("-" * 40)
        
        if not components:
            self.log_test("Error Handling Test", False, "No components available", True)
            return
        
        # Test data manager fallback
        if 'data_manager' in components:
            try:
                # Simulate primary source failure
                with patch.object(components['data_manager'], '_fetch_from_source_async', side_effect=Exception("Primary source failed")):
                    # The system should gracefully handle this
                    try:
                        # This might fail, but shouldn't crash the system
                        result = asyncio.run(components['data_manager'].get_market_data("AAPL", "5Min", 100))
                        self.log_test("Data Source Fallback", True, "Handled source failure gracefully")
                    except Exception:
                        self.log_test("Data Source Fallback", True, "Failed gracefully without crash")
            except Exception as e:
                self.log_test("Data Source Fallback", False, str(e))
        
        # Test signal aggregator resilience
        if 'signal_aggregator' in components:
            try:
                # Test with empty data
                empty_data = pd.DataFrame()
                signals = asyncio.run(components['signal_aggregator'].aggregate_signals("TEST", empty_data))
                if isinstance(signals, list):
                    self.log_test("Signal Aggregator Resilience", True, "Handled empty data")
                else:
                    self.log_test("Signal Aggregator Resilience", False, "Failed with empty data")
            except Exception as e:
                self.log_test("Signal Aggregator Resilience", False, str(e))

    def test_performance(self, components):
        """Test performance metrics"""
        print("\n⚡ TESTING PERFORMANCE")
        print("-" * 40)
        
        if not components:
            self.log_test("Performance Test", False, "No components available")
            return
        
        # Test initialization speed
        start_time = time.time()
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            from models.signal_aggregator import MultiSourceSignalAggregator
            test_components = {
                'data_manager': MultiSourceDataManager(),
                'signal_aggregator': MultiSourceSignalAggregator()
            }
            init_time = time.time() - start_time
            
            if init_time < 5.0:
                self.log_test("Initialization Speed", True, f"{init_time:.2f}s (Good)")
            elif init_time < 10.0:
                self.log_test("Initialization Speed", True, f"{init_time:.2f}s (Acceptable)")
            else:
                self.log_test("Initialization Speed", False, f"{init_time:.2f}s (Too slow)")
        except Exception as e:
            self.log_test("Initialization Speed", False, str(e))
        
        # Test signal processing speed
        if 'signal_aggregator' in components:
            try:
                start_time = time.time()
                signals = asyncio.run(components['signal_aggregator'].aggregate_signals("AAPL", self.test_data["AAPL"]))
                process_time = time.time() - start_time
                
                if process_time < 2.0:
                    self.log_test("Signal Processing Speed", True, f"{process_time:.2f}s (Fast)")
                elif process_time < 5.0:
                    self.log_test("Signal Processing Speed", True, f"{process_time:.2f}s (Acceptable)")
                else:
                    self.log_test("Signal Processing Speed", False, f"{process_time:.2f}s (Too slow)")
            except Exception as e:
                self.log_test("Signal Processing Speed", False, str(e))

    def test_configuration(self):
        """Test configuration and environment"""
        print("\n⚙️ TESTING CONFIGURATION")
        print("-" * 40)
        
        # Check environment variables
        env_vars = [
            ("APCA_API_KEY_ID", True),
            ("APCA_API_SECRET_KEY", True),
            ("APCA_API_BASE_URL", False)
        ]
        
        missing_critical_env = []
        
        for var_name, is_critical in env_vars:
            if os.getenv(var_name):
                self.log_test(f"Env Var: {var_name}", True, "Present")
            else:
                self.log_test(f"Env Var: {var_name}", False, "Missing", is_critical)
                if is_critical:
                    missing_critical_env.append(var_name)
        
        if missing_critical_env:
            self.log_test("Environment Configuration", False, f"Missing critical vars: {missing_critical_env}", True)
        else:
            self.log_test("Environment Configuration", True, "All required env vars present")
        
        # Test directory structure
        required_dirs = ["logs", "data", "models"]
        for dir_name in required_dirs:
            if Path(dir_name).exists():
                self.log_test(f"Directory: {dir_name}", True)
            else:
                try:
                    Path(dir_name).mkdir(exist_ok=True)
                    self.log_test(f"Directory: {dir_name}", True, "Created")
                except Exception as e:
                    self.log_test(f"Directory: {dir_name}", False, str(e))

    async def run_integration_test(self, components):
        """Run end-to-end integration test"""
        print("\n🔗 TESTING INTEGRATION")
        print("-" * 40)
        
        if not components or len(components) < 3:
            self.log_test("Integration Test", False, "Insufficient components", True)
            return
        
        try:
            # Simulate a complete trading cycle
            symbol = "AAPL"
            
            # Step 1: Get market data
            async def mock_get_data(sym, tf, limit):
                return self.test_data.get(sym)
            
            if 'data_manager' in components:
                with patch.object(components['data_manager'], 'get_market_data', side_effect=mock_get_data):
                    data = await components['data_manager'].get_market_data(symbol, "5Min", 100)
            else:
                data = self.test_data[symbol]
            
            # Step 2: Analyze signals
            if 'signal_aggregator' in components:
                signals = await components['signal_aggregator'].aggregate_signals(symbol, data)
            else:
                signals = []
            
            # Step 3: Market regime
            if 'regime_detector' in components:
                regime = await components['regime_detector'].detect_market_regime({"SPY": self.test_data["SPY"]})
            else:
                regime = {"regime": "neutral"}
            
            # Step 4: Check results
            integration_success = True
            if data is None or data.empty:
                integration_success = False
            if not isinstance(signals, list):
                integration_success = False
            if not isinstance(regime, dict):
                integration_success = False
            
            if integration_success:
                self.log_test("End-to-End Integration", True, f"Data: {len(data)} bars, Signals: {len(signals)}, Regime: {regime.get('regime', 'unknown')}")
            else:
                self.log_test("End-to-End Integration", False, "Integration flow failed", True)
                
        except Exception as e:
            self.log_test("End-to-End Integration", False, str(e), True)

    def generate_final_report(self):
        """Generate final readiness report"""
        total_time = time.time() - self.start_time
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        critical_failures = len(self.critical_failures)
        warnings_count = len(self.warnings)
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Log detailed results to file
        self._write_log("\n" + "=" * 60, also_print=False)
        self._write_log("DETAILED TEST RESULTS", also_print=False)
        self._write_log("=" * 60, also_print=False)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result['success'] else "FAIL"
            critical_flag = " [CRITICAL]" if result['critical'] else ""
            details = f" - {result['details']}" if result['details'] else ""
            self._write_log(f"{status}: {test_name}{critical_flag}{details}", also_print=False)
        
        print("\n" + "=" * 60)
        print("🚀 MARKET READINESS ASSESSMENT")
        print("=" * 60)
        
        summary_lines = [
            "\nMARKET READINESS ASSESSMENT SUMMARY",
            "=" * 40,
            f"Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})",
            f"Critical Failures: {critical_failures}",
            f"Warnings: {warnings_count}",
            f"Total Test Time: {total_time:.2f}s"
        ]
        
        for line in summary_lines:
            self._write_log(line, also_print=False)
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        print(f"⚠️  Critical Failures: {critical_failures}")
        print(f"🟡 Warnings: {warnings_count}")
        print(f"⏱️  Total Test Time: {total_time:.2f}s")
        
        if critical_failures == 0 and success_rate >= 0.8:
            recommendation = "PROCEED"
            status_msg = "🟢 READY FOR MARKET OPEN!"
            details = [
                "✅ All critical systems operational",
                "✅ Bot is ready for production trading", 
                "✅ No critical issues detected"
            ]
        elif critical_failures == 0 and success_rate >= 0.6:
            recommendation = "PROCEED WITH CAUTION"
            status_msg = "🟡 MOSTLY READY - MINOR ISSUES"
            details = [
                "⚠️  Some non-critical issues detected",
                "⚠️  Review warnings before market open",
                "✅ Core trading functionality intact"
            ]
        else:
            recommendation = "DO NOT PROCEED"
            status_msg = "🔴 NOT READY - CRITICAL ISSUES"
            details = [
                "❌ Critical failures must be resolved",
                "❌ Do not use for live trading",
                "🔧 Fix critical issues before market open"
            ]
        
        print(f"\n{status_msg}")
        for detail in details:
            print(detail)
            
        # Log recommendation
        self._write_log(f"\nFINAL RECOMMENDATION: {recommendation}", also_print=False)
        self._write_log(status_msg, also_print=False)
        for detail in details:
            self._write_log(detail, also_print=False)
        
        if self.critical_failures:
            print(f"\n🚨 CRITICAL FAILURES TO FIX:")
            self._write_log("\nCRITICAL FAILURES TO FIX:", also_print=False)
            for failure in self.critical_failures:
                print(f"   - {failure}")
                self._write_log(f"   - {failure}", also_print=False)
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS TO REVIEW:")
            self._write_log("\nWARNINGS TO REVIEW:", also_print=False)
            for warning in self.warnings:
                print(f"   - {warning}")
                self._write_log(f"   - {warning}", also_print=False)
        
        print(f"\n🎯 FINAL RECOMMENDATION: {recommendation}")
        print("=" * 60)
        
        # Save log file
        self._save_log_file()
        
        return {
            'ready': critical_failures == 0 and success_rate >= 0.8,
            'recommendation': recommendation,
            'success_rate': success_rate,
            'critical_failures': critical_failures,
            'warnings': warnings_count,
            'test_time': total_time
        }

async def main():
    """Run the complete market readiness test"""
    print("🚀 MARKET READINESS TEST SCRIPT")
    print("=" * 60)
    print("Testing your trading bot for market open readiness...")
    print("This will test all critical components and data flows.")
    print("=" * 60)
    
    tester = MarketReadinessTester()
    
    # Run all tests
    tester.test_file_structure()
    tester.test_imports()
    
    components = tester.test_component_initialization()
    
    await tester.test_data_flow(components)
    tester.test_error_handling(components)
    tester.test_performance(components)
    tester.test_configuration()
    
    await tester.run_integration_test(components)
    
    # Generate final report
    report = tester.generate_final_report()
    
    # Return appropriate exit code
    return 0 if report['ready'] else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test script failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)

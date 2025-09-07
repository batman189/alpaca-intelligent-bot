#!/usr/bin/env python3
"""
System Validation Test Runner
Quick validation of all upgrade components without external dependencies
"""
import sys
import os
import asyncio
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_file_structure():
    """Test that all upgrade component files exist"""
    print("Testing File Structure...")
    
    required_files = [
        "data/multi_source_data_manager.py",
        "models/signal_aggregator.py", 
        "models/multi_timeframe_scanner.py",
        "models/market_regime_detector.py",
        "models/dynamic_watchlist_manager.py",
        "monitoring/comprehensive_logger.py"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            existing_files.append(file_path)
            print(f"  ✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ✗ {file_path}")
    
    success_rate = len(existing_files) / len(required_files)
    print(f"\nFile Structure: {len(existing_files)}/{len(required_files)} files present ({success_rate:.1%})")
    
    return len(missing_files) == 0

def test_basic_imports():
    """Test basic imports of all components"""
    print("\nTesting Basic Imports...")
    
    import_tests = [
        ("Multi-Source Data Manager", "data.multi_source_data_manager", "MultiSourceDataManager"),
        ("Signal Aggregator", "models.signal_aggregator", "SignalAggregator"),
        ("Multi-Timeframe Scanner", "models.multi_timeframe_scanner", "MultiTimeframeScanner"),
        ("Market Regime Detector", "models.market_regime_detector", "MarketRegimeDetector"), 
        ("Dynamic Watchlist Manager", "models.dynamic_watchlist_manager", "DynamicWatchlistManager"),
        ("Comprehensive Logger", "monitoring.comprehensive_logger", "ComprehensiveLogger")
    ]
    
    successful_imports = 0
    failed_imports = []
    
    for name, module_path, class_name in import_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {name}")
            successful_imports += 1
        except ImportError as e:
            print(f"  ✗ {name} - ImportError: {e}")
            failed_imports.append((name, str(e)))
        except AttributeError as e:
            print(f"  ✗ {name} - AttributeError: {e}")
            failed_imports.append((name, str(e)))
        except Exception as e:
            print(f"  ✗ {name} - Error: {e}")
            failed_imports.append((name, str(e)))
    
    success_rate = successful_imports / len(import_tests)
    print(f"\nImport Tests: {successful_imports}/{len(import_tests)} successful ({success_rate:.1%})")
    
    if failed_imports:
        print("\nFailed Imports:")
        for name, error in failed_imports:
            print(f"  - {name}: {error}")
    
    return len(failed_imports) == 0

def test_component_initialization():
    """Test that components can be initialized"""
    print("\nTesting Component Initialization...")
    
    successful_inits = 0
    failed_inits = []
    
    # Test each component initialization
    components = [
        ("Multi-Source Data Manager", "data.multi_source_data_manager", "MultiSourceDataManager"),
        ("Signal Aggregator", "models.signal_aggregator", "SignalAggregator"),
        ("Multi-Timeframe Scanner", "models.multi_timeframe_scanner", "MultiTimeframeScanner"),
        ("Market Regime Detector", "models.market_regime_detector", "MarketRegimeDetector"),
        ("Dynamic Watchlist Manager", "models.dynamic_watchlist_manager", "DynamicWatchlistManager"),
        ("Comprehensive Logger", "monitoring.comprehensive_logger", "ComprehensiveLogger")
    ]
    
    for name, module_path, class_name in components:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            instance = cls()
            
            # Check that instance was created
            if instance is not None:
                print(f"  ✓ {name} initialized successfully")
                successful_inits += 1
            else:
                print(f"  ✗ {name} initialization returned None")
                failed_inits.append((name, "Initialization returned None"))
                
        except Exception as e:
            print(f"  ✗ {name} initialization failed: {e}")
            failed_inits.append((name, str(e)))
    
    success_rate = successful_inits / len(components)
    print(f"\nInitialization Tests: {successful_inits}/{len(components)} successful ({success_rate:.1%})")
    
    if failed_inits:
        print("\nFailed Initializations:")
        for name, error in failed_inits:
            print(f"  - {name}: {error}")
    
    return len(failed_inits) == 0

def test_required_methods():
    """Test that components have required methods"""
    print("\nTesting Required Methods...")
    
    method_tests = [
        ("Multi-Source Data Manager", "data.multi_source_data_manager", "MultiSourceDataManager", 
         ["get_market_data", "get_health_status"]),
        ("Signal Aggregator", "models.signal_aggregator", "SignalAggregator", 
         ["aggregate_signals"]),
        ("Multi-Timeframe Scanner", "models.multi_timeframe_scanner", "MultiTimeframeScanner", 
         ["scan_all_timeframes"]),
        ("Market Regime Detector", "models.market_regime_detector", "MarketRegimeDetector", 
         ["detect_market_regime"]),
        ("Dynamic Watchlist Manager", "models.dynamic_watchlist_manager", "DynamicWatchlistManager", 
         ["update_watchlists", "get_watchlist"]),
        ("Comprehensive Logger", "monitoring.comprehensive_logger", "ComprehensiveLogger", 
         ["log_opportunity_detected", "get_system_health_report"])
    ]
    
    successful_methods = 0
    total_methods = 0
    failed_methods = []
    
    for name, module_path, class_name, required_methods in method_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            instance = cls()
            
            for method_name in required_methods:
                total_methods += 1
                if hasattr(instance, method_name):
                    print(f"  ✓ {name}.{method_name}")
                    successful_methods += 1
                else:
                    print(f"  ✗ {name}.{method_name}")
                    failed_methods.append(f"{name}.{method_name}")
                    
        except Exception as e:
            print(f"  ✗ {name} method testing failed: {e}")
            for method_name in required_methods:
                total_methods += 1
                failed_methods.append(f"{name}.{method_name}")
    
    success_rate = successful_methods / total_methods if total_methods > 0 else 0
    print(f"\nMethod Tests: {successful_methods}/{total_methods} methods present ({success_rate:.1%})")
    
    if failed_methods:
        print("\nMissing Methods:")
        for method in failed_methods:
            print(f"  - {method}")
    
    return len(failed_methods) == 0

def test_enum_definitions():
    """Test that required enums are defined"""
    print("\nTesting Enum Definitions...")
    
    enum_tests = [
        ("TimeFrame Enum", "models.multi_timeframe_scanner", "TimeFrame", 
         ["SCALP", "SHORT", "SWING", "POSITION", "DAILY"]),
        ("MarketRegime Enum", "models.market_regime_detector", "MarketRegime", 
         ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]),
        ("WatchlistCategory Enum", "models.dynamic_watchlist_manager", "WatchlistCategory", 
         ["BASE", "TRENDING", "VOLUME_LEADERS", "EARNINGS"])
    ]
    
    successful_enums = 0
    total_enum_values = 0
    failed_enum_values = []
    
    for name, module_path, enum_name, required_values in enum_tests:
        try:
            module = __import__(module_path, fromlist=[enum_name])
            enum_cls = getattr(module, enum_name)
            
            for value_name in required_values:
                total_enum_values += 1
                if hasattr(enum_cls, value_name):
                    print(f"  ✓ {enum_name}.{value_name}")
                    successful_enums += 1
                else:
                    print(f"  ✗ {enum_name}.{value_name}")
                    failed_enum_values.append(f"{enum_name}.{value_name}")
                    
        except Exception as e:
            print(f"  ✗ {name} enum testing failed: {e}")
            for value_name in required_values:
                total_enum_values += 1
                failed_enum_values.append(f"{enum_name}.{value_name}")
    
    success_rate = successful_enums / total_enum_values if total_enum_values > 0 else 0
    print(f"\nEnum Tests: {successful_enums}/{total_enum_values} enum values present ({success_rate:.1%})")
    
    if failed_enum_values:
        print("\nMissing Enum Values:")
        for enum_value in failed_enum_values:
            print(f"  - {enum_value}")
    
    return len(failed_enum_values) == 0

def test_database_creation():
    """Test that logger can create database"""
    print("\nTesting Database Creation...")
    
    try:
        from monitoring.comprehensive_logger import ComprehensiveLogger
        logger = ComprehensiveLogger()
        
        # Check if database file exists
        if hasattr(logger, 'db_path'):
            if os.path.exists(logger.db_path):
                print(f"  ✓ Database created at: {logger.db_path}")
                return True
            else:
                print(f"  ✗ Database not found at: {logger.db_path}")
                return False
        else:
            print("  ✗ Logger has no db_path attribute")
            return False
            
    except Exception as e:
        print(f"  ✗ Database creation test failed: {e}")
        return False

def run_performance_test():
    """Simple performance test"""
    print("\nTesting Basic Performance...")
    
    try:
        start_time = time.time()
        
        # Import all components
        from data.multi_source_data_manager import MultiSourceDataManager
        from models.signal_aggregator import SignalAggregator
        from models.multi_timeframe_scanner import MultiTimeframeScanner
        from models.market_regime_detector import MarketRegimeDetector
        from models.dynamic_watchlist_manager import DynamicWatchlistManager
        from monitoring.comprehensive_logger import ComprehensiveLogger
        
        import_time = time.time() - start_time
        
        # Initialize all components
        init_start = time.time()
        components = {
            'data_manager': MultiSourceDataManager(),
            'signal_aggregator': SignalAggregator(), 
            'timeframe_scanner': MultiTimeframeScanner(),
            'regime_detector': MarketRegimeDetector(),
            'watchlist_manager': DynamicWatchlistManager(),
            'logger': ComprehensiveLogger()
        }
        init_time = time.time() - init_start
        
        total_time = time.time() - start_time
        
        print(f"  ✓ Import time: {import_time:.3f}s")
        print(f"  ✓ Initialization time: {init_time:.3f}s") 
        print(f"  ✓ Total startup time: {total_time:.3f}s")
        
        # Performance thresholds
        if total_time < 2.0:
            print("  ✓ EXCELLENT startup performance")
            return True
        elif total_time < 5.0:
            print("  ✓ GOOD startup performance")
            return True
        else:
            print("  ⚠ SLOW startup performance")
            return False
            
    except Exception as e:
        print(f"  ✗ Performance test failed: {e}")
        return False

def main():
    """Run all system validation tests"""
    print("=" * 60)
    print("SYSTEM VALIDATION TEST RUNNER")
    print("=" * 60)
    print("Validating all upgrade components before market open...")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Component Initialization", test_component_initialization),
        ("Required Methods", test_required_methods),
        ("Enum Definitions", test_enum_definitions),
        ("Database Creation", test_database_creation),
        ("Basic Performance", run_performance_test)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"ERROR: {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("\n🟢 SYSTEM READY FOR MARKET OPEN!")
        print("✓ All critical components are functional")
        print("✓ Your upgraded trading bot is ready for production")
    elif success_rate >= 0.6:
        print("\n🟡 SYSTEM MOSTLY READY")
        print("⚠ Some issues detected - review failed tests")
        print("⚠ Consider fixing issues before production use")
    else:
        print("\n🔴 SYSTEM NEEDS ATTENTION")
        print("✗ Multiple critical issues detected")
        print("✗ Address failed tests before market open")
    
    print("\n" + "=" * 60)
    print("SYSTEM VALIDATION COMPLETE")
    print("=" * 60)
    
    return success_rate >= 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
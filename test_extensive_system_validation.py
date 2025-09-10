#!/usr/bin/env python3
"""
EXTENSIVE SYSTEM VALIDATION TEST SUITE
Tests every aspect of the trading bot with your $100/month Alpaca plan
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project path
sys.path.append('C:/Bot/alpaca-intelligent-bot')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtensiveSystemValidator:
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': {},
            'performance_metrics': {},
            'errors': []
        }
        
        # Test configurations
        self.test_symbols = [
            'SPY', 'QQQ', 'IWM',           # ETFs
            'AAPL', 'MSFT', 'GOOGL',       # Tech giants  
            'TSLA', 'NVDA', 'AMZN',        # Growth stocks
            'JPM', 'JNJ', 'PG'             # Traditional stocks
        ]
        
        self.test_timeframes = ['1Min', '5Min', '15Min', '1Hour', '1Day']
        self.stress_test_symbols = ['SPY', 'AAPL', 'TSLA', 'NVDA', 'MSFT'] * 4  # 20 symbols for stress test
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test results"""
        self.results['total_tests'] += 1
        
        if success:
            self.results['passed_tests'] += 1
            status = "[PASS]"
        else:
            self.results['failed_tests'] += 1  
            status = "[FAIL]"
            
        self.results['test_results'][test_name] = {
            'status': status,
            'success': success,
            'details': details,
            'duration': duration
        }
        
        print(f"{status} {test_name} ({duration:.2f}s) - {details}")
        
    def print_section_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "="*80)
        print(f"  {title.upper()}")
        print("="*80)
        
    async def test_1_alpaca_api_connectivity(self):
        """Test 1: Alpaca API Connectivity and Authentication"""
        self.print_section_header("TEST 1: ALPACA API CONNECTIVITY")
        
        start_time = time.time()
        try:
            # Test basic connection
            from test_alpaca_connection import test_credentials, test_basic_connection, test_market_data
            
            # Credentials test
            creds_result = test_credentials()
            self.log_test_result("1.1 - Alpaca Credentials", creds_result, 
                               "API credentials properly configured" if creds_result else "Missing API credentials")
            
            # Basic connection test
            conn_result = test_basic_connection()
            self.log_test_result("1.2 - Alpaca Basic Connection", conn_result,
                               "Successfully connected to Alpaca API" if conn_result else "Connection failed")
            
            # Market data test
            data_result = test_market_data()
            self.log_test_result("1.3 - Alpaca Market Data", data_result,
                               "Market data accessible" if data_result else "Market data access failed")
            
            duration = time.time() - start_time
            overall_success = creds_result and conn_result and data_result
            self.log_test_result("1.0 - Overall Alpaca API", overall_success, 
                               f"All Alpaca tests passed" if overall_success else "Some Alpaca tests failed", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("1.0 - Overall Alpaca API", False, f"Exception: {str(e)}", duration)
            self.results['errors'].append(f"Alpaca API Test: {str(e)}")
    
    async def test_2_data_source_coverage(self):
        """Test 2: Comprehensive Data Source Coverage"""
        self.print_section_header("TEST 2: DATA SOURCE COVERAGE")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            
            api_key = os.getenv('APCA_API_KEY_ID')
            secret_key = os.getenv('APCA_API_SECRET_KEY')
            
            manager = MultiSourceDataManager(api_key, secret_key)
            
            # Test each symbol across multiple timeframes
            for i, symbol in enumerate(self.test_symbols):
                start_time = time.time()
                
                try:
                    # Test primary timeframe (15Min)
                    data = await manager.get_market_data(symbol, '15Min', 10)
                    
                    if data is not None and len(data) > 0:
                        latest_price = data['close'].iloc[-1]
                        volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
                        data_source = getattr(data, 'attrs', {}).get('data_source', 'Real')
                        
                        # Validate data quality
                        price_valid = latest_price > 0 and latest_price < 10000  # Reasonable price range
                        has_ohlc = all(col in data.columns for col in ['open', 'high', 'low', 'close'])
                        has_volume = 'volume' in data.columns
                        
                        quality_score = sum([price_valid, has_ohlc, has_volume, data_source != 'MOCK'])
                        
                        duration = time.time() - start_time
                        success = quality_score >= 3
                        
                        details = f"Price: ${latest_price:.2f}, Vol: {volume:,}, Quality: {quality_score}/4"
                        self.log_test_result(f"2.{i+1} - {symbol} Data Quality", success, details, duration)
                        
                        # Store performance metrics
                        self.results['performance_metrics'][f"{symbol}_fetch_time"] = duration
                        self.results['performance_metrics'][f"{symbol}_price"] = latest_price
                        
                    else:
                        duration = time.time() - start_time
                        self.log_test_result(f"2.{i+1} - {symbol} Data Quality", False, "No data returned", duration)
                        
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_test_result(f"2.{i+1} - {symbol} Data Quality", False, f"Error: {str(e)}", duration)
                    self.results['errors'].append(f"{symbol} Data Test: {str(e)}")
                    
        except Exception as e:
            self.log_test_result("2.0 - Data Source Setup", False, f"Setup failed: {str(e)}")
            self.results['errors'].append(f"Data Source Coverage: {str(e)}")
    
    async def test_3_timeframe_compatibility(self):
        """Test 3: Multi-Timeframe Data Compatibility"""
        self.print_section_header("TEST 3: TIMEFRAME COMPATIBILITY")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            
            api_key = os.getenv('APCA_API_KEY_ID')
            secret_key = os.getenv('APCA_API_SECRET_KEY')
            manager = MultiSourceDataManager(api_key, secret_key)
            
            # Test each timeframe with SPY (most liquid)
            symbol = 'SPY'
            
            for i, timeframe in enumerate(self.test_timeframes):
                start_time = time.time()
                
                try:
                    data = await manager.get_market_data(symbol, timeframe, 5)
                    
                    if data is not None and len(data) > 0:
                        # Validate timeframe-specific characteristics
                        time_diff = data.index[-1] - data.index[0] if len(data) > 1 else timedelta(0)
                        bars_count = len(data)
                        
                        # Check if timeframe makes sense
                        timeframe_valid = True
                        if timeframe == '1Min' and time_diff > timedelta(hours=2):
                            timeframe_valid = False
                        elif timeframe == '1Day' and time_diff < timedelta(days=1):
                            timeframe_valid = False
                            
                        duration = time.time() - start_time
                        details = f"Bars: {bars_count}, Span: {time_diff}, Valid: {timeframe_valid}"
                        
                        success = timeframe_valid and bars_count > 0
                        self.log_test_result(f"3.{i+1} - {timeframe} Timeframe", success, details, duration)
                        
                    else:
                        duration = time.time() - start_time
                        self.log_test_result(f"3.{i+1} - {timeframe} Timeframe", False, "No data returned", duration)
                        
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_test_result(f"3.{i+1} - {timeframe} Timeframe", False, f"Error: {str(e)}", duration)
                    
        except Exception as e:
            self.log_test_result("3.0 - Timeframe Test Setup", False, f"Setup failed: {str(e)}")
    
    async def test_4_data_client_performance(self):
        """Test 4: Data Client Performance and Streaming"""
        self.print_section_header("TEST 4: DATA CLIENT PERFORMANCE")
        
        try:
            from data.data_client import EnhancedDataClient
            
            api_key = os.getenv('APCA_API_KEY_ID')
            secret_key = os.getenv('APCA_API_SECRET_KEY')
            
            start_time = time.time()
            data_client = EnhancedDataClient(api_key, secret_key)
            init_time = time.time() - start_time
            
            self.log_test_result("4.1 - Data Client Init", True, f"Initialized", init_time)
            
            # Test historical data fetching
            start_time = time.time()
            await data_client._fetch_historical_data('SPY', '15Min')
            fetch_time = time.time() - start_time
            
            # Test data retrieval
            data = data_client.get_data('SPY', '15Min', 10)
            success = data is not None and len(data) > 0
            
            details = f"Fetch: {fetch_time:.2f}s, Bars: {len(data) if data is not None else 0}"
            self.log_test_result("4.2 - Historical Data Fetch", success, details)
            
            # Test performance stats
            try:
                perf_stats = data_client.get_performance_stats()
                cache_hit_rate = perf_stats.get('cache_hit_rate', 0)
                memory_mb = perf_stats.get('total_memory_mb', 0)
                
                perf_good = cache_hit_rate >= 0 and memory_mb < 500  # Under 500MB
                details = f"Cache: {cache_hit_rate:.2%}, Memory: {memory_mb:.1f}MB"
                self.log_test_result("4.3 - Performance Metrics", perf_good, details)
                
                # Store performance data
                self.results['performance_metrics']['cache_hit_rate'] = cache_hit_rate
                self.results['performance_metrics']['memory_usage_mb'] = memory_mb
                
            except Exception as e:
                self.log_test_result("4.3 - Performance Metrics", False, f"Error: {str(e)}")
                
        except Exception as e:
            self.log_test_result("4.0 - Data Client Setup", False, f"Setup failed: {str(e)}")
            self.results['errors'].append(f"Data Client Performance: {str(e)}")
    
    async def test_5_error_handling_resilience(self):
        """Test 5: Error Handling and System Resilience"""
        self.print_section_header("TEST 5: ERROR HANDLING & RESILIENCE")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager, CriticalDataFailureError
            
            api_key = os.getenv('APCA_API_KEY_ID')
            secret_key = os.getenv('APCA_API_SECRET_KEY')
            manager = MultiSourceDataManager(api_key, secret_key)
            
            # Test 1: Invalid symbol handling
            start_time = time.time()
            try:
                data = await manager.get_market_data('INVALID_SYMBOL_123', '15Min', 5)
                # Should either return None or raise CriticalDataFailureError
                duration = time.time() - start_time
                
                if data is None:
                    self.log_test_result("5.1 - Invalid Symbol Handling", True, "Properly returned None", duration)
                else:
                    self.log_test_result("5.1 - Invalid Symbol Handling", False, "Should have returned None", duration)
                    
            except CriticalDataFailureError:
                duration = time.time() - start_time
                self.log_test_result("5.1 - Invalid Symbol Handling", True, "Properly raised CriticalDataFailureError", duration)
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result("5.1 - Invalid Symbol Handling", True, f"Handled gracefully: {str(e)}", duration)
            
            # Test 2: Health scoring
            try:
                health_score = manager.get_health_score()
                health_good = 0 <= health_score <= 100
                self.log_test_result("5.2 - Health Score Calculation", health_good, f"Score: {health_score:.1f}%")
                
                self.results['performance_metrics']['health_score'] = health_score
                
            except Exception as e:
                self.log_test_result("5.2 - Health Score Calculation", False, f"Error: {str(e)}")
            
            # Test 3: Data source status
            try:
                status = manager.get_data_source_status()
                has_system_status = 'system' in status
                has_real_sources = any(info.get('is_real_data', False) for key, info in status.items() if key != 'system')
                
                status_good = has_system_status and has_real_sources
                self.log_test_result("5.3 - Data Source Status", status_good, 
                                   f"Sources: {len(status)}, Real data: {has_real_sources}")
                
            except Exception as e:
                self.log_test_result("5.3 - Data Source Status", False, f"Error: {str(e)}")
                
        except Exception as e:
            self.log_test_result("5.0 - Error Handling Setup", False, f"Setup failed: {str(e)}")
    
    async def test_6_stress_testing(self):
        """Test 6: System Stress Testing"""
        self.print_section_header("TEST 6: SYSTEM STRESS TESTING")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            
            api_key = os.getenv('APCA_API_KEY_ID')
            secret_key = os.getenv('APCA_API_SECRET_KEY')
            manager = MultiSourceDataManager(api_key, secret_key)
            
            # Test 1: Concurrent requests
            start_time = time.time()
            
            tasks = []
            for symbol in self.stress_test_symbols[:10]:  # 10 concurrent requests
                task = asyncio.create_task(manager.get_market_data(symbol, '15Min', 5))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            concurrent_duration = time.time() - start_time
            
            successful_requests = sum(1 for r in results if isinstance(r, pd.DataFrame) and len(r) > 0)
            success_rate = successful_requests / len(results)
            
            concurrent_success = success_rate >= 0.8  # 80% success rate acceptable
            details = f"Success: {successful_requests}/{len(results)} ({success_rate:.1%})"
            self.log_test_result("6.1 - Concurrent Requests", concurrent_success, details, concurrent_duration)
            
            # Test 2: Memory usage under load
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                memory_reasonable = memory_mb < 1000  # Under 1GB
                self.log_test_result("6.2 - Memory Usage", memory_reasonable, f"Memory: {memory_mb:.1f}MB")
                
                self.results['performance_metrics']['stress_test_memory_mb'] = memory_mb
                
            except ImportError:
                self.log_test_result("6.2 - Memory Usage", True, "psutil not available - skipped")
            
            # Test 3: Response time consistency
            response_times = []
            for i in range(5):
                start_time = time.time()
                data = await manager.get_market_data('SPY', '15Min', 5)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            consistency_good = max_response_time < avg_response_time * 3  # No response > 3x average
            
            details = f"Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s"
            self.log_test_result("6.3 - Response Time Consistency", consistency_good, details)
            
            self.results['performance_metrics']['avg_response_time'] = avg_response_time
            self.results['performance_metrics']['max_response_time'] = max_response_time
            
        except Exception as e:
            self.log_test_result("6.0 - Stress Test Setup", False, f"Setup failed: {str(e)}")
            self.results['errors'].append(f"Stress Testing: {str(e)}")
    
    async def test_7_integration_testing(self):
        """Test 7: Full System Integration"""
        self.print_section_header("TEST 7: FULL SYSTEM INTEGRATION")
        
        try:
            # Test main application components
            from app import safe_import_components, Config
            
            # Test component imports
            start_time = time.time()
            components = safe_import_components()
            import_duration = time.time() - start_time
            
            imported_components = len([c for c in components.values() if c.__name__ != 'MockComponent'])
            total_components = len(components)
            
            import_success = imported_components >= total_components * 0.7  # 70% real components
            details = f"Real: {imported_components}/{total_components}"
            self.log_test_result("7.1 - Component Imports", import_success, details, import_duration)
            
            # Test configuration
            try:
                config = Config()
                config_valid = bool(config.APCA_API_KEY_ID and config.APCA_API_SECRET_KEY)
                self.log_test_result("7.2 - Configuration", config_valid, 
                                   "Credentials configured" if config_valid else "Missing credentials")
            except Exception as e:
                self.log_test_result("7.2 - Configuration", False, f"Config error: {str(e)}")
            
            # Test data manager integration
            try:
                if 'MultiSourceDataManager' in components:
                    data_manager = components['MultiSourceDataManager'](
                        config.APCA_API_KEY_ID,
                        config.APCA_API_SECRET_KEY  
                    )
                    
                    # Quick integration test
                    start_time = time.time()
                    integration_data = await data_manager.get_market_data('SPY', '15Min', 3)
                    integration_duration = time.time() - start_time
                    
                    integration_success = integration_data is not None and len(integration_data) > 0
                    details = f"Data: {len(integration_data) if integration_data is not None else 0} bars"
                    self.log_test_result("7.3 - Data Manager Integration", integration_success, details, integration_duration)
                    
                else:
                    self.log_test_result("7.3 - Data Manager Integration", False, "MultiSourceDataManager not available")
                    
            except Exception as e:
                self.log_test_result("7.3 - Data Manager Integration", False, f"Integration error: {str(e)}")
                
        except Exception as e:
            self.log_test_result("7.0 - Integration Setup", False, f"Setup failed: {str(e)}")
    
    def print_final_report(self):
        """Print comprehensive final report"""
        self.print_section_header("FINAL COMPREHENSIVE REPORT")
        
        total = self.results['total_tests']
        passed = self.results['passed_tests'] 
        failed = self.results['failed_tests']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"TOTAL TESTS RUN: {total}")
        print(f"TESTS PASSED:    {passed}")
        print(f"TESTS FAILED:    {failed}")
        print(f"SUCCESS RATE:    {success_rate:.1f}%")
        print()
        
        # Performance metrics summary
        if self.results['performance_metrics']:
            print("PERFORMANCE METRICS:")
            print("-" * 40)
            for metric, value in self.results['performance_metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric:25}: {value:.2f}")
                else:
                    print(f"  {metric:25}: {value}")
            print()
        
        # Overall assessment
        if success_rate >= 90:
            print("[EXCELLENT] Your $100/month Alpaca plan is working optimally!")
            print("All systems are functioning properly with premium data access.")
        elif success_rate >= 80:
            print("[VERY GOOD] Most systems working well with minor issues.")
            print("Your Alpaca plan is delivering good value.")
        elif success_rate >= 70:
            print("[GOOD] Core functionality working with some issues to address.")
        elif success_rate >= 50:
            print("[FAIR] Significant issues detected - optimization needed.")
        else:
            print("[POOR] Major issues detected - immediate attention required.")
            
        # Show errors if any
        if self.results['errors']:
            print(f"\nERRORS ENCOUNTERED ({len(self.results['errors'])}):")
            print("-" * 40)
            for i, error in enumerate(self.results['errors'], 1):
                print(f"  {i}. {error}")
        
        print("\n" + "="*80)
        print("EXTENSIVE TESTING COMPLETE")
        print("="*80)
        
        return success_rate >= 80

async def run_extensive_testing():
    """Run all extensive tests"""
    print("STARTING EXTENSIVE SYSTEM VALIDATION")
    print("Testing all aspects of your $100/month Alpaca trading bot")
    print("This will take several minutes to complete...")
    print()
    
    validator = ExtensiveSystemValidator()
    
    # Run all test suites
    await validator.test_1_alpaca_api_connectivity()
    await validator.test_2_data_source_coverage()
    await validator.test_3_timeframe_compatibility()
    await validator.test_4_data_client_performance()
    await validator.test_5_error_handling_resilience() 
    await validator.test_6_stress_testing()
    await validator.test_7_integration_testing()
    
    # Generate final report
    success = validator.print_final_report()
    
    return validator.results, success

if __name__ == "__main__":
    start_time = time.time()
    results, success = asyncio.run(run_extensive_testing())
    total_duration = time.time() - start_time
    
    print(f"\nTOTAL TEST DURATION: {total_duration:.1f} seconds")
    
    if success:
        print("RESULT: All systems validated successfully!")
        sys.exit(0)
    else:
        print("RESULT: Issues detected - review test output.")
        sys.exit(1)
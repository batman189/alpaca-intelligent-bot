#!/usr/bin/env python3
"""
Integration Stress Tests for Professional Trading Bot
Tests system under heavy load and error conditions
"""
import asyncio
import concurrent.futures
import time
import random
import threading
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class StressTester:
    """Comprehensive stress testing for the trading bot"""
    
    def __init__(self):
        self.test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX"]
        self.results = {}
        
    def generate_mock_data(self, symbol, periods=1000):
        """Generate realistic mock market data"""
        base_price = random.uniform(50, 500)
        dates = pd.date_range('2024-01-01', periods=periods, freq='1min')
        
        # Generate realistic price movement
        returns = np.random.normal(0, 0.002, periods)  # 0.2% volatility per minute
        prices = [base_price]
        
        for i in range(1, periods):
            new_price = prices[i-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * random.uniform(1.0, 1.02) for p in prices],
            'low': [p * random.uniform(0.98, 1.0) for p in prices],
            'close': prices,
            'volume': np.random.randint(10000, 1000000, periods)
        })
    
    async def test_parallel_data_fetching(self):
        """Test multiple simultaneous data requests"""
        print("🧪 Testing Parallel Data Fetching...")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            manager = MultiSourceDataManager()
            
            # Create multiple concurrent requests
            async def fetch_data(symbol):
                with patch.object(manager, '_fetch_from_alpaca', 
                                return_value=self.generate_mock_data(symbol)):
                    return await manager.get_market_data(symbol, "1min")
            
            start_time = time.time()
            
            # Run 8 symbols in parallel
            tasks = [fetch_data(symbol) for symbol in self.test_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            print(f"✅ Parallel data fetch: {success_count}/8 successful in {duration:.2f}s")
            
            return {
                'success_rate': success_count / len(results),
                'duration': duration,
                'avg_per_symbol': duration / len(results)
            }
            
        except Exception as e:
            print(f"❌ Parallel data fetching test failed: {e}")
            return {'success_rate': 0, 'duration': float('inf'), 'avg_per_symbol': float('inf')}
    
    async def test_signal_aggregation_load(self):
        """Test signal aggregation under heavy load"""
        print("🧪 Testing Signal Aggregation Load...")
        
        try:
            from models.signal_aggregator import SignalAggregator
            aggregator = SignalAggregator()
            
            # Mock data fetching to return realistic data
            async def mock_fetch_data(symbol):
                return self.generate_mock_data(symbol, 200)
            
            with patch.object(aggregator, '_fetch_market_data', side_effect=mock_fetch_data):
                start_time = time.time()
                
                # Process multiple symbols simultaneously
                tasks = [aggregator.aggregate_signals(symbol) for symbol in self.test_symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                duration = time.time() - start_time
                
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                total_signals = sum(len(r) for r in results if isinstance(r, list))
                
                print(f"✅ Signal aggregation: {success_count}/8 symbols, {total_signals} signals in {duration:.2f}s")
                
                return {
                    'success_rate': success_count / len(results),
                    'duration': duration,
                    'signals_per_second': total_signals / duration if duration > 0 else 0
                }
                
        except Exception as e:
            print(f"❌ Signal aggregation load test failed: {e}")
            return {'success_rate': 0, 'duration': float('inf'), 'signals_per_second': 0}
    
    async def test_multi_timeframe_performance(self):
        """Test multi-timeframe scanning performance"""
        print("🧪 Testing Multi-Timeframe Performance...")
        
        try:
            from models.multi_timeframe_scanner import MultiTimeframeScanner
            scanner = MultiTimeframeScanner()
            
            # Mock data for different timeframes
            async def mock_fetch_timeframe_data(symbol, timeframe):
                periods_map = {'1Min': 200, '5Min': 100, '15Min': 50, '1Hour': 25, '1Day': 10}
                periods = periods_map.get(timeframe, 100)
                return self.generate_mock_data(symbol, periods)
            
            with patch.object(scanner, '_fetch_data_for_timeframe', side_effect=mock_fetch_timeframe_data):
                start_time = time.time()
                
                # Test multiple symbols across all timeframes
                tasks = [scanner.scan_all_timeframes(symbol) for symbol in self.test_symbols[:4]]  # Limit to 4 for performance
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                duration = time.time() - start_time
                
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                total_opportunities = sum(len(r) for r in results if isinstance(r, list))
                
                print(f"✅ Multi-timeframe: {success_count}/4 symbols, {total_opportunities} opportunities in {duration:.2f}s")
                
                return {
                    'success_rate': success_count / len(results),
                    'duration': duration,
                    'opportunities_per_second': total_opportunities / duration if duration > 0 else 0
                }
                
        except Exception as e:
            print(f"❌ Multi-timeframe performance test failed: {e}")
            return {'success_rate': 0, 'duration': float('inf'), 'opportunities_per_second': 0}
    
    async def test_error_resilience(self):
        """Test system resilience under error conditions"""
        print("🧪 Testing Error Resilience...")
        
        try:
            from data.multi_source_data_manager import MultiSourceDataManager
            from models.signal_aggregator import SignalAggregator
            
            manager = MultiSourceDataManager()
            aggregator = SignalAggregator()
            
            error_scenarios = []
            
            # Test data source failures
            with patch.object(manager, '_fetch_from_alpaca', side_effect=Exception("Alpaca timeout")):
                with patch.object(manager, '_fetch_from_iex', return_value=self.generate_mock_data("AAPL")):
                    try:
                        data = await manager.get_market_data("AAPL", "1min")
                        error_scenarios.append("alpaca_failover" if data is not None else "alpaca_failover_failed")
                    except:
                        error_scenarios.append("alpaca_failover_failed")
            
            # Test partial signal processing failures
            original_method = aggregator._analyze_technical_signals
            
            async def failing_technical_analysis(*args, **kwargs):
                if random.random() < 0.5:  # 50% failure rate
                    raise Exception("Technical analysis failed")
                return await original_method(*args, **kwargs)
            
            with patch.object(aggregator, '_analyze_technical_signals', side_effect=failing_technical_analysis):
                try:
                    signals = await aggregator.aggregate_signals("AAPL")
                    error_scenarios.append("partial_signal_recovery" if signals else "signal_failure")
                except:
                    error_scenarios.append("signal_failure")
            
            resilience_score = len([s for s in error_scenarios if "recovery" in s or "failover" in s]) / len(error_scenarios)
            
            print(f"✅ Error resilience: {resilience_score:.1%} recovery rate")
            print(f"   Scenarios tested: {', '.join(error_scenarios)}")
            
            return {
                'resilience_score': resilience_score,
                'scenarios': error_scenarios
            }
            
        except Exception as e:
            print(f"❌ Error resilience test failed: {e}")
            return {'resilience_score': 0, 'scenarios': []}
    
    def test_memory_usage(self):
        """Test memory usage under load"""
        print("🧪 Testing Memory Usage...")
        
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple components
            components = []
            for i in range(10):
                try:
                    from data.multi_source_data_manager import MultiSourceDataManager
                    from models.signal_aggregator import SignalAggregator
                    from monitoring.comprehensive_logger import ComprehensiveLogger
                    
                    components.extend([
                        MultiSourceDataManager(),
                        SignalAggregator(), 
                        ComprehensiveLogger()
                    ])
                except:
                    pass
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clean up
            del components
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = peak_memory - initial_memory
            memory_cleanup = peak_memory - final_memory
            
            print(f"✅ Memory usage: {memory_growth:.1f}MB growth, {memory_cleanup:.1f}MB cleaned up")
            
            return {
                'initial_memory': initial_memory,
                'peak_memory': peak_memory,
                'final_memory': final_memory,
                'memory_growth': memory_growth,
                'cleanup_efficiency': memory_cleanup / memory_growth if memory_growth > 0 else 1.0
            }
            
        except ImportError:
            print("⚠️  psutil not available for memory testing")
            return {'memory_growth': 0, 'cleanup_efficiency': 1.0}
        except Exception as e:
            print(f"❌ Memory usage test failed: {e}")
            return {'memory_growth': float('inf'), 'cleanup_efficiency': 0}
    
    async def test_concurrent_logging(self):
        """Test logging system under concurrent access"""
        print("🧪 Testing Concurrent Logging...")
        
        try:
            from monitoring.comprehensive_logger import ComprehensiveLogger
            logger = ComprehensiveLogger()
            
            # Create concurrent logging tasks
            async def log_events(thread_id):
                for i in range(50):
                    logger.log_opportunity_detected(f"SYM{thread_id}", "test", 50.0, {"test": True})
                    logger.log_system_error(f"component_{thread_id}", "test_error", {"iteration": i})
                    await asyncio.sleep(0.001)  # Small delay
            
            start_time = time.time()
            
            # Run 5 concurrent logging tasks
            tasks = [log_events(i) for i in range(5)]
            await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            
            # Check log integrity
            health_report = logger.get_system_health_report()
            logged_opportunities = health_report.get('opportunities_detected', 0)
            logged_errors = health_report.get('total_errors', 0)
            
            print(f"✅ Concurrent logging: {logged_opportunities} opportunities, {logged_errors} errors in {duration:.2f}s")
            
            return {
                'duration': duration,
                'opportunities_logged': logged_opportunities,
                'errors_logged': logged_errors,
                'logs_per_second': (logged_opportunities + logged_errors) / duration if duration > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Concurrent logging test failed: {e}")
            return {'duration': float('inf'), 'logs_per_second': 0}
    
    async def run_all_stress_tests(self):
        """Run all stress tests"""
        print("🚀 INTEGRATION STRESS TESTS")
        print("=" * 50)
        print("Testing system under heavy load and error conditions...")
        print()
        
        # Run all async tests
        test_methods = [
            self.test_parallel_data_fetching,
            self.test_signal_aggregation_load,
            self.test_multi_timeframe_performance,
            self.test_error_resilience,
            self.test_concurrent_logging
        ]
        
        results = {}
        for test_method in test_methods:
            try:
                result = await test_method()
                results[test_method.__name__] = result
            except Exception as e:
                print(f"❌ {test_method.__name__} failed: {e}")
                results[test_method.__name__] = {'error': str(e)}
        
        # Run memory test (synchronous)
        try:
            memory_result = self.test_memory_usage()
            results['test_memory_usage'] = memory_result
        except Exception as e:
            print(f"❌ Memory test failed: {e}")
            results['test_memory_usage'] = {'error': str(e)}
        
        return results

def generate_performance_report(results):
    """Generate a comprehensive performance report"""
    print("\n📊 PERFORMANCE REPORT")
    print("=" * 40)
    
    # Extract key metrics
    data_fetch = results.get('test_parallel_data_fetching', {})
    signal_agg = results.get('test_signal_aggregation_load', {})
    timeframe = results.get('test_multi_timeframe_performance', {})
    resilience = results.get('test_error_resilience', {})
    logging = results.get('test_concurrent_logging', {})
    memory = results.get('test_memory_usage', {})
    
    print(f"🔄 Data Fetching:")
    print(f"   Success Rate: {data_fetch.get('success_rate', 0):.1%}")
    print(f"   Avg Time/Symbol: {data_fetch.get('avg_per_symbol', 0):.3f}s")
    
    print(f"\n📈 Signal Processing:")
    print(f"   Success Rate: {signal_agg.get('success_rate', 0):.1%}")
    print(f"   Signals/Second: {signal_agg.get('signals_per_second', 0):.1f}")
    
    print(f"\n⏰ Multi-Timeframe:")
    print(f"   Success Rate: {timeframe.get('success_rate', 0):.1%}")
    print(f"   Opportunities/Second: {timeframe.get('opportunities_per_second', 0):.1f}")
    
    print(f"\n🛡️ Error Resilience:")
    print(f"   Recovery Rate: {resilience.get('resilience_score', 0):.1%}")
    
    print(f"\n📝 Concurrent Logging:")
    print(f"   Logs/Second: {logging.get('logs_per_second', 0):.1f}")
    
    print(f"\n💾 Memory Usage:")
    print(f"   Peak Growth: {memory.get('memory_growth', 0):.1f}MB")
    print(f"   Cleanup Efficiency: {memory.get('cleanup_efficiency', 0):.1%}")
    
    # Overall assessment
    success_rates = [
        data_fetch.get('success_rate', 0),
        signal_agg.get('success_rate', 0),
        timeframe.get('success_rate', 0),
        resilience.get('resilience_score', 0)
    ]
    
    avg_success = sum(success_rates) / len(success_rates)
    
    print(f"\n🎯 OVERALL SYSTEM PERFORMANCE")
    print("-" * 30)
    if avg_success > 0.8:
        print("🟢 EXCELLENT - System ready for production")
    elif avg_success > 0.6:
        print("🟡 GOOD - System functional with minor issues")
    else:
        print("🔴 NEEDS ATTENTION - Review failed components")
    
    print(f"Average Success Rate: {avg_success:.1%}")

async def main():
    """Run all stress tests"""
    tester = StressTester()
    
    print("⚡ Starting comprehensive stress testing...")
    print("This will test the system under heavy load conditions.")
    print()
    
    start_time = time.time()
    results = await tester.run_all_stress_tests()
    total_duration = time.time() - start_time
    
    print(f"\n⏱️  Total test duration: {total_duration:.2f}s")
    
    # Generate performance report
    generate_performance_report(results)
    
    print(f"\n🏁 STRESS TESTING COMPLETE")
    print("=" * 40)
    print("Your trading bot has been tested under heavy load conditions.")
    print("Review the performance report above for any areas needing attention.")

if __name__ == "__main__":
    asyncio.run(main())
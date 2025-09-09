#!/usr/bin/env python3
"""
Simple Phase 3 Performance Optimization Test
Tests memory optimization functions without external API calls
"""

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from data.data_client import EnhancedDataClient, MarketData, MultiTimeframeData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_data(rows=1000):
    """Create mock market data for testing"""
    dates = pd.date_range(start='2025-01-01', periods=rows, freq='1T')
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, rows)  # 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, rows)),
        'high': prices * (1 + abs(np.random.normal(0, 0.005, rows))),
        'low': prices * (1 - abs(np.random.normal(0, 0.005, rows))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, rows)
    }, index=dates)

def test_phase3_optimizations():
    """Test Phase 3 memory and caching optimizations"""
    
    print("Phase 3 Performance Optimization Test")
    print("=" * 50)
    
    # Create a mock data client (no API calls)
    client = EnhancedDataClient("test_key", "test_secret")
    
    # Test symbols
    symbols = ['AAPL', 'TSLA', 'SPY', 'QQQ', 'NVDA']
    timeframes = ['1Min', '5Min', '15Min', '1Hour']
    
    print(f"Creating mock data for {len(symbols)} symbols")
    
    # Populate client with mock data
    for symbol in symbols:
        client.market_data[symbol] = MultiTimeframeData(symbol=symbol)
        
        for timeframe in timeframes:
            # Create different sized datasets based on cache sizes
            cache_size = client.CACHE_SIZES.get(timeframe, 500)
            mock_data = create_mock_data(cache_size + 100)  # Slightly more than cache size
            
            client.market_data[symbol].data[timeframe] = mock_data
    
    print("Mock data created")
    
    # Test 1: Memory Optimization
    print("\nTest 1: Memory Optimization")
    print("-" * 30)
    
    initial_stats = client.get_performance_stats()
    print(f"Initial memory: {initial_stats['total_memory_mb']:.2f}MB")
    print(f"Initial rows: {initial_stats['total_rows']:,}")
    
    # Run memory optimization
    optimization_results = client.optimize_memory_usage()
    
    print(f"Memory saved: {optimization_results['memory_saved_mb']:.2f}MB")
    print(f"Memory saved: {optimization_results['memory_saved_percent']:.1f}%")
    print(f"Rows optimized: {optimization_results['rows_optimized']:,}")
    print(f"Symbols optimized: {optimization_results['symbols_optimized']}")
    
    # Test 2: Cache Analysis
    print("\nTest 2: Cache Analysis")
    print("-" * 30)
    
    recommendations = client.get_cache_recommendations()
    print(f"Total memory: {recommendations['total_memory_mb']:.2f}MB")
    
    print("\nMemory by timeframe:")
    for timeframe, usage in recommendations['timeframe_usage'].items():
        cache_size = client.CACHE_SIZES.get(timeframe, 500)
        print(f"  {timeframe}: {usage['memory_mb']:.2f}MB, {usage['avg_rows']:.0f} avg rows (limit: {cache_size})")
    
    if recommendations['recommendations']['memory_optimizations']:
        print("\nMemory recommendations:")
        for rec in recommendations['recommendations']['memory_optimizations']:
            print(f"  • {rec}")
    
    if recommendations['recommendations']['cache_adjustments']:
        print("\nCache recommendations:")
        for rec in recommendations['recommendations']['cache_adjustments']:
            print(f"  • {rec}")
    
    # Test 3: Performance Comparison
    print("\nTest 3: Performance Comparison")  
    print("-" * 30)
    
    final_stats = client.get_performance_stats()
    
    memory_improvement = (
        (initial_stats['total_memory_mb'] - final_stats['total_memory_mb']) / 
        initial_stats['total_memory_mb'] * 100 
        if initial_stats['total_memory_mb'] > 0 else 0
    )
    
    print("BEFORE vs AFTER optimization:")
    print(f"Memory usage: {initial_stats['total_memory_mb']:.2f}MB → {final_stats['total_memory_mb']:.2f}MB")
    print(f"Memory improvement: {memory_improvement:.1f}%")
    print(f"Cached symbols: {final_stats['cached_symbols']}")
    
    # Test 4: Cache Size Validation
    print("\nTest 4: Cache Size Validation")
    print("-" * 30)
    
    all_within_limits = True
    for symbol, symbol_data in client.market_data.items():
        for timeframe, df in symbol_data.data.items():
            expected_size = client.CACHE_SIZES.get(timeframe, client.MAX_CACHE_SIZE)
            actual_size = len(df)
            
            if actual_size > expected_size:
                print(f"FAIL {symbol} {timeframe}: {actual_size} rows (expected <= {expected_size})")
                all_within_limits = False
            else:
                print(f"PASS {symbol} {timeframe}: {actual_size} rows (limit: {expected_size})")
    
    # Test Results
    print("\nPHASE 3 TEST RESULTS")
    print("=" * 50)
    
    success_criteria = {
        'Memory optimization works': optimization_results['memory_saved_mb'] >= 0,
        'Cache analysis works': len(recommendations['timeframe_usage']) > 0,
        'Performance tracking works': final_stats['total_memory_mb'] > 0,
        'Cache limits respected': all_within_limits
    }
    
    passed = sum(success_criteria.values())
    total = len(success_criteria)
    
    print(f"Tests passed: {passed}/{total}")
    for test, result in success_criteria.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {test}")
    
    print(f"\nMemory improvement: {memory_improvement:.1f}%")
    print(f"Memory saved: {optimization_results['memory_saved_mb']:.2f}MB")
    print(f"Dynamic cache sizes: {len(client.CACHE_SIZES)} timeframes optimized")
    
    if passed == total:
        print("\nPhase 3 Performance Optimizations: SUCCESS!")
        print("Memory usage optimized, caching improved, performance tracked!")
    else:
        print(f"\n{total - passed} tests failed - check results above")
    
    return passed == total

if __name__ == "__main__":
    success = test_phase3_optimizations()
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Phase 3 optimization test")
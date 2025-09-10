#!/usr/bin/env python3
"""
Comprehensive Data Source Validation Test
Tests both Alpaca ($100/month plan) and Yahoo Finance with multiple symbols
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime

# Add project path
sys.path.append('C:/Bot/alpaca-intelligent-bot')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

async def test_comprehensive_data_sources():
    """Test all data sources with multiple symbols and timeframes"""
    from data.multi_source_data_manager import MultiSourceDataManager
    
    print("=" * 60)
    print("COMPREHENSIVE DATA SOURCE VALIDATION")
    print("Testing $100/month Alpaca plan + Yahoo Finance")
    print("=" * 60)
    
    # Get credentials
    api_key = os.getenv('APCA_API_KEY_ID')
    secret_key = os.getenv('APCA_API_SECRET_KEY')
    
    print(f"Alpaca API Key: {'[SET]' if api_key else '[NOT SET]'}")
    print(f"Alpaca Secret:  {'[SET]' if secret_key else '[NOT SET]'}")
    print()
    
    # Initialize manager
    manager = MultiSourceDataManager(api_key, secret_key)
    
    # Test symbols (diverse set)
    test_symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']
    test_timeframes = ['15Min', '1Hour', '1Day']
    
    print("TESTING MULTIPLE SYMBOLS AND TIMEFRAMES")
    print("-" * 40)
    
    results = {
        'successful': 0,
        'failed': 0,
        'details': []
    }
    
    for symbol in test_symbols:
        for timeframe in test_timeframes:
            try:
                print(f"Testing {symbol} - {timeframe}... ", end="")
                
                data = await manager.get_market_data(symbol, timeframe, 5)
                
                if data is not None and len(data) > 0:
                    latest_price = data['close'].iloc[-1]
                    latest_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
                    
                    print(f"[SUCCESS] - Price: ${latest_price:.2f}, Volume: {latest_volume:,}")
                    
                    results['successful'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'status': 'SUCCESS',
                        'price': latest_price,
                        'volume': latest_volume,
                        'bars': len(data)
                    })
                else:
                    print("[FAILED] - No data returned")
                    results['failed'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'status': 'FAILED',
                        'error': 'No data returned'
                    })
                    
            except Exception as e:
                print(f"[ERROR] - {str(e)}")
                results['failed'] += 1
                results['details'].append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    # Test data source health
    print("\n" + "=" * 40)
    print("DATA SOURCE HEALTH CHECK")
    print("=" * 40)
    
    try:
        status = manager.get_data_source_status()
        health_score = manager.get_health_score()
        
        print(f"Overall Health Score: {health_score:.1f}%")
        print()
        
        for source, info in status.items():
            if source != 'system':
                active_status = "[ACTIVE]" if info['active'] else "[INACTIVE]"
                print(f"{source:15} {active_status:10} Errors: {info['error_count']}")
        
        # System status
        if 'system' in status:
            sys_info = status['system']
            if sys_info.get('trading_halted', False):
                print(f"\n[WARN] TRADING HALTED: {sys_info.get('halt_reason', 'Unknown')}")
            else:
                print(f"\n[OK] Trading System: OPERATIONAL")
    
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    total_tests = results['successful'] + results['failed']
    success_rate = (results['successful'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n[EXCELLENT] Your $100/month Alpaca plan is working optimally!")
        print("   Real market data is flowing properly from both sources.")
    elif success_rate >= 60:
        print("\n[GOOD] Most data sources working, minor issues detected.")
    else:
        print("\n[ATTENTION NEEDED] Multiple data source failures detected.")
        print("   Your $100/month plan may not be configured correctly.")
    
    # Show any failures
    failures = [d for d in results['details'] if d['status'] != 'SUCCESS']
    if failures:
        print(f"\nFAILED TESTS ({len(failures)}):")
        for fail in failures:
            print(f"  {fail['symbol']} {fail['timeframe']}: {fail.get('error', 'Unknown error')}")
    
    # Value verification
    print("\n" + "=" * 40)
    print("$100/MONTH PLAN VALUE CHECK")
    print("=" * 40)
    
    successful_symbols = set(d['symbol'] for d in results['details'] if d['status'] == 'SUCCESS')
    
    print(f"Symbols with data: {len(successful_symbols)}/{len(test_symbols)}")
    print(f"Data sources working: {', '.join(successful_symbols) if successful_symbols else 'None'}")
    
    if len(successful_symbols) >= 3:
        print("[GREAT VALUE] Getting diverse market data from your plan")
    elif len(successful_symbols) >= 1:
        print("[PARTIAL VALUE] Limited symbols working")
    else:
        print("[NO VALUE] No symbols returning data - plan may be misconfigured")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(test_comprehensive_data_sources())
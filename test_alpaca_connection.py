#!/usr/bin/env python3
"""
Test script to verify Alpaca API connection and account information
"""

import os
import sys
import asyncio
from datetime import datetime
import logging

# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()  # This will load .env file
    load_dotenv('intelligent-trading-bot.env')  # Also try the specific filename
    print(f"Environment loading attempted...")
except ImportError:
    print("dotenv not available")
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_credentials():
    """Check if Alpaca credentials are configured"""
    api_key = os.getenv('APCA_API_KEY_ID')
    secret_key = os.getenv('APCA_API_SECRET_KEY')
    base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    logger.info("=== Alpaca Credentials Test ===")
    logger.info(f"API Key ID: {'Set' if api_key else 'NOT SET'}")
    logger.info(f"Secret Key: {'Set' if secret_key else 'NOT SET'}")
    logger.info(f"Base URL: {base_url}")
    
    if not api_key or not secret_key:
        logger.error("❌ Alpaca credentials are not properly configured!")
        return False
    
    logger.info("✅ Credentials are configured")
    return True

def test_basic_connection():
    """Test basic REST API connection"""
    try:
        from alpaca_trade_api import REST
        
        api_key = os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('APCA_API_SECRET_KEY')
        base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        logger.info("=== Basic Connection Test ===")
        
        # Initialize API client
        api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )
        
        # Test account info
        logger.info("Testing account information...")
        account = api.get_account()
        
        logger.info("✅ Successfully connected to Alpaca!")
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Account Number: {account.account_number}")
        logger.info(f"Trading Blocked: {account.trading_blocked}")
        logger.info(f"Transfers Blocked: {account.transfers_blocked}")
        logger.info(f"Pattern Day Trader: {account.pattern_day_trader}")
        logger.info(f"Equity: ${float(account.equity):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing alpaca-trade-api: {e}")
        logger.info("Install with: pip install alpaca-trade-api")
        return False
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

def test_market_data():
    """Test market data access"""
    try:
        from alpaca_trade_api import REST
        
        api_key = os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('APCA_API_SECRET_KEY')
        base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        logger.info("=== Market Data Test ===")
        
        api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )
        
        # Test getting market data
        symbol = 'SPY'
        logger.info(f"Testing market data for {symbol}...")
        
        # Get latest quote
        quote = api.get_latest_quote(symbol)
        logger.info(f"Latest quote for {symbol}:")
        logger.info(f"  Bid: ${quote.bid_price:.2f} (size: {quote.bid_size})")
        logger.info(f"  Ask: ${quote.ask_price:.2f} (size: {quote.ask_size})")
        logger.info(f"  Timestamp: {quote.timestamp}")
        
        # Get some recent bars
        bars = api.get_bars(symbol, '1Min', limit=5)
        logger.info(f"Recent 1-minute bars for {symbol}:")
        for i, bar in enumerate(bars[-3:]):  # Show last 3 bars
            logger.info(f"  {i+1}. {bar.t}: O=${bar.o:.2f} H=${bar.h:.2f} L=${bar.l:.2f} C=${bar.c:.2f} V={bar.v}")
        
        logger.info("✅ Market data access working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Market data test failed: {e}")
        return False

def test_positions():
    """Test positions and orders"""
    try:
        from alpaca_trade_api import REST
        
        api_key = os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('APCA_API_SECRET_KEY')
        base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        logger.info("=== Positions and Orders Test ===")
        
        api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )
        
        # Get current positions
        positions = api.list_positions()
        logger.info(f"Current positions: {len(positions)}")
        
        if positions:
            for pos in positions[:5]:  # Show first 5 positions
                logger.info(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
                logger.info(f"    Current: ${float(pos.market_value):,.2f}, P&L: ${float(pos.unrealized_pl):,.2f}")
        else:
            logger.info("  No current positions")
        
        # Get recent orders
        orders = api.list_orders(status='all', limit=10)
        logger.info(f"Recent orders: {len(orders)}")
        
        if orders:
            for order in orders[:3]:  # Show first 3 orders
                logger.info(f"  {order.symbol}: {order.side} {order.qty} @ {order.order_type}")
                logger.info(f"    Status: {order.status}, Created: {order.created_at}")
        else:
            logger.info("  No recent orders")
        
        logger.info("✅ Positions and orders access working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Positions/orders test failed: {e}")
        return False

async def test_data_client():
    """Test the enhanced data client"""
    try:
        logger.info("=== Enhanced Data Client Test ===")
        
        # Import our data client
        from data.data_client import EnhancedDataClient
        
        api_key = os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('APCA_API_SECRET_KEY')
        base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Initialize data client
        data_client = EnhancedDataClient(api_key, secret_key, base_url)
        
        # Test getting some data
        symbol = 'SPY'
        timeframe = '15Min'
        
        logger.info(f"Testing data retrieval for {symbol} on {timeframe}...")
        
        # Fetch historical data
        await data_client._fetch_historical_data(symbol, timeframe)
        
        # Try to get data
        data = data_client.get_data(symbol, timeframe, 10)
        
        if data is not None and len(data) > 0:
            logger.info(f"✅ Retrieved {len(data)} bars of data")
            latest = data.iloc[-1]
            logger.info(f"Latest bar: O=${latest['open']:.2f} H=${latest['high']:.2f} L=${latest['low']:.2f} C=${latest['close']:.2f}")
            
            # Show some technical indicators if available
            import pandas as pd
            if 'rsi' in data.columns and not pd.isna(latest.get('rsi')):
                logger.info(f"Technical indicators: RSI={latest['rsi']:.1f}")
        else:
            logger.warning("⚠️ No data retrieved")
        
        logger.info("✅ Enhanced Data Client test completed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Could not import data client: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Data client test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Alpaca Connection Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Credentials Check", test_credentials),
        ("Basic Connection", test_basic_connection), 
        ("Market Data", test_market_data),
        ("Positions & Orders", test_positions),
    ]
    
    results = {}
    
    # Run synchronous tests
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            logger.info("")  # Add spacing
        except Exception as e:
            logger.error(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
            logger.info("")
    
    # Run async test
    try:
        logger.info("Running async data client test...")
        result = asyncio.run(test_data_client())
        results["Data Client"] = result
    except Exception as e:
        logger.error(f"❌ Data Client test failed: {e}")
        results["Data Client"] = False
    
    # Summary
    logger.info("=" * 50)
    logger.info("🏁 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 50)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Alpaca connection is working properly.")
        return True
    else:
        logger.error(f"⚠️ {total - passed} test(s) failed. Please check your configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
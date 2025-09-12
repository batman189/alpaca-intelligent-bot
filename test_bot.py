#!/usr/bin/env python3
"""
Production Trading Bot Test Suite
Comprehensive tests for ML models, risk management, and trading logic
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Import the bot
from ml_options_bot import MLOptionsBot

class TestMLOptionsBot(unittest.TestCase):
    """Test suite for ML Options Bot"""
    
    def setUp(self):
        """Setup test environment"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'APCA_API_KEY_ID': 'test_key',
            'APCA_API_SECRET_KEY': 'test_secret',
            'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets',
            'WATCHLIST': 'SPY,AAPL',
            'MAX_POSITIONS': '3',
            'RISK_PER_TRADE': '0.02'
        })
        self.env_patcher.start()
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Cleanup test environment"""
        self.env_patcher.stop()
    
    def create_test_data(self):
        """Create realistic test market data"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1min')
        np.random.seed(42)
        
        # Generate realistic price series
        returns = np.random.normal(0, 0.001, len(dates))  # 0.1% volatility per minute
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.randint(1000, 10000, len(dates))
        
        self.test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0005, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
            'close': prices,
            'volume': volumes
        }, index=dates)
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        with patch('alpaca_trade_api.REST'):
            bot = MLOptionsBot()
            
            # Test with our sample data
            result = bot.add_technical_indicators(self.test_data.copy())
            
            # Check that indicators were added
            expected_indicators = ['sma_5', 'sma_15', 'rsi', 'bb_position', 'macd']
            for indicator in expected_indicators:
                self.assertIn(indicator, result.columns, f"Missing indicator: {indicator}")
            
            # Check RSI is in valid range
            rsi_values = result['rsi'].dropna()
            self.assertTrue(all(0 <= val <= 100 for val in rsi_values), "RSI out of range")
            
            # Check Bollinger Band position
            bb_values = result['bb_position'].dropna()
            self.assertTrue(all(-0.5 <= val <= 1.5 for val in bb_values), "BB position out of range")
    
    def test_ml_feature_preparation(self):
        """Test ML feature preparation"""
        with patch('alpaca_trade_api.REST'):
            bot = MLOptionsBot()
            
            # Add indicators first
            data_with_indicators = bot.add_technical_indicators(self.test_data.copy())
            
            # Test feature preparation
            X, y = bot.prepare_ml_features(data_with_indicators)
            
            self.assertIsNotNone(X, "Features should not be None")
            self.assertIsNotNone(y, "Target should not be None")
            self.assertEqual(len(X), len(y), "Features and target length mismatch")
            self.assertGreater(len(X.columns), 10, "Should have multiple features")
            
            # Check for NaN values
            self.assertFalse(X.isnull().any().any(), "Features contain NaN values")
            self.assertFalse(y.isnull().any(), "Target contains NaN values")

def run_tests():
    """Run all tests"""
    print("🧪 Running Production Trading Bot Test Suite...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [TestMLOptionsBot]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
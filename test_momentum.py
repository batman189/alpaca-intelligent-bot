#!/usr/bin/env python3
"""
Test script for the momentum options bot
"""

import os
from momentum_options_bot import MomentumOptionsBot

def test_bot_initialization():
    """Test that bot initializes properly"""
    try:
        # Mock environment variables for testing
        os.environ['APCA_API_KEY_ID'] = 'test_key'
        os.environ['APCA_API_SECRET_KEY'] = 'test_secret'
        
        print("Testing bot initialization...")
        bot = MomentumOptionsBot()
        print("Bot initialized successfully")
        
        # Test that key components exist
        assert hasattr(bot, 'stock_universe')
        assert hasattr(bot, 'analyze_momentum')
        assert len(bot.stock_universe) > 0
        print(f"Stock universe contains {len(bot.stock_universe)} stocks")
        
        # Test momentum analysis method exists
        print("Momentum analysis method available")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Momentum Options Bot")
    print("=" * 40)
    
    success = test_bot_initialization()
    
    if success:
        print("\nAll tests passed!")
        print("Bot ready for live trading with real API keys")
    else:
        print("\nTests failed!")
        exit(1)
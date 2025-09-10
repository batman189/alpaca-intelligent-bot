#!/usr/bin/env python3
"""
SIMPLE WORKING TRADING BOT
No over-engineering - just detect moves and trade
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.append('.')

from data.multi_source_data_manager import MultiSourceDataManager

class SimpleWorkingBot:
    def __init__(self):
        self.data_manager = MultiSourceDataManager()
        self.watchlist = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'UNH', 'SPY', 'QQQ']
        self.move_threshold = 1.5  # 1.5% move threshold
        self.trades_made = 0
        
        logger.info("🤖 Simple Working Bot initialized")
        logger.info(f"📊 Monitoring: {', '.join(self.watchlist)}")
        logger.info(f"🎯 Move threshold: {self.move_threshold}%")
    
    async def check_market_moves(self):
        """Check for significant market moves"""
        logger.info("🔍 Scanning for market opportunities...")
        
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                # Get recent data
                data = await self.data_manager.get_market_data(symbol, '1Day', 5)
                
                if data is not None and len(data) >= 2:
                    current = data.iloc[-1]['close']
                    previous = data.iloc[-2]['close']
                    
                    change_pct = ((current - previous) / previous) * 100
                    
                    logger.info(f"📈 {symbol}: ${current:.2f} ({change_pct:+.2f}%)")
                    
                    # Check if move is significant
                    if abs(change_pct) >= self.move_threshold:
                        direction = "📈 UP" if change_pct > 0 else "📉 DOWN"
                        logger.info(f"🎯 OPPORTUNITY: {symbol} {direction} {change_pct:+.2f}%")
                        
                        opportunities.append({
                            'symbol': symbol,
                            'current_price': current,
                            'change_pct': change_pct,
                            'direction': 'bullish' if change_pct > 0 else 'bearish',
                            'strength': abs(change_pct)
                        })
                else:
                    logger.warning(f"⚠️ No data for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error checking {symbol}: {e}")
        
        return opportunities
    
    async def execute_trade(self, opportunity):
        """Execute a trade based on opportunity"""
        symbol = opportunity['symbol']
        direction = opportunity['direction']
        strength = opportunity['strength']
        price = opportunity['current_price']
        
        logger.info(f"🚀 EXECUTING TRADE:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Direction: {direction}")
        logger.info(f"   Strength: {strength:.2f}%")
        logger.info(f"   Price: ${price:.2f}")
        
        # Simulate trade execution (replace with real trading)
        self.trades_made += 1
        
        trade_details = {
            'trade_id': f"TRADE_{self.trades_made:03d}",
            'symbol': symbol,
            'direction': direction,
            'entry_price': price,
            'timestamp': datetime.now(),
            'reason': f"{strength:.2f}% move detected"
        }
        
        logger.info(f"✅ TRADE EXECUTED: {trade_details['trade_id']}")
        logger.info(f"📋 Reason: {trade_details['reason']}")
        
        # In real implementation, this would:
        # 1. Calculate position size
        # 2. Choose options strategy 
        # 3. Execute via Alpaca API
        # 4. Set stop loss/take profit
        
        return trade_details
    
    async def run_trading_cycle(self):
        """Run one trading cycle"""
        logger.info("🔄 Starting trading cycle...")
        
        try:
            # Check for opportunities
            opportunities = await self.check_market_moves()
            
            if opportunities:
                logger.info(f"🎯 Found {len(opportunities)} opportunities")
                
                # Trade on the strongest opportunity
                best_opportunity = max(opportunities, key=lambda x: x['strength'])
                
                if best_opportunity['strength'] >= self.move_threshold:
                    await self.execute_trade(best_opportunity)
                else:
                    logger.info("📊 No opportunities meet threshold")
            else:
                logger.info("📊 No significant moves detected")
                
        except Exception as e:
            logger.error(f"❌ Trading cycle error: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Simple Working Bot...")
        logger.info("=" * 50)
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"🔄 CYCLE {cycle}")
                
                await self.run_trading_cycle()
                
                logger.info(f"📊 Total trades made: {self.trades_made}")
                logger.info("-" * 30)
                
                # Wait before next cycle (5 minutes)
                await asyncio.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Bot error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

async def main():
    """Main function"""
    # Set environment variables if not set
    if not os.getenv('APCA_API_KEY_ID'):
        os.environ['APCA_API_KEY_ID'] = 'PKR3HD0PCBMO5PHE05QP'
        os.environ['APCA_API_SECRET_KEY'] = 'nBq5PyAnsqAMNd5y1UgKvGscTtcZeG3qFB69TVzy'
        os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
    
    bot = SimpleWorkingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n🛑 Bot stopped")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
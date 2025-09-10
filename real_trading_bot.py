#!/usr/bin/env python3
"""
REAL TRADING BOT - ACTUALLY EXECUTES TRADES IN ALPACA
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import alpaca_trade_api as tradeapi

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.append('.')

from data.multi_source_data_manager import MultiSourceDataManager

class RealTradingBot:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=os.getenv('APCA_API_KEY_ID', 'PKR3HD0PCBMO5PHE05QP'),
            secret_key=os.getenv('APCA_API_SECRET_KEY', 'nBq5PyAnsqAMNd5y1UgKvGscTtcZeG3qFB69TVzy'),
            base_url=os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        self.data_manager = MultiSourceDataManager()
        self.watchlist = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'SPY', 'QQQ']
        self.move_threshold = 1.5  # 1.5% move threshold
        self.position_size = 100  # $100 per trade (small size for testing)
        self.trades_made = 0
        
        logger.info("REAL TRADING BOT initialized")
        logger.info(f"Monitoring: {', '.join(self.watchlist)}")
        logger.info(f"Move threshold: {self.move_threshold}%")
        logger.info(f"Position size: ${self.position_size}")
    
    async def check_account_status(self):
        """Check account status before trading"""
        try:
            account = self.api.get_account()
            
            if account.trading_blocked:
                logger.error("TRADING BLOCKED - Cannot execute trades")
                return False
            
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            logger.info(f"Account Equity: ${equity:,.2f}")
            logger.info(f"Buying Power: ${buying_power:,.2f}")
            
            if buying_power < self.position_size:
                logger.error(f"Insufficient buying power for ${self.position_size} trade")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking account: {e}")
            return False
    
    async def check_market_moves(self):
        """Check for significant market moves"""
        logger.info("Scanning for market opportunities...")
        
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                # Get recent data
                data = await self.data_manager.get_market_data(symbol, '1Day', 5)
                
                if data is not None and len(data) >= 2:
                    current = data.iloc[-1]['close']
                    previous = data.iloc[-2]['close']
                    
                    change_pct = ((current - previous) / previous) * 100
                    
                    logger.info(f"{symbol}: ${current:.2f} ({change_pct:+.2f}%)")
                    
                    # Check if move is significant
                    if abs(change_pct) >= self.move_threshold:
                        direction = "UP" if change_pct > 0 else "DOWN"
                        logger.info(f"OPPORTUNITY: {symbol} {direction} {change_pct:+.2f}%")
                        
                        opportunities.append({
                            'symbol': symbol,
                            'current_price': current,
                            'change_pct': change_pct,
                            'direction': 'buy' if change_pct > 0 else 'sell',
                            'strength': abs(change_pct)
                        })
                else:
                    logger.warning(f"No data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
        
        return opportunities
    
    async def execute_real_trade(self, opportunity):
        """Execute REAL trade in Alpaca account"""
        symbol = opportunity['symbol']
        direction = opportunity['direction'] 
        strength = opportunity['strength']
        price = opportunity['current_price']
        
        # Calculate shares based on position size
        shares = int(self.position_size / price)
        if shares < 1:
            shares = 1
        
        logger.info(f"EXECUTING REAL TRADE:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Direction: {direction}")
        logger.info(f"   Shares: {shares}")
        logger.info(f"   Est. Cost: ${shares * price:.2f}")
        logger.info(f"   Reason: {strength:.2f}% move")
        
        try:
            # Submit market order to Alpaca
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side=direction,  # 'buy' or 'sell'
                type='market',
                time_in_force='day'
            )
            
            self.trades_made += 1
            
            logger.info(f"ORDER SUBMITTED: {order.id}")
            logger.info(f"   Status: {order.status}")
            logger.info(f"   Symbol: {order.symbol}")
            logger.info(f"   Quantity: {order.qty}")
            logger.info(f"   Side: {order.side}")
            
            # Wait a moment and check order status
            await asyncio.sleep(5)
            
            updated_order = self.api.get_order(order.id)
            logger.info(f"ORDER UPDATE: {updated_order.status}")
            
            if updated_order.filled_qty and int(updated_order.filled_qty) > 0:
                avg_fill_price = float(updated_order.filled_avg_price or 0)
                logger.info(f"TRADE EXECUTED:")
                logger.info(f"   Filled: {updated_order.filled_qty} shares")
                logger.info(f"   Avg Price: ${avg_fill_price:.2f}")
                logger.info(f"   Total: ${int(updated_order.filled_qty) * avg_fill_price:.2f}")
            
            return {
                'order_id': order.id,
                'symbol': symbol,
                'status': updated_order.status,
                'filled_qty': updated_order.filled_qty,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"TRADE EXECUTION FAILED: {e}")
            return None
    
    async def run_trading_cycle(self):
        """Run one trading cycle"""
        logger.info("Starting trading cycle...")
        
        try:
            # Check account first
            if not await self.check_account_status():
                logger.error("Account check failed - skipping cycle")
                return
            
            # Check for opportunities
            opportunities = await self.check_market_moves()
            
            if opportunities:
                logger.info(f"Found {len(opportunities)} opportunities")
                
                # Trade on the strongest opportunity
                best_opportunity = max(opportunities, key=lambda x: x['strength'])
                
                if best_opportunity['strength'] >= self.move_threshold:
                    # Check if we already have a position in this symbol
                    try:
                        position = self.api.get_position(best_opportunity['symbol'])
                        logger.info(f"Already have position in {best_opportunity['symbol']}: {position.qty} shares")
                    except:
                        # No position exists, safe to trade
                        await self.execute_real_trade(best_opportunity)
                else:
                    logger.info("No opportunities meet threshold")
            else:
                logger.info("No significant moves detected")
                
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("Starting REAL Trading Bot...")
        logger.info("=" * 50)
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"CYCLE {cycle}")
                
                await self.run_trading_cycle()
                
                logger.info(f"Total trades made: {self.trades_made}")
                logger.info("-" * 30)
                
                # Wait before next cycle (10 minutes for real trading)
                await asyncio.sleep(600)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Bot error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

async def main():
    """Main function"""
    # Set environment variables
    os.environ['APCA_API_KEY_ID'] = 'PKR3HD0PCBMO5PHE05QP'
    os.environ['APCA_API_SECRET_KEY'] = 'nBq5PyAnsqAMNd5y1UgKvGscTtcZeG3qFB69TVzy'
    os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
    
    bot = RealTradingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nBot stopped")
    except Exception as e:
        print(f"Fatal error: {e}")
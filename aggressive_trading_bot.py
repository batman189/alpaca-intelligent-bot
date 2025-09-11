#!/usr/bin/env python3
"""
AGGRESSIVE TRADING BOT - ACTUALLY TRADES
This bot will make trades when it sees opportunities like UNH +$11 days
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AggressiveTradingBot:
    def __init__(self):
        # Get API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY') 
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not all([self.api_key, self.secret_key]):
            raise ValueError("Missing Alpaca API credentials")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
        
        # Trading symbols
        self.watchlist = ['UNH', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
        
        logger.info(f"🚀 Aggressive Trading Bot initialized with {len(self.watchlist)} symbols")
    
    def get_real_time_data(self, symbol: str) -> dict:
        """Get real-time market data"""
        try:
            # Get current quote
            quote = self.api.get_latest_quote(symbol)
            
            # Get recent bars for trend analysis
            bars = self.api.get_bars(symbol, '1Min', limit=100).df
            if bars.empty:
                return None
            
            current_price = quote.askprice
            
            # Calculate momentum
            if len(bars) >= 10:
                price_10_min_ago = bars['close'].iloc[-10]
                momentum_10min = (current_price - price_10_min_ago) / price_10_min_ago * 100
            else:
                momentum_10min = 0
            
            if len(bars) >= 30:
                price_30_min_ago = bars['close'].iloc[-30]
                momentum_30min = (current_price - price_30_min_ago) / price_30_min_ago * 100
            else:
                momentum_30min = 0
            
            # Get daily data
            today_bars = self.api.get_bars(symbol, '1Day', limit=5).df
            if not today_bars.empty:
                yesterday_close = today_bars['close'].iloc[-2] if len(today_bars) >= 2 else current_price
                daily_change_pct = (current_price - yesterday_close) / yesterday_close * 100
            else:
                daily_change_pct = 0
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'daily_change_pct': daily_change_pct,
                'momentum_10min': momentum_10min,
                'momentum_30min': momentum_30min,
                'volume': bars['volume'].iloc[-1] if not bars.empty else 0,
                'avg_volume': bars['volume'].mean() if len(bars) > 10 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def should_buy(self, data: dict) -> tuple[bool, str, float]:
        """Aggressive buy logic - actually makes trades"""
        symbol = data['symbol']
        daily_change = data['daily_change_pct']
        momentum_10 = data['momentum_10min']
        momentum_30 = data['momentum_30min']
        volume = data['volume']
        avg_volume = data['avg_volume']
        
        # AGGRESSIVE RULES - If your classmates are making money, we need to be more aggressive
        
        # Rule 1: Strong daily momentum (like UNH +$11)
        if daily_change > 3.0:  # More than 3% up today
            confidence = min(daily_change / 10.0, 0.9)  # Higher gains = higher confidence
            return True, f"Strong daily momentum: +{daily_change:.1f}% today", confidence
        
        # Rule 2: Strong short-term momentum
        if momentum_10 > 1.5 and momentum_30 > 2.0:  # Accelerating upward
            confidence = min((momentum_10 + momentum_30) / 10.0, 0.8)
            return True, f"Accelerating momentum: {momentum_30:.1f}% (30min), {momentum_10:.1f}% (10min)", confidence
        
        # Rule 3: Volume breakout with positive momentum
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2.0 and momentum_10 > 0.5:  # High volume + positive momentum
            confidence = min(volume_ratio / 5.0, 0.7)
            return True, f"Volume breakout: {volume_ratio:.1f}x avg volume with +{momentum_10:.1f}% momentum", confidence
        
        # Rule 4: Steady upward trend
        if momentum_30 > 1.0 and momentum_10 > 0.3 and daily_change > 1.0:
            confidence = 0.6
            return True, f"Steady upward trend: {daily_change:.1f}% daily, {momentum_30:.1f}% 30min", confidence
        
        return False, "No clear buy signal", 0.0
    
    def should_sell(self, symbol: str) -> tuple[bool, str]:
        """Check if we should sell existing positions"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                if position.symbol == symbol:
                    unrealized_pl_pct = float(position.unrealized_plpc) * 100
                    
                    # Take profit at 5%+
                    if unrealized_pl_pct > 5.0:
                        return True, f"Taking profit: +{unrealized_pl_pct:.1f}%"
                    
                    # Stop loss at -3%
                    if unrealized_pl_pct < -3.0:
                        return True, f"Stop loss: {unrealized_pl_pct:.1f}%"
            
            return False, "Hold position"
            
        except Exception as e:
            logger.error(f"Error checking positions for {symbol}: {e}")
            return False, "Error checking position"
    
    def calculate_position_size(self, confidence: float) -> int:
        """Calculate position size based on confidence and account value"""
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Risk 2-5% of account based on confidence
            risk_pct = 0.02 + (confidence * 0.03)  # 2% to 5%
            risk_amount = buying_power * risk_pct
            
            # Minimum $100, maximum $2000 per trade
            risk_amount = max(100, min(risk_amount, 2000))
            
            return int(risk_amount)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 100  # Default $100
    
    async def execute_trade(self, symbol: str, action: str, reason: str, confidence: float):
        """Execute the actual trade"""
        try:
            if action == 'buy':
                # Check if we already have a position
                try:
                    position = self.api.get_position(symbol)
                    if float(position.qty) > 0:
                        logger.info(f"💼 Already holding {symbol}, skipping buy")
                        return
                except:
                    pass  # No position exists, continue with buy
                
                # Calculate position size
                notional = self.calculate_position_size(confidence)
                
                # Place market buy order
                order = self.api.submit_order(
                    symbol=symbol,
                    notional=notional,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                logger.info(f"✅ BUY {symbol}: ${notional} - {reason} (confidence: {confidence:.1%})")
                return order
                
            elif action == 'sell':
                # Get current position
                position = self.api.get_position(symbol)
                qty = abs(int(float(position.qty)))
                
                if qty > 0:
                    # Place market sell order
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"💰 SELL {symbol}: {qty} shares - {reason}")
                    return order
                    
        except Exception as e:
            logger.error(f"❌ Trade execution failed for {symbol}: {e}")
            return None
    
    async def trading_cycle(self):
        """Main trading logic - this is what actually makes money"""
        logger.info("🔄 Starting trading cycle...")
        
        for symbol in self.watchlist:
            try:
                # Get real-time market data
                data = self.get_real_time_data(symbol)
                if not data:
                    continue
                
                # Check for sell signals first
                should_sell_now, sell_reason = self.should_sell(symbol)
                if should_sell_now:
                    await self.execute_trade(symbol, 'sell', sell_reason, 0.8)
                    continue
                
                # Check for buy signals
                should_buy_now, buy_reason, confidence = self.should_buy(data)
                
                if should_buy_now:
                    logger.info(f"🎯 {symbol}: BUY SIGNAL - {buy_reason}")
                    await self.execute_trade(symbol, 'buy', buy_reason, confidence)
                else:
                    logger.info(f"⚪ {symbol}: HOLD - {buy_reason} (${data['current_price']:.2f}, {data['daily_change_pct']:+.1f}%)")
                
                # Small delay between symbols
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
        
        # Show portfolio status
        await self.show_portfolio_status()
    
    async def show_portfolio_status(self):
        """Show current portfolio performance"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            portfolio_value = float(account.portfolio_value)
            day_pl = float(account.todays_pl)
            day_pl_pct = (day_pl / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            logger.info(f"📊 Portfolio: ${portfolio_value:,.2f} | Today: ${day_pl:+,.2f} ({day_pl_pct:+.2f}%)")
            
            if positions:
                logger.info(f"💼 Active Positions: {len(positions)}")
                for pos in positions:
                    pl_pct = float(pos.unrealized_plpc) * 100
                    logger.info(f"   {pos.symbol}: ${pos.market_value} ({pl_pct:+.1f}%)")
            else:
                logger.info("💼 No active positions")
                
        except Exception as e:
            logger.error(f"Error showing portfolio status: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 AGGRESSIVE TRADING BOT STARTED")
        logger.info("💡 This bot ACTUALLY TRADES when it sees opportunities!")
        
        try:
            while True:
                # Check if market is open
                clock = self.api.get_clock()
                if clock.is_open:
                    await self.trading_cycle()
                    
                    # Wait 60 seconds between cycles during market hours
                    await asyncio.sleep(60)
                else:
                    logger.info("🕐 Market closed, waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes when market closed
                    
        except KeyboardInterrupt:
            logger.info("👋 Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Critical error: {e}")

def main():
    """Main entry point"""
    print("=" * 60)
    print("⚡ AGGRESSIVE TRADING BOT")
    print("=" * 60)
    print("This bot ACTUALLY TRADES when it sees opportunities!")
    print("If UNH is up $11, it will BUY!")
    print("If TSLA is trending up, it will BUY!")
    print("=" * 60)
    print()
    
    bot = AggressiveTradingBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()
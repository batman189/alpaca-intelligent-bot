#!/usr/bin/env python3
"""
MOMENTUM OPTIONS TRADING BOT
Based on proven Alpaca long-short strategy, adapted for momentum options trading
Catches big moves early using volume surge detection and breakout patterns
"""

import os
import datetime
import threading
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MomentumOptionsBot:
    def __init__(self):
        # Alpaca API setup
        self.api_key = os.environ.get('APCA_API_KEY_ID')
        self.secret_key = os.environ.get('APCA_API_SECRET_KEY') 
        self.base_url = os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")
        
        self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
        
        # High-momentum stock universe (same as successful Alpaca example)
        self.stock_universe = [
            'TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ',
            'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'ROKU', 'ZM', 'SHOP', 'SQ',
            'SNAP', 'TWTR', 'UBER', 'LYFT', 'NIO', 'PLTR', 'GME', 'AMC', 'BB'
        ]
        
        # Initialize tracking variables
        self.momentum_stocks = []
        self.breakout_stocks = []
        self.current_positions = {}
        self.last_analysis = {}
        
        # Trading parameters
        self.max_positions = 5
        self.risk_per_trade = 0.02  # 2% per trade
        self.min_volume_surge = 2.0  # 2x average volume
        self.min_price_move = 0.02   # 2% price move threshold
        
        print(f"Momentum Options Bot initialized")
        print(f"Tracking {len(self.stock_universe)} stocks")
        print(f"Risk per trade: {self.risk_per_trade:.1%}")

    def run(self):
        """Main trading loop - based on Alpaca long-short pattern"""
        print("=" * 60)
        print("MOMENTUM OPTIONS TRADING BOT")
        print("=" * 60)
        print("Catches big moves using volume surge + breakout detection")
        print("Based on proven Alpaca trading patterns")
        print("=" * 60)
        
        # Cancel existing orders first (following Alpaca pattern)
        self.cancel_all_orders()
        
        # Wait for market to open
        self.await_market_open()
        
        # Main trading loop
        while True:
            try:
                clock = self.api.get_clock()
                if not clock.is_open:
                    print(f"[CLOCK] Market closed. Next open: {clock.next_open}")
                    time.sleep(600)  # Check every 10 minutes
                    continue
                
                # Check if we need to close positions near market close
                closing_time = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
                current_time = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
                time_to_close = closing_time - current_time
                
                if time_to_close < (60 * 15):  # 15 minutes before close
                    print("[CLOSE] Market closing soon. Closing all positions.")
                    self.close_all_positions()
                    time.sleep(60 * 15)  # Wait for market close
                    continue
                
                # Run momentum analysis and trading
                self.run_momentum_analysis()
                
                # Wait 1 minute between cycles (following Alpaca pattern)
                print("[WAIT] Next analysis in 1 minute...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n[EXIT] Bot stopped by user")
                break
            except Exception as e:
                print(f"[ERROR] Error in main loop: {e}")
                time.sleep(60)

    def await_market_open(self):
        """Wait for market to open - exact copy of Alpaca pattern"""
        print("Waiting for market to open...")
        is_open = self.api.get_clock().is_open
        while not is_open:
            clock = self.api.get_clock()
            opening_time = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
            current_time = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            time_to_open = int((opening_time - current_time) / 60)
            print(f"{time_to_open} minutes til market open.")
            time.sleep(60)
            is_open = self.api.get_clock().is_open
        print("[SUCCESS] Market opened.")

    def cancel_all_orders(self):
        """Cancel all existing orders - following Alpaca pattern"""
        orders = self.api.list_orders(status="open")
        for order in orders:
            try:
                self.api.cancel_order(order.id)
                print(f"[CANCEL] Cancelled order: {order.symbol}")
            except Exception as e:
                print(f"[WARNING] Failed to cancel order {order.id}: {e}")

    def run_momentum_analysis(self):
        """Main momentum analysis - detect volume surges and breakouts"""
        print("\n[ANALYSIS] Running momentum analysis...")
        
        momentum_candidates = []
        
        # Analyze each stock for momentum signals
        for symbol in self.stock_universe:
            try:
                momentum_score = self.analyze_momentum(symbol)
                if momentum_score:
                    momentum_candidates.append(momentum_score)
                    print(f"[TARGET] {symbol}: Score={momentum_score['score']:.2f}, "
                          f"Volume={momentum_score['volume_ratio']:.1f}x, "
                          f"Price={momentum_score['price_change']:.1%}")
                    
            except Exception as e:
                print(f"[WARNING] Error analyzing {symbol}: {e}")
                continue
        
        # Sort by momentum score (highest first)
        momentum_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top candidates for options trading
        top_candidates = momentum_candidates[:self.max_positions]
        
        if top_candidates:
            print(f"\n[LAUNCH] Top momentum candidates: {[c['symbol'] for c in top_candidates]}")
            
            # Execute options trades on top candidates
            for candidate in top_candidates:
                self.execute_momentum_options_trade(candidate)
        else:
            print("[SLEEP] No momentum opportunities found")

    def analyze_momentum(self, symbol: str) -> Optional[Dict]:
        """Analyze momentum using volume surge + price breakout detection"""
        try:
            # Get recent data (following Alpaca's data fetching pattern)
            end_date = pd.Timestamp.now().normalize()
            start_date = end_date - pd.Timedelta(days=5)
            
            # Get minute bars for detailed analysis
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Minute,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=1000
            ).df
            
            if bars.empty or len(bars) < 50:
                return None
            
            # Calculate momentum indicators
            current_price = bars['close'].iloc[-1]
            
            # 1. Volume surge detection (key momentum indicator)
            recent_volume = bars['volume'].iloc[-10:].mean()
            avg_volume = bars['volume'].iloc[-100:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
            
            # 2. Price momentum calculation
            price_10min_ago = bars['close'].iloc[-10]
            price_change = (current_price - price_10min_ago) / price_10min_ago
            
            # 3. Breakout detection
            recent_high = bars['high'].iloc[-20:].max()
            recent_low = bars['low'].iloc[-20:].min()
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            # 4. Volatility expansion
            recent_volatility = bars['close'].iloc[-10:].pct_change().std()
            avg_volatility = bars['close'].iloc[-100:].pct_change().std()
            volatility_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1
            
            # Calculate combined momentum score
            momentum_score = 0
            
            # Volume surge component (most important)
            if volume_ratio >= self.min_volume_surge:
                momentum_score += min(volume_ratio * 20, 50)  # Cap at 50 points
            
            # Price momentum component
            if abs(price_change) >= self.min_price_move:
                momentum_score += abs(price_change) * 100  # Convert to points
            
            # Breakout component
            if price_position > 0.8:  # Near recent highs
                momentum_score += 20
            elif price_position < 0.2:  # Near recent lows
                momentum_score += 15
            
            # Volatility expansion component
            if volatility_ratio > 1.5:
                momentum_score += 10
            
            # Only return if we have a meaningful score
            if momentum_score >= 30:  # Minimum threshold
                return {
                    'symbol': symbol,
                    'score': momentum_score,
                    'current_price': current_price,
                    'price_change': price_change,
                    'volume_ratio': volume_ratio,
                    'volatility_ratio': volatility_ratio,
                    'direction': 'bullish' if price_change > 0 else 'bearish',
                    'timestamp': datetime.datetime.now()
                }
            
            return None
            
        except Exception as e:
            print(f"[WARNING] Error in momentum analysis for {symbol}: {e}")
            return None

    def execute_momentum_options_trade(self, momentum_data: Dict):
        """Execute options trade based on momentum signal"""
        try:
            symbol = momentum_data['symbol']
            direction = momentum_data['direction']
            current_price = momentum_data['current_price']
            confidence = min(momentum_data['score'] / 100, 0.95)  # Convert to 0-95%
            
            print(f"\n[SIGNAL] MOMENTUM SIGNAL: {symbol}")
            print(f"   [DIRECTION] Direction: {direction.upper()}")
            print(f"   [PRICE] Price: ${current_price:.2f}")
            print(f"   [SCORE] Score: {momentum_data['score']:.1f}")
            print(f"   [VOLUME] Volume: {momentum_data['volume_ratio']:.1f}x")
            print(f"   [CONFIDENCE] Confidence: {confidence:.1%}")
            
            # Calculate position size based on account equity
            account = self.api.get_account()
            equity = float(account.equity)
            risk_amount = equity * self.risk_per_trade
            
            # For now, simulate options order (since options trading requires special setup)
            # In production, this would be actual options orders
            print(f"   [RISK] Risk Amount: ${risk_amount:.0f}")
            print(f"   [ORDER] Would execute: {direction} options on {symbol}")
            
            # Log the trade decision
            print(f"   [SUCCESS] {symbol} momentum trade logged")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error executing options trade: {e}")
            return False

    def close_all_positions(self):
        """Close all positions before market close - following Alpaca pattern"""
        try:
            positions = self.api.list_positions()
            
            if not positions:
                print("[EMPTY] No positions to close")
                return
            
            for position in positions:
                try:
                    symbol = position.symbol
                    qty = abs(int(float(position.qty)))
                    side = 'sell' if position.side == 'long' else 'buy'
                    
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )
                    
                    print(f"[CLOSE] Closing position: {qty} {symbol} ({side})")
                    
                except Exception as e:
                    print(f"[WARNING] Error closing position {position.symbol}: {e}")
                    
        except Exception as e:
            print(f"[ERROR] Error in close_all_positions: {e}")

    def show_portfolio_status(self):
        """Show current portfolio status"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            portfolio_value = float(account.portfolio_value)
            todays_pl = float(account.todays_pl)
            todays_pl_pct = (todays_pl / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            print(f"\n[PORTFOLIO] Portfolio: ${portfolio_value:,.2f}")
            print(f"[P&L] Today's P&L: ${todays_pl:+,.2f} ({todays_pl_pct:+.2f}%)")
            
            if positions:
                print("[POSITIONS] Current Positions:")
                for pos in positions:
                    unrealized_pl_pct = float(pos.unrealized_plpc) * 100
                    print(f"   {pos.symbol}: {pos.qty} @ ${pos.avg_entry_price} ({unrealized_pl_pct:+.1f}%)")
            else:
                print("[EMPTY] No current positions")
                
        except Exception as e:
            print(f"[WARNING] Error showing portfolio: {e}")


if __name__ == "__main__":
    try:
        bot = MomentumOptionsBot()
        bot.run()
    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
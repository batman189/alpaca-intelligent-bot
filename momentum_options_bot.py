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
        
        # Trading parameters - CONSERVATIVE SETTINGS
        self.max_positions = 3  # Reduced from 5
        self.risk_per_trade = 0.01  # Reduced to 1% from 2%
        self.max_daily_trades = 2  # Prevent day trading violations
        self.min_volume_surge = 3.0  # Increased from 2x to 3x
        self.min_price_move = 0.03   # Increased from 2% to 3%
        self.min_momentum_score = 50  # Higher quality threshold
        
        # Day trading protection
        self.daily_trade_count = 0
        self.last_trade_date = None
        
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
            
            # Only return if we have a HIGH QUALITY score
            if momentum_score >= self.min_momentum_score:  # Much higher threshold
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
            # DAY TRADING PROTECTION
            if not self.can_make_trade():
                print(f"   [BLOCKED] Day trading limit reached ({self.daily_trade_count}/{self.max_daily_trades})")
                return False
            
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
            
            print(f"   [RISK] Risk Amount: ${risk_amount:.0f}")
            
            # Generate options symbol and execute real trade
            options_symbol = self.generate_options_symbol(symbol, direction, current_price)
            if options_symbol:
                success = self.place_options_order(options_symbol, direction, risk_amount, current_price)
                if success:
                    print(f"   [SUCCESS] {symbol} options order placed: {options_symbol}")
                    return True
                else:
                    print(f"   [ERROR] Failed to place {symbol} options order")
                    return False
            else:
                print(f"   [ERROR] Could not generate options symbol for {symbol}")
                return False
            
        except Exception as e:
            print(f"[ERROR] Error executing options trade: {e}")
            return False

    def can_make_trade(self) -> bool:
        """Check if we can make a trade without PDT violations"""
        import datetime
        today = datetime.date.today()
        
        # Reset daily counter if new day
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
        
        # Check if we've hit our daily limit
        if self.daily_trade_count >= self.max_daily_trades:
            return False
        
        # Check existing positions to avoid over-concentration
        try:
            positions = self.api.list_positions()
            if len(positions) >= self.max_positions:
                print(f"   [BLOCKED] Max positions reached ({len(positions)}/{self.max_positions})")
                return False
        except:
            pass  # Continue if can't check positions
        
        return True

    def record_trade(self):
        """Record that a trade was made"""
        self.daily_trade_count += 1
        print(f"   [TRACKING] Daily trades: {self.daily_trade_count}/{self.max_daily_trades}")

    def generate_options_symbol(self, symbol: str, direction: str, current_price: float) -> Optional[str]:
        """Generate proper options symbol for Alpaca trading"""
        try:
            import datetime
            
            # Calculate expiration date (2 weeks from now, next Friday)
            today = datetime.date.today()
            days_until_friday = (4 - today.weekday()) % 7  # Friday is 4
            if days_until_friday == 0:  # If today is Friday, get next Friday
                days_until_friday = 7
            
            # Add 2 weeks to get the Friday after next
            expiration_date = today + datetime.timedelta(days=days_until_friday + 7)
            exp_str = expiration_date.strftime("%y%m%d")  # YYMMDD format
            
            # Calculate strike price based on direction and momentum
            if direction == 'bullish':
                # Slightly OTM calls - 2% above current price
                strike_price = round(current_price * 1.02, 0)
                option_type = 'C'
            else:
                # Slightly OTM puts - 2% below current price  
                strike_price = round(current_price * 0.98, 0)
                option_type = 'P'
            
            # Format strike price (multiply by 1000 for options format)
            strike_formatted = f"{int(strike_price * 1000):08d}"
            
            # Alpaca options format: SYMBOL + YYMMDD + C/P + 8-digit strike
            options_symbol = f"{symbol}{exp_str}{option_type}{strike_formatted}"
            
            print(f"   [OPTIONS] Generated symbol: {options_symbol}")
            print(f"   [STRIKE] Strike: ${strike_price:.0f} ({option_type})")
            print(f"   [EXPIRY] Expiration: {expiration_date}")
            
            return options_symbol
            
        except Exception as e:
            print(f"[ERROR] Error generating options symbol: {e}")
            return None

    def place_options_order(self, options_symbol: str, direction: str, risk_amount: float, current_price: float) -> bool:
        """Place actual options order through Alpaca API"""
        try:
            # CONSERVATIVE premium estimate - options can be expensive!
            estimated_premium = current_price * 0.15  # 15% estimate (more realistic)
            
            # Calculate number of contracts based on risk amount
            # Each options contract represents 100 shares
            contract_cost = estimated_premium * 100
            contracts = max(1, int(risk_amount / contract_cost))
            
            # STRICT limits to prevent large losses
            contracts = min(contracts, 3)  # Max 3 contracts
            
            # Double-check total cost doesn't exceed risk limit
            total_estimated_cost = contract_cost * contracts
            if total_estimated_cost > risk_amount * 1.5:  # 50% buffer
                contracts = max(1, int(risk_amount / contract_cost))
                print(f"   [WARNING] Reduced contracts due to cost: {contracts}")
            
            print(f"   [ESTIMATE] Premium: ${estimated_premium:.2f}")
            print(f"   [CONTRACTS] Quantity: {contracts}")
            print(f"   [COST] Estimated cost: ${contract_cost * contracts:.0f}")
            
            # Submit the options order
            order = self.api.submit_order(
                symbol=options_symbol,
                qty=contracts,
                side='buy',  # Always buying options (calls or puts)
                type='market',
                time_in_force='day'
            )
            
            print(f"   [ORDER] Order ID: {order.id}")
            print(f"   [STATUS] Order submitted successfully")
            
            # Record the trade for day trading protection
            self.record_trade()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to place options order: {e}")
            
            # If options trading fails, could be due to:
            # 1. Options not approved on account
            # 2. Invalid options symbol
            # 3. Market closed or options not available
            print(f"   [INFO] Options trading may not be enabled on account")
            print(f"   [INFO] Check Alpaca dashboard for options approval status")
            
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
#!/usr/bin/env python3
"""
ML-BASED PREDICTIVE OPTIONS TRADING BOT
Uses machine learning to predict price movements BEFORE they happen
Trades options on SPY, UNH, TSLA, QQQ, AAPL, NVDA based on ML predictions
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOptionsBot:
    def __init__(self):
        # Alpaca API setup
        self.api_key = os.environ.get('APCA_API_KEY_ID')
        self.secret_key = os.environ.get('APCA_API_SECRET_KEY') 
        self.base_url = os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")
        
        self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
        
        # HIGH-SWING TICKERS as requested
        self.tickers = ['SPY', 'UNH', 'TSLA', 'QQQ', 'AAPL', 'NVDA']
        
        # ML Models for each ticker
        self.models = {}
        self.scalers = {}
        self.model_trained = {}
        
        # Trading state
        self.predictions = {}
        self.last_training = {}
        
        logger.info(f"🤖 ML Options Bot initialized for: {', '.join(self.tickers)}")

    async def get_market_data(self, symbol, days=30):
        """Get real market data for ML training and prediction"""
        try:
            # Get historical bars for training
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get 1-minute bars for detailed analysis
            bars = self.api.get_bars(
                symbol, 
                '1Min', 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=10000
            ).df
            
            if bars.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return None
            
            # Add technical indicators
            bars = self.add_technical_indicators(bars)
            
            return bars
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    def add_technical_indicators(self, df):
        """Add technical indicators for ML features"""
        try:
            # Price-based features
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_15'] = df['close'].rolling(window=15).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Momentum indicators
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_15'] = df['close'].pct_change(15)
            
            # Volatility indicators
            df['price_volatility'] = df['close'].rolling(window=20).std()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price position indicators
            df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
            df['price_vs_sma15'] = (df['close'] - df['sma_15']) / df['sma_15']
            df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
            
            # Bollinger Bands
            bb_window = 20
            bb_std = df['close'].rolling(window=bb_window).std()
            df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD-like indicator
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    def prepare_ml_features(self, df):
        """Prepare features for ML model"""
        try:
            feature_columns = [
                'sma_5', 'sma_15', 'sma_50', 'rsi', 'momentum_5', 'momentum_15',
                'price_volatility', 'volume_ratio', 'price_vs_sma5', 'price_vs_sma15', 
                'price_vs_sma50', 'bb_position', 'macd', 'macd_signal', 'macd_histogram'
            ]
            
            # Select features that exist
            available_features = [col for col in feature_columns if col in df.columns]
            
            if not available_features:
                logger.error("No technical indicators available for ML")
                return None, None
            
            X = df[available_features].copy()
            
            # Create target: future price movement (15 minutes ahead)
            y = df['close'].shift(-15).pct_change()  # 15-minute future return
            
            # Remove rows with NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                logger.warning("Insufficient data for ML training")
                return None, None
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None, None

    async def train_ml_model(self, symbol):
        """Train ML model for price prediction"""
        try:
            logger.info(f"🧠 Training ML model for {symbol}...")
            
            # Get market data
            df = await self.get_market_data(symbol, days=60)  # 60 days of data
            if df is None or len(df) < 200:
                logger.warning(f"Insufficient data for {symbol}")
                return False
            
            # Prepare features
            X, y = self.prepare_ml_features(df)
            if X is None or y is None:
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False  # Don't shuffle to maintain time order
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble of models
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train models
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Test accuracy
            rf_score = rf_model.score(X_test_scaled, y_test)
            gb_score = gb_model.score(X_test_scaled, y_test)
            
            # Store best model
            if rf_score > gb_score:
                self.models[symbol] = rf_model
                best_score = rf_score
                model_type = "Random Forest"
            else:
                self.models[symbol] = gb_model
                best_score = gb_score
                model_type = "Gradient Boosting"
            
            self.scalers[symbol] = scaler
            self.model_trained[symbol] = True
            self.last_training[symbol] = datetime.now()
            
            logger.info(f"✅ {symbol} model trained: {model_type} (R²: {best_score:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False

    async def predict_price_movement(self, symbol):
        """Predict future price movement using ML"""
        try:
            if symbol not in self.models:
                logger.warning(f"No trained model for {symbol}")
                return None
            
            # Get recent data for prediction
            df = await self.get_market_data(symbol, days=5)  # Recent data
            if df is None or len(df) < 50:
                return None
            
            # Prepare features
            X, _ = self.prepare_ml_features(df)
            if X is None:
                return None
            
            # Use most recent data point
            latest_features = X.iloc[-1:].values
            
            # Scale features
            latest_features_scaled = self.scalers[symbol].transform(latest_features)
            
            # Make prediction
            predicted_return = self.models[symbol].predict(latest_features_scaled)[0]
            
            # Convert to probability and direction
            confidence = min(abs(predicted_return) * 100, 0.95)  # Scale to confidence
            direction = "bullish" if predicted_return > 0 else "bearish"
            
            current_price = df['close'].iloc[-1]
            
            prediction = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_return': predicted_return,
                'direction': direction,
                'confidence': confidence,
                'signal_strength': abs(predicted_return),
                'timestamp': datetime.now()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return None

    def should_trade_options(self, prediction):
        """Decide if we should trade options based on ML prediction"""
        if not prediction:
            return None
        
        symbol = prediction['symbol']
        predicted_return = prediction['predicted_return']
        confidence = prediction['confidence']
        signal_strength = prediction['signal_strength']
        
        # Trading thresholds - looking for strong signals
        min_signal_strength = 0.005  # 0.5% minimum predicted move
        min_confidence = 0.3        # 30% minimum confidence
        
        if signal_strength < min_signal_strength or confidence < min_confidence:
            return {
                'action': 'hold',
                'reason': f"Weak signal: {signal_strength:.1%} strength, {confidence:.1%} confidence"
            }
        
        # Strong bullish signal - buy calls
        if predicted_return > min_signal_strength:
            return {
                'action': 'buy_calls',
                'reason': f"ML predicts {predicted_return:.2%} upward move",
                'confidence': confidence,
                'predicted_move': predicted_return
            }
        
        # Strong bearish signal - buy puts  
        if predicted_return < -min_signal_strength:
            return {
                'action': 'buy_puts',
                'reason': f"ML predicts {predicted_return:.2%} downward move", 
                'confidence': confidence,
                'predicted_move': predicted_return
            }
        
        return {
            'action': 'hold',
            'reason': f"Neutral prediction: {predicted_return:.2%}"
        }

    async def execute_options_trade(self, symbol, decision, current_price):
        """Execute options trade based on ML decision"""
        try:
            if decision['action'] == 'hold':
                return
            
            # Calculate option parameters
            if decision['action'] == 'buy_calls':
                # Slightly OTM calls
                strike = round(current_price * 1.02, 0)  # 2% OTM
                option_type = "CALL"
            else:  # buy_puts
                # Slightly OTM puts
                strike = round(current_price * 0.98, 0)  # 2% OTM  
                option_type = "PUT"
            
            # Calculate position size based on confidence
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Risk 1-5% based on confidence
            risk_pct = 0.01 + (decision['confidence'] * 0.04)
            risk_amount = buying_power * risk_pct
            risk_amount = max(100, min(risk_amount, 2000))  # $100-$2000 range
            
            # Estimate premium and contracts
            estimated_premium = current_price * 0.03  # Rough 3% premium estimate
            contracts = max(1, int(risk_amount / (estimated_premium * 100)))
            
            option_symbol = f"{symbol}_{option_type}_{strike}_14D"
            
            logger.info(f"🎯 ML OPTIONS TRADE: {decision['action'].upper()}")
            logger.info(f"   📊 {symbol}: ${current_price:.2f}")
            logger.info(f"   🤖 ML Prediction: {decision['predicted_move']:.2%} move")  
            logger.info(f"   📈 Confidence: {decision['confidence']:.1%}")
            logger.info(f"   💰 Option: {contracts} contracts of {option_symbol}")
            logger.info(f"   💵 Risk: ${risk_amount:.0f} ({risk_pct:.1%} of account)")
            logger.info(f"   💡 Reason: {decision['reason']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing options trade: {e}")
            return False

    async def trading_cycle(self):
        """Main ML-based trading cycle"""
        logger.info("🤖 Running ML analysis cycle...")
        
        for symbol in self.tickers:
            try:
                # Check if model needs training/retraining
                needs_training = (
                    symbol not in self.models or 
                    (datetime.now() - self.last_training.get(symbol, datetime.min)).hours > 4
                )
                
                if needs_training:
                    await self.train_ml_model(symbol)
                    if not self.model_trained.get(symbol, False):
                        continue
                
                # Get ML prediction
                prediction = await self.predict_price_movement(symbol)
                if not prediction:
                    continue
                
                # Make trading decision
                decision = self.should_trade_options(prediction)
                
                # Log prediction and decision
                if decision['action'] != 'hold':
                    logger.info(f"🚨 {symbol}: ML SIGNAL - {decision['action'].upper()}")
                    logger.info(f"   🤖 Predicted: {prediction['predicted_return']:.2%} move")
                    logger.info(f"   📈 Confidence: {prediction['confidence']:.1%}")
                    
                    # Execute trade
                    await self.execute_options_trade(symbol, decision, prediction['current_price'])
                else:
                    logger.info(f"⚪ {symbol}: ${prediction['current_price']:.2f} - HOLD")
                    logger.info(f"   🤖 ML: {prediction['predicted_return']:+.2%} ({decision['reason']})")
                
            except Exception as e:
                logger.error(f"Error in trading cycle for {symbol}: {e}")
        
        logger.info("-" * 60)

    async def show_portfolio(self):
        """Show portfolio status"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            portfolio_value = float(account.portfolio_value)
            todays_pl = float(account.todays_pl)
            todays_pl_pct = (todays_pl / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            logger.info(f"💼 Portfolio: ${portfolio_value:,.2f} | Today: ${todays_pl:+,.2f} ({todays_pl_pct:+.2f}%)")
            
            if positions:
                for pos in positions:
                    unrealized_pl_pct = float(pos.unrealized_plpc) * 100
                    logger.info(f"   📊 {pos.symbol}: {pos.qty} @ ${pos.avg_entry_price} ({unrealized_pl_pct:+.1f}%)")
            
        except Exception as e:
            logger.error(f"Error showing portfolio: {e}")

    async def run(self):
        """Main bot execution"""
        logger.info("=" * 80)
        logger.info("🤖 ML-BASED PREDICTIVE OPTIONS TRADING BOT")
        logger.info("=" * 80)
        logger.info("Uses machine learning to predict price movements BEFORE they happen")
        logger.info(f"Tracking: {', '.join(self.tickers)}")
        logger.info("Trades options based on ML predictions, not hard rules")
        logger.info("=" * 80)
        
        while True:
            try:
                clock = self.api.get_clock()
                if clock.is_open:
                    await self.trading_cycle()
                    await self.show_portfolio()
                    
                    logger.info("⏳ Next ML analysis in 5 minutes...")
                    await asyncio.sleep(300)  # 5 minutes between cycles
                else:
                    logger.info(f"🕐 Market closed. Next open: {clock.next_open}")
                    await asyncio.sleep(600)  # 10 minutes when closed
                    
            except KeyboardInterrupt:
                logger.info("👋 ML Options Bot stopped")
                break
            except Exception as e:
                logger.error(f"Critical error: {e}")
                await asyncio.sleep(300)

if __name__ == "__main__":
    bot = MLOptionsBot()
    asyncio.run(bot.run())
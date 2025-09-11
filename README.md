# ML-Based Predictive Options Trading Bot

## What it does:
- **Machine Learning models** predict price movements BEFORE they happen
- Tracks **SPY, UNH, TSLA, QQQ, AAPL, NVDA** (high-swing tickers)
- **Auto-trains** ML models every 4 hours on real market data
- **Predictive trading** - catches moves before 2-3% gains, not after
- **Options execution** based on ML confidence levels

## ML Features:
- **Random Forest & Gradient Boosting** ensemble models
- **15+ technical indicators** (RSI, MACD, Bollinger Bands, momentum)
- **Real-time predictions** of 15-minute price movements  
- **Dynamic position sizing** based on ML confidence
- **No fake/mock data** - pure real market analysis

## Environment Variables:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Example Logs:
```
🤖 Training ML model for UNH...
✅ UNH model trained: Random Forest (R²: 0.673)
🚨 UNH: ML SIGNAL - BUY_CALLS
   🤖 Predicted: +1.8% move
   📈 Confidence: 75%
🎯 ML OPTIONS TRADE: BUY_CALLS
   💰 Option: 3 contracts of UNH_CALL_460_14D
```

## Deploy to Render:
1. Set environment variables in Render
2. Start command: `python app.py`
3. Build command: `pip install -r requirements.txt`

**Machine learning that actually predicts moves instead of reacting to them.**
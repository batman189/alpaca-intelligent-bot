# Simple Options Trading Bot

## What it does:
- Watches **UNH** and **TSLA** for big moves
- **UNH up $5+ or 2%+** → BUY CALLS
- **TSLA up $10+ or 3%+** → BUY CALLS

## Environment Variables:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Run locally:
```bash
python app.py
```

## Deploy to Render:
1. Set environment variables in Render
2. Start command: `python app.py`
3. Build command: `pip install -r requirements.txt`

That's it. No bloat, no complex ML, just simple options trading on big moves.
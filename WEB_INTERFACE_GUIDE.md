# Trading Bot Web Interface Guide

## Quick Start

### Start Web Interface
```bash
python web_interface.py
```

### Access URLs
- **Main Dashboard:** http://localhost:5000/
- **Health Check:** http://localhost:5000/health
- **Configuration:** http://localhost:5000/config
- **Performance:** http://localhost:5000/performance
- **Intelligence:** http://localhost:5000/intelligence
- **Trades:** http://localhost:5000/trades

## Port Configuration

### Default Port: 5000
The web interface runs on port **5000** by default.

### Change Port
Set the `PORT` environment variable:

```bash
# Option 1: Set environment variable
set PORT=8080
python web_interface.py

# Option 2: In .env file
PORT=8080

# Option 3: Export (Linux/Mac)
export PORT=8080
python web_interface.py
```

### Port Priority
The web interface checks for ports in this order:
1. `PORT` environment variable
2. Default: `5000`

## URL Consistency

### Will URLs Change?
**No** - URLs are designed to be consistent:

- **Local Development:** Always `http://localhost:[PORT]/`
- **Same Endpoints:** All API paths remain the same regardless of port
- **Environment Variable:** Port only changes if you set `PORT` environment variable

### Production Deployment
For production, you can:
1. **Set PORT environment variable** (recommended for cloud deployment)
2. **Use reverse proxy** (nginx/apache) to map to standard ports (80/443)
3. **Keep default 5000** for local development

## Environment Variables

### Web Interface Configuration
```bash
# .env file
PORT=5000                    # Web server port
WATCHLIST=SPY,QQQ,AAPL      # Symbols to monitor
ENABLE_TRADING=false         # Trading mode (true/false)
ANALYSIS_INTERVAL=300        # Analysis frequency (seconds)
ENABLE_SENIOR_ANALYST=true   # AI analysis (true/false)
CONFIDENCE_THRESHOLD=0.60    # Trading confidence threshold
MAX_POSITIONS=2              # Maximum open positions
```

## API Endpoints

### Health & Status
- `GET /` - HTML Dashboard
- `GET /health` - JSON health status
- `GET /config` - Current configuration
- `GET /performance` - Performance metrics

### Trading & Analysis
- `GET /intelligence` - Senior Analyst status
- `GET /trades` - Trading history
- `GET /test` - API test endpoint

## Dashboard Features

### Real-Time Monitoring
- **Auto-refresh:** Every 30 seconds
- **System Status:** Online/offline indicator
- **Account Metrics:** Equity, buying power, positions
- **Performance:** Win rate, P&L, Sharpe ratio

### Configuration Display
- **Watchlist:** Current symbols being monitored
- **Trading Status:** Enabled/disabled
- **Analysis Settings:** Intervals and thresholds
- **AI Status:** Senior Analyst availability

### Mobile Responsive
- Works on desktop, tablet, and mobile
- Responsive grid layout
- Touch-friendly buttons and navigation

## Integration with Main Bot

### Standalone Mode
- Web interface works independently
- Shows configuration and test data
- Useful for development and testing

### Connected Mode
When main trading bot (`app.py`) is running:
- Real-time data from trading bot
- Live performance metrics
- Actual account information
- Trading history and positions

### Both Running Together
```bash
# Terminal 1: Start main bot
python app.py

# Terminal 2: Start web interface
python web_interface.py
```

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
netstat -an | findstr :5000

# Use different port
set PORT=8080
python web_interface.py
```

### Cannot Access from Browser
1. **Check if server is running:** Look for "Running on http://localhost:5000"
2. **Try localhost:** http://localhost:5000 (not 127.0.0.1)
3. **Check firewall:** Windows Defender might block the connection
4. **Use different port:** Set PORT environment variable

### Unicode Display Issues
If you see Unicode errors in console:
- **Web interface still works** - errors are cosmetic
- **Use browser** instead of command line for viewing
- **Dashboard displays correctly** in web browser

## Security Notes

### Development Only
- Current setup is for **development use only**
- Uses Flask development server
- No authentication/authorization

### Production Considerations
For production deployment:
1. **Use production WSGI server** (gunicorn, waitress)
2. **Add authentication** for sensitive endpoints
3. **Use HTTPS** for secure connections
4. **Implement rate limiting** for API endpoints

## Support

### Web Interface Issues
- Check server logs in terminal
- Verify .env configuration
- Test with simple_web_test.py first

### Trading Bot Issues
- Check app.py logs
- Verify Alpaca API credentials
- Review trading_bot.log file
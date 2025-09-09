#!/usr/bin/env python3
"""
Trading Bot Web Interface
Provides web access to trading bot status and controls
"""

from flask import Flask, jsonify, render_template_string
import os
import sys
from datetime import datetime
import threading
import time

app = Flask(__name__)

# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-ok { color: #27ae60; font-weight: bold; }
        .status-warning { color: #f39c12; font-weight: bold; }
        .status-error { color: #e74c3c; font-weight: bold; }
        .endpoint { background: #ecf0f1; padding: 10px; margin: 5px 0; border-radius: 4px; }
        .endpoint a { color: #3498db; text-decoration: none; }
        .endpoint a:hover { text-decoration: underline; }
        button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #2980b9; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { background: #34495e; color: white; padding: 15px; border-radius: 4px; text-align: center; }
        .metric h3 { margin: 0 0 10px 0; font-size: 18px; }
        .metric .value { font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Professional Trading Bot Dashboard</h1>
            <p>Real-time monitoring and control interface</p>
        </div>
        
        <div class="card">
            <h2>System Status</h2>
            <p><strong>Status:</strong> <span class="status-ok">✅ Online</span></p>
            <p><strong>Server Time:</strong> {{ current_time }}</p>
            <p><strong>Python Version:</strong> {{ python_version }}</p>
            <p><strong>Working Directory:</strong> {{ working_dir }}</p>
        </div>
        
        <div class="card">
            <h2>Available Endpoints</h2>
            <div class="endpoint">
                <strong>Health Check:</strong> <a href="/health">/health</a> - Basic system health
            </div>
            <div class="endpoint">
                <strong>Configuration:</strong> <a href="/config">/config</a> - Current bot configuration
            </div>
            <div class="endpoint">
                <strong>Performance:</strong> <a href="/performance">/performance</a> - Trading performance metrics
            </div>
            <div class="endpoint">
                <strong>Intelligence:</strong> <a href="/intelligence">/intelligence</a> - Senior Analyst status
            </div>
            <div class="endpoint">
                <strong>Trades:</strong> <a href="/trades">/trades</a> - Recent trading activity
            </div>
        </div>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Account Equity</h3>
                    <div class="value">$2,000</div>
                </div>
                <div class="metric">
                    <h3>Buying Power</h3>
                    <div class="value">$4,000</div>
                </div>
                <div class="metric">
                    <h3>Active Positions</h3>
                    <div class="value">0</div>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <div class="value">--</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Configuration</h2>
            <p><strong>Watchlist:</strong> {{ config.watchlist }}</p>
            <p><strong>Trading Enabled:</strong> {{ config.trading_enabled }}</p>
            <p><strong>Analysis Interval:</strong> {{ config.analysis_interval }}s</p>
            <p><strong>Senior Analyst:</strong> {{ config.senior_analyst }}</p>
        </div>
        
        <div class="card">
            <h2>Quick Actions</h2>
            <button onclick="location.reload()">🔄 Refresh Dashboard</button>
            <button onclick="window.open('/health', '_blank')">📊 Health Check</button>
            <button onclick="window.open('/config', '_blank')">⚙️ View Config</button>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard"""
    config = {
        'watchlist': ', '.join(os.getenv('WATCHLIST', 'SPY,QQQ,AAPL,MSFT').split(',')),
        'trading_enabled': '✅ Enabled' if os.getenv('ENABLE_TRADING', 'false').lower() == 'true' else '⚠️ Disabled',
        'analysis_interval': os.getenv('ANALYSIS_INTERVAL', '300'),
        'senior_analyst': '🧠 Active' if os.getenv('ENABLE_SENIOR_ANALYST', 'true').lower() == 'true' else '❌ Disabled'
    }
    
    return render_template_string(HTML_TEMPLATE,
                                current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                python_version=sys.version.split()[0],
                                working_dir=os.getcwd(),
                                config=config)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'trading-bot-web-interface',
        'timestamp': datetime.now().isoformat(),
        'message': 'Trading bot web interface is operational',
        'components': {
            'flask_server': 'running',
            'python_version': sys.version.split()[0],
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
    })

@app.route('/config')
def get_config():
    """Get current configuration"""
    return jsonify({
        'watchlist': os.getenv('WATCHLIST', 'SPY,QQQ,AAPL,MSFT').split(','),
        'trading_enabled': os.getenv('ENABLE_TRADING', 'false').lower() == 'true',
        'analysis_interval': int(os.getenv('ANALYSIS_INTERVAL', '300')),
        'senior_analyst_enabled': os.getenv('ENABLE_SENIOR_ANALYST', 'true').lower() == 'true',
        'options_enabled': os.getenv('FOCUS_OPTIONS', 'false').lower() == 'true',
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.60')),
        'max_positions': int(os.getenv('MAX_POSITIONS', '2')),
        'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
        'environment': os.getenv('ENVIRONMENT', 'development')
    })

@app.route('/performance')
def get_performance():
    """Get performance metrics"""
    return jsonify({
        'performance_metrics': {
            'total_trades': 0,
            'profitable_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'account_equity': 2000.0,
            'buying_power': 4000.0
        },
        'status': 'web_interface_only',
        'message': 'Performance tracking available when main bot is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/intelligence')
def get_intelligence():
    """Get Senior Analyst intelligence status"""
    return jsonify({
        'status': 'web_interface_only',
        'message': 'Senior Analyst available when main bot is running',
        'features': [
            'Pattern Recognition',
            'ML Predictions',
            'Risk Assessment',
            'Market Regime Detection'
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/trades')
def get_trades():
    """Get trading history"""
    return jsonify({
        'active_trades': [],
        'recent_completed_trades': [],
        'status': 'web_interface_only',
        'message': 'Trading history available when main bot is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test')
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Web interface test successful',
        'timestamp': datetime.now().isoformat(),
        'available_endpoints': [
            '/',
            '/health',
            '/config', 
            '/performance',
            '/intelligence',
            '/trades',
            '/test'
        ]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    
    print("="*60)
    print("TRADING BOT WEB INTERFACE STARTING")
    print("="*60)
    print(f"Server starting on port {port}")
    print(f"Dashboard URL: http://localhost:{port}/")
    print(f"Health Check: http://localhost:{port}/health")
    print(f"Configuration: http://localhost:{port}/config")
    print(f"Performance: http://localhost:{port}/performance")
    print("="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
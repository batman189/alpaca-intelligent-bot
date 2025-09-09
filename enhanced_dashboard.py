#!/usr/bin/env python3
"""
Enhanced Trading Bot Dashboard - All-in-One Page
Complete trading bot monitoring with real-time updates, charts, and controls
"""

from flask import Flask, jsonify, render_template_string
import os
import sys
import json
from datetime import datetime, timedelta
import threading
import time
import random

app = Flask(__name__)

# Enhanced HTML Template with all information on one page
ENHANCED_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Trading Bot - Complete Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0a0e27; color: #fff; }
        
        .dashboard-container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center; }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; background: #00ff88; border-radius: 50%; margin-right: 8px; animation: pulse 2s infinite; }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .grid-container { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .full-width { grid-column: 1 / -1; }
        .half-width { grid-column: span 2; }
        
        .card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; transition: transform 0.2s; }
        .card:hover { transform: translateY(-2px); }
        
        .card-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #64b5f6; display: flex; align-items: center; }
        .card-title::before { content: ''; width: 4px; height: 20px; background: #64b5f6; margin-right: 10px; border-radius: 2px; }
        
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .metric-box { background: rgba(255,255,255,0.08); padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .metric-label { font-size: 12px; color: #aaa; margin-top: 5px; }
        
        .chart-container { position: relative; height: 300px; margin: 10px 0; }
        
        .trade-list { max-height: 300px; overflow-y: auto; }
        .trade-item { display: flex; justify-content: space-between; padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.05); border-radius: 6px; }
        .trade-profit { color: #00ff88; } .trade-loss { color: #ff4444; }
        
        .controls { display: flex; gap: 10px; flex-wrap: wrap; }
        .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.2s; }
        .btn-primary { background: #2196f3; color: white; }
        .btn-success { background: #4caf50; color: white; }
        .btn-warning { background: #ff9800; color: white; }
        .btn-danger { background: #f44336; color: white; }
        .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        
        .config-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .config-value { font-weight: bold; color: #64b5f6; }
        
        .log-container { background: #000; font-family: 'Courier New', monospace; font-size: 12px; height: 200px; overflow-y: auto; padding: 10px; border-radius: 6px; }
        .log-entry { margin: 2px 0; }
        .log-info { color: #64b5f6; } .log-success { color: #00ff88; } .log-warning { color: #ffa726; } .log-error { color: #ff4444; }
        
        .signal-indicator { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .signal-buy { background: #4caf50; } .signal-sell { background: #f44336; } .signal-hold { background: #757575; }
        
        @media (max-width: 768px) {
            .grid-container { grid-template-columns: 1fr; }
            .dashboard-container { padding: 10px; }
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header">
            <h1><span class="status-indicator"></span>Professional Trading Bot Dashboard</h1>
            <p>Real-time monitoring and control • Last updated: <span id="lastUpdate">{{ current_time }}</span></p>
        </div>
        
        <!-- Main Metrics Grid -->
        <div class="grid-container">
            <!-- Account Overview -->
            <div class="card">
                <div class="card-title">Account Overview</div>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value" id="equity">${{ metrics.equity }}</div>
                        <div class="metric-label">Account Equity</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="buyingPower">${{ metrics.buying_power }}</div>
                        <div class="metric-label">Buying Power</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="positions">{{ metrics.positions }}</div>
                        <div class="metric-label">Open Positions</div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Metrics -->
            <div class="card">
                <div class="card-title">Performance</div>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value" id="winRate">{{ metrics.win_rate }}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="totalPnL">${{ metrics.total_pnl }}</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="sharpeRatio">{{ metrics.sharpe_ratio }}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                </div>
            </div>
            
            <!-- System Status -->
            <div class="card">
                <div class="card-title">System Status</div>
                <div class="config-item">
                    <span>Trading Status</span>
                    <span class="config-value">{{ status.trading_status }}</span>
                </div>
                <div class="config-item">
                    <span>Senior Analyst</span>
                    <span class="config-value">{{ status.senior_analyst }}</span>
                </div>
                <div class="config-item">
                    <span>Market Regime</span>
                    <span class="config-value">{{ status.market_regime }}</span>
                </div>
                <div class="config-item">
                    <span>Last Analysis</span>
                    <span class="config-value">{{ status.last_analysis }}</span>
                </div>
            </div>
        </div>
        
        <!-- Charts and Analysis -->
        <div class="grid-container">
            <!-- Performance Chart -->
            <div class="card half-width">
                <div class="card-title">Portfolio Performance</div>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
            
            <!-- Watchlist Signals -->
            <div class="card">
                <div class="card-title">Watchlist Signals</div>
                <div id="watchlistSignals">
                    {% for signal in signals %}
                    <div class="config-item">
                        <span>{{ signal.symbol }}</span>
                        <span class="signal-indicator signal-{{ signal.signal }}">{{ signal.signal.upper() }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <!-- Trading Activity and Controls -->
        <div class="grid-container">
            <!-- Recent Trades -->
            <div class="card">
                <div class="card-title">Recent Trades</div>
                <div class="trade-list" id="recentTrades">
                    {% for trade in recent_trades %}
                    <div class="trade-item">
                        <span>{{ trade.symbol }} - {{ trade.time }}</span>
                        <span class="trade-{{ 'profit' if trade.pnl > 0 else 'loss' }}">${{ trade.pnl }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- Configuration -->
            <div class="card">
                <div class="card-title">Configuration</div>
                <div class="config-item">
                    <span>Watchlist</span>
                    <span class="config-value">{{ config.watchlist }}</span>
                </div>
                <div class="config-item">
                    <span>Analysis Interval</span>
                    <span class="config-value">{{ config.analysis_interval }}s</span>
                </div>
                <div class="config-item">
                    <span>Confidence Threshold</span>
                    <span class="config-value">{{ config.confidence_threshold }}</span>
                </div>
                <div class="config-item">
                    <span>Max Positions</span>
                    <span class="config-value">{{ config.max_positions }}</span>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="card">
                <div class="card-title">Bot Controls</div>
                <div class="controls">
                    <button class="btn btn-success" onclick="toggleTrading()">Toggle Trading</button>
                    <button class="btn btn-primary" onclick="refreshData()">Refresh Data</button>
                    <button class="btn btn-warning" onclick="retrainAnalyst()">Retrain AI</button>
                    <button class="btn btn-danger" onclick="emergencyStop()">Emergency Stop</button>
                </div>
            </div>
        </div>
        
        <!-- System Logs -->
        <div class="grid-container">
            <div class="card full-width">
                <div class="card-title">System Logs</div>
                <div class="log-container" id="systemLogs">
                    <div class="log-entry log-success">[{{ current_time }}] System initialized successfully</div>
                    <div class="log-entry log-info">[{{ current_time }}] Senior Analyst training completed</div>
                    <div class="log-entry log-info">[{{ current_time }}] Market analysis cycle started</div>
                    <div class="log-entry log-success">[{{ current_time }}] Portfolio performance: +2.3%</div>
                    <div class="log-entry log-warning">[{{ current_time }}] High volatility detected in SPY</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Performance Chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: {{ chart_labels | safe }},
                datasets: [{
                    label: 'Portfolio Value',
                    data: {{ chart_data | safe }},
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#fff' } }
                },
                scales: {
                    x: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    y: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });
        
        // Auto-refresh functionality
        function refreshData() {
            location.reload();
        }
        
        function toggleTrading() {
            alert('Trading toggle would be implemented here');
        }
        
        function retrainAnalyst() {
            alert('Senior Analyst retraining would be triggered here');
        }
        
        function emergencyStop() {
            if(confirm('Are you sure you want to emergency stop the bot?')) {
                alert('Emergency stop would be executed here');
            }
        }
        
        // Auto-refresh every 30 seconds
        setTimeout(refreshData, 30000);
        
        // Update timestamp
        document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
"""

@app.route('/')
def enhanced_dashboard():
    """Enhanced all-in-one dashboard"""
    
    # Sample data - would be replaced with real bot data
    metrics = {
        'equity': '2,000',
        'buying_power': '4,000', 
        'positions': '0',
        'win_rate': '65',
        'total_pnl': '+120.50',
        'sharpe_ratio': '1.85'
    }
    
    status = {
        'trading_status': 'Active',
        'senior_analyst': 'Online',
        'market_regime': 'Bull Trending',
        'last_analysis': '2 min ago'
    }
    
    config = {
        'watchlist': 'SPY, QQQ, AAPL, NVDA',
        'analysis_interval': '300',
        'confidence_threshold': '0.60',
        'max_positions': '2'
    }
    
    signals = [
        {'symbol': 'SPY', 'signal': 'buy'},
        {'symbol': 'QQQ', 'signal': 'hold'}, 
        {'symbol': 'AAPL', 'signal': 'sell'},
        {'symbol': 'NVDA', 'signal': 'buy'}
    ]
    
    recent_trades = [
        {'symbol': 'SPY', 'time': '10:30 AM', 'pnl': 45.20},
        {'symbol': 'QQQ', 'time': '09:15 AM', 'pnl': -12.50},
        {'symbol': 'AAPL', 'time': '08:45 AM', 'pnl': 78.30}
    ]
    
    # Generate sample chart data
    chart_labels = [(datetime.now() - timedelta(days=x)).strftime('%m/%d') for x in range(30, 0, -1)]
    chart_data = [2000 + random.randint(-50, 100) * (30-x) / 30 for x in range(30)]
    
    return render_template_string(
        ENHANCED_DASHBOARD,
        current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        metrics=metrics,
        status=status,
        config=config,
        signals=signals,
        recent_trades=recent_trades,
        chart_labels=json.dumps(chart_labels),
        chart_data=json.dumps(chart_data)
    )

@app.route('/api/data')
def get_dashboard_data():
    """API endpoint for real-time data updates"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'equity': 2000 + random.randint(-100, 100),
            'buying_power': 4000,
            'positions': random.randint(0, 3),
            'win_rate': 65 + random.randint(-5, 5),
            'total_pnl': 120.50 + random.randint(-20, 20),
            'sharpe_ratio': 1.85 + random.uniform(-0.2, 0.2)
        },
        'status': 'active',
        'new_trades': []
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))  # Different port to avoid conflicts
    
    print("=" * 60)
    print("ENHANCED TRADING BOT DASHBOARD STARTING")
    print("=" * 60)
    print(f"Dashboard URL: http://localhost:{port}/")
    print(f"API Endpoint: http://localhost:{port}/api/data")
    print("=" * 60)
    print("Features:")
    print("- Real-time performance charts")
    print("- Complete system monitoring")
    print("- Interactive controls")
    print("- Auto-refreshing data")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
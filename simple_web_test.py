#!/usr/bin/env python3
"""
Simple Web Interface Test for Trading Bot
"""

from flask import Flask, jsonify
import os
import sys

app = Flask(__name__)

@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'trading-bot-web-interface',
        'message': 'Web interface is working!',
        'python_version': sys.version,
        'working_directory': os.getcwd()
    })

@app.route('/test')
def test_endpoint():
    return jsonify({
        'status': 'success',
        'message': 'Test endpoint working',
        'available_endpoints': [
            '/',
            '/test',
            '/config',
            '/status'
        ]
    })

@app.route('/config')
def get_config():
    return jsonify({
        'watchlist': os.getenv('WATCHLIST', 'SPY,QQQ,AAPL').split(','),
        'trading_enabled': os.getenv('ENABLE_TRADING', 'false').lower() == 'true',
        'analysis_interval': os.getenv('ANALYSIS_INTERVAL', '300'),
        'environment': 'development'
    })

@app.route('/status')
def get_status():
    return jsonify({
        'bot_status': 'web_interface_only',
        'flask_working': True,
        'message': 'This is a simplified web interface test'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting simple web interface on port {port}")
    print(f"Visit: http://localhost:{port}/")
    print(f"Available endpoints:")
    print(f"  http://localhost:{port}/")
    print(f"  http://localhost:{port}/test")
    print(f"  http://localhost:{port}/config")
    print(f"  http://localhost:{port}/status")
    
    app.run(host='0.0.0.0', port=port, debug=False)
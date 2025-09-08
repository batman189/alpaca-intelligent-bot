#!/usr/bin/env python3
"""
Render-optimized startup script for the trading bot
Handles environment-specific configurations and graceful degradation
"""

import os
import sys
import logging
import tempfile
from datetime import datetime

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def setup_directories_safely():
    """Setup directories with fallbacks for read-only filesystems"""
    directories = ['logs', 'data', 'models', 'monitoring']
    created_dirs = []
    
    for directory in directories:
        try:
            # Try to create in current directory first
            os.makedirs(directory, exist_ok=True)
            
            # Test if we can write to it
            test_file = os.path.join(directory, '.test_write')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                created_dirs.append(directory)
                logger.info(f"✅ Created writable directory: {directory}")
            except (OSError, PermissionError):
                # If we can't write, use temp directory
                temp_dir = tempfile.mkdtemp(prefix=f'{directory}_')
                logger.warning(f"⚠️ Using temp directory for {directory}: {temp_dir}")
                created_dirs.append(temp_dir)
                
        except (OSError, PermissionError) as e:
            # Use temp directory as fallback
            try:
                temp_dir = tempfile.mkdtemp(prefix=f'{directory}_')
                logger.warning(f"⚠️ Failed to create {directory}, using temp: {temp_dir}")
                created_dirs.append(temp_dir)
            except Exception as e2:
                logger.error(f"❌ Could not create any directory for {directory}: {e2}")
                created_dirs.append(None)
    
    return created_dirs

def setup_environment():
    """Setup environment for Render deployment"""
    
    # Set Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Environment-specific settings
    if os.getenv('ENVIRONMENT', 'production') == 'production':
        # Production settings
        os.environ.setdefault('ENABLE_TRADING', 'false')  # Safe default
        os.environ.setdefault('DEBUG', 'false')
        os.environ.setdefault('ANALYSIS_INTERVAL', '300')  # 5 minutes
        os.environ.setdefault('MAX_POSITIONS', '2')  # Conservative
    else:
        # Development settings
        os.environ.setdefault('ENABLE_TRADING', 'false')
        os.environ.setdefault('DEBUG', 'true')
    
    # Set safe defaults
    os.environ.setdefault('PORT', '10000')
    os.environ.setdefault('CONFIDENCE_THRESHOLD', '0.70')
    os.environ.setdefault('WATCHLIST', 'SPY,QQQ,AAPL,MSFT')
    
    logger.info(f"🚀 Starting trading bot on Render")
    logger.info(f"📊 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"🔧 Trading enabled: {os.getenv('ENABLE_TRADING', 'false')}")
    logger.info(f"🎯 Watchlist: {os.getenv('WATCHLIST', 'N/A')}")

def check_dependencies():
    """Check for critical dependencies"""
    required_packages = [
        'pandas', 'numpy', 'flask'
    ]
    
    optional_packages = [
        'sklearn', 'yfinance'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            missing_required.append(package)
            logger.error(f"❌ {package} missing (REQUIRED)")
    
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"⚠️ {package} missing (optional)")
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        return False
    
    if missing_optional:
        logger.warning(f"Missing optional packages: {missing_optional}")
        logger.warning("Bot will run with reduced functionality")
    
    return True

def main():
    """Main startup function"""
    try:
        logger.info("=" * 50)
        logger.info("🤖 PROFESSIONAL TRADING BOT STARTUP")
        logger.info("=" * 50)
        
        # Setup directories first
        setup_directories_safely()
        
        # Setup environment
        setup_environment()
        
        # Check dependencies
        if not check_dependencies():
            logger.error("❌ Critical dependencies missing. Starting fallback web server.")
            
            # Start minimal web server
            try:
                from flask import Flask
                app = Flask(__name__)
                
                @app.route('/')
                def health():
                    return {
                        'status': 'degraded',
                        'message': 'Dependencies missing, running in web-only mode',
                        'timestamp': datetime.now().isoformat()
                    }
                
                @app.route('/health')
                def health_check():
                    return {'status': 'ok', 'mode': 'web-only'}
                
                port = int(os.getenv('PORT', 10000))
                logger.info(f"🌐 Starting fallback web server on port {port}")
                app.run(host='0.0.0.0', port=port)
                
            except Exception as e:
                logger.error(f"❌ Even fallback web server failed: {e}")
                sys.exit(1)
        
        # Import and start the main application
        try:
            from app import run_bot
            logger.info("✅ Bot modules imported successfully")
            
            # Start the bot
            logger.info("🚀 Starting trading bot...")
            run_bot()
            
        except ImportError as e:
            logger.error(f"❌ Failed to import bot modules: {e}")
            
            # Try to start just the web server as fallback
            try:
                from flask import Flask
                app = Flask(__name__)
                
                @app.route('/')
                def health():
                    return {
                        'status': 'degraded',
                        'message': 'Bot modules failed to load, running in web-only mode',
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    }
                
                @app.route('/health')
                def health_check():
                    return {'status': 'ok', 'mode': 'web-only'}
                
                port = int(os.getenv('PORT', 10000))
                logger.info(f"🌐 Starting fallback web server on port {port}")
                app.run(host='0.0.0.0', port=port)
                
            except Exception as e2:
                logger.error(f"❌ Even fallback web server failed: {e2}")
                sys.exit(1)
        
        except Exception as e:
            logger.error(f"❌ Bot startup failed: {e}")
            
            # Emergency web server
            try:
                from flask import Flask
                app = Flask(__name__)
                
                @app.route('/')
                def emergency():
                    return {
                        'status': 'error',
                        'message': f'Bot failed to start: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                
                @app.route('/health')
                def emergency_health():
                    return {'status': 'error', 'mode': 'emergency'}
                
                port = int(os.getenv('PORT', 10000))
                logger.info(f"🚑 Starting emergency web server on port {port}")
                app.run(host='0.0.0.0', port=port)
                
            except Exception as e2:
                logger.error(f"❌ Emergency server failed: {e2}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"💥 Critical startup failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

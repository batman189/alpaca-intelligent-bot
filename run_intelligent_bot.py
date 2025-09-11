#!/usr/bin/env python3
"""
INTELLIGENT OPTIONS BOT RUNNER
Launch script for the sophisticated options trading bot

This script starts the intelligent options bot with proper error handling,
logging, and monitoring capabilities.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
import signal
import traceback
from aggressive_trading_bot import AggressiveTradingBot

class BotRunner:
    """Bot runner with proper lifecycle management"""
    
    def __init__(self):
        self.bot = None
        self.running = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Setup logging configuration
        log_filename = f"logs/intelligent_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"🚀 Intelligent Options Bot Runner Starting")
        logger.info(f"📁 Log file: {log_filename}")
        
        return logger
    
    def _validate_environment(self) -> bool:
        """Validate required environment variables"""
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY',
            'ALPACA_BASE_URL'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        self.logger.info("✅ Environment validation passed")
        return True
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            self.logger.info(f"📡 Received signal {sig}, initiating graceful shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize_bot(self):
        """Initialize the intelligent options bot"""
        try:
            self.logger.info("🤖 Initializing Intelligent Options Bot...")
            
            self.bot = AggressiveTradingBot()
            
            self.logger.info("✅ Bot initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize bot: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def run_bot(self):
        """Main bot execution loop"""
        self.running = True
        self.logger.info("🚀 Starting intelligent options trading...")
        
        try:
            while self.running:
                try:
                    # Run one trading cycle
                    await self.bot.trading_cycle()
                    
                    # Wait before next cycle (60 seconds)
                    for _ in range(60):
                        if not self.running:
                            break
                        await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in trading cycle: {e}")
                    self.logger.error(traceback.format_exc())
                    
                    # Wait 5 minutes on error
                    self.logger.info("⏳ Waiting 5 minutes before retry...")
                    for _ in range(300):
                        if not self.running:
                            break
                        await asyncio.sleep(1)
        
        except Exception as e:
            self.logger.error(f"❌ Critical error in bot execution: {e}")
            self.logger.error(traceback.format_exc())
        
        finally:
            self.logger.info("🛑 Bot execution stopped")
    
    async def shutdown_bot(self):
        """Graceful bot shutdown"""
        if self.bot:
            try:
                self.logger.info("📊 Generating final performance report...")
                await self.bot.show_portfolio_status()
                
                self.logger.info("✅ Bot shutdown complete")
                
            except Exception as e:
                self.logger.error(f"❌ Error during shutdown: {e}")
    
    async def run(self):
        """Main execution method"""
        try:
            # Validate environment
            if not self._validate_environment():
                sys.exit(1)
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Initialize bot
            if not await self.initialize_bot():
                sys.exit(1)
            
            # Run bot
            await self.run_bot()
            
        except KeyboardInterrupt:
            self.logger.info("👋 Keyboard interrupt received")
        
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            self.logger.error(traceback.format_exc())
            sys.exit(1)
        
        finally:
            await self.shutdown_bot()

def main():
    """Main entry point"""
    # Print startup banner
    print("=" * 60)
    print("🧠 INTELLIGENT OPTIONS TRADING BOT")
    print("=" * 60)
    print("Advanced ML-based options trading with pattern recognition")
    print("Real-time learning and adaptive decision making")
    print("NO HARDCODED RULES - Pure intelligence")
    print("=" * 60)
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Load environment variables from .env file
    env_file = "intelligent-trading-bot.env"
    if os.path.exists(env_file):
        print(f"📁 Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    else:
        print("⚠️  No .env file found - using system environment variables")
    
    # Create and run bot
    runner = BotRunner()
    
    # Check if we're in a Jupyter notebook or similar
    try:
        if hasattr(__builtins__, '__IPYTHON__'):
            # Running in Jupyter - create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(runner.run())
        else:
            # Normal execution
            asyncio.run(runner.run())
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
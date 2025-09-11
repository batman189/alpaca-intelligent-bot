#!/usr/bin/env python3
"""
Render startup script for ML Options Bot
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the bot
from ml_options_bot import MLOptionsBot
import asyncio

if __name__ == "__main__":
    bot = MLOptionsBot()
    asyncio.run(bot.run())
#!/usr/bin/env python3
"""
ML-BASED PREDICTIVE OPTIONS TRADING BOT
Uses machine learning to predict price movements and trade options
"""

from ml_options_bot import MLOptionsBot
import asyncio

if __name__ == "__main__":
    bot = MLOptionsBot()
    asyncio.run(bot.run())
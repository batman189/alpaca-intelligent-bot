"""
REMOVED: Sample data generation - NO FAKE DATA ALLOWED IN PRODUCTION

This module previously generated synthetic market data which was dangerous 
for trading operations. All functions now raise exceptions to prevent 
accidental use of fake data.

Use real data sources (Alpaca API, Yahoo Finance) instead.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    """REMOVED: Sample data generation is not allowed for trading safety"""
    raise Exception("SAMPLE DATA GENERATION DISABLED - Use real market data sources only")

def get_historical_data(symbol):
    """REMOVED: This function used fake data - use real data manager instead"""
    raise Exception(f"FAKE HISTORICAL DATA DISABLED - Use MultiSourceDataManager.get_market_data() for real {symbol} data")

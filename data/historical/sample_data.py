import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate sample historical data for training"""
    # Create date range for the past year
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    
    # Generate sample price data
    np.random.seed(42)
    base_price = 450  # SPY around $450
    returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.lognormal(15, 1, len(dates))  # Realistic volume
    }, index=dates)
    
    return df

def get_historical_data(symbol):
    """Get historical data for a symbol (uses sample data for now)"""
    # In a real scenario, you'd load from CSV files or database
    # For now, we'll use generated sample data
    return generate_sample_data()

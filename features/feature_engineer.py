import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.technical_indicators = [
            'rsi', 'macd', 'bbands', 'atr', 'obv', 'vwap', 
            'stoch', 'adx', 'cci', 'williams_r'
        ]
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Try to calculate technical indicators
        try:
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.obv(append=True)
            
            # Additional features if indicators were calculated
            if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns:
                df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
                df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
                
        except Exception as e:
            logger.warning(f"Technical indicators failed: {e}. Using basic features only.")
        
        # Volume features
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        return df
        
    def create_market_context_features(self, primary_data: Dict, spy_data: pd.DataFrame) -> pd.DataFrame:
        if primary_data is None or spy_data is None:
            return None
            
        df = primary_data.copy()
        
        # Market relative performance
        df['spy_returns'] = spy_data['close'].pct_change()
        df['alpha'] = df['returns'] - df['spy_returns']
        df['relative_strength'] = df['close'] / spy_data['close'] * 100
        
        return df
        
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return None
            
        latest_data = df.iloc[[-1]].copy()
        
        # Basic features that should always be available
        base_features = ['close', 'volume', 'returns', 'volatility', 'momentum', 
                        'volume_ma_ratio', 'volume_spike']
        
        # Technical indicators that might be available
        technical_features = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                             'BBU_20_2.0', 'BBL_20_2.0', 'BBM_20_2.0', 'ATRr_14', 'OBV',
                             'bb_width', 'bb_position']
        
        # Only include features that actually exist
        available_features = []
        for feature in base_features + technical_features:
            if feature in latest_data.columns:
                available_features.append(feature)
                
        return latest_data[available_features]

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
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.obv(append=True)
        df.ta.vwap(append=True)
        df.ta.stoch(append=True)
        df.ta.adx(length=14, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.willr(length=14, append=True)
        
        df['price_vs_vwap'] = df['close'] / df['VWAP_D'] - 1
        df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
        df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        return df
        
    def create_market_context_features(self, primary_data: Dict, spy_data: pd.DataFrame) -> pd.DataFrame:
        if primary_data is None or spy_data is None:
            return None
            
        df = primary_data.copy()
        
        df['spy_returns'] = spy_data['close'].pct_change()
        df['alpha'] = df['returns'] - df['spy_returns']
        df['relative_strength'] = df['close'] / spy_data['close'] * 100
        
        returns_correlation = df['returns'].rolling(20).corr(df['spy_returns'])
        df['market_correlation'] = returns_correlation
        
        return df
        
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return None
            
        latest_data = df.iloc[[-1]].copy()
        
        feature_columns = [
            'close', 'volume', 'returns', 'volatility', 'momentum',
            'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'BBU_20_2.0', 'BBL_20_2.0', 'BBM_20_2.0', 'BBP_20_2.0',
            'ATRr_14', 'OBV', 'VWAP_D', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'ADX_14', 'CCI_20_0.015', 'WILLR_14', 'price_vs_vwap',
            'bb_width', 'bb_position', 'volume_ma_ratio', 'volume_spike',
            'alpha', 'relative_strength', 'market_correlation'
        ]
        
        available_features = [col for col in feature_columns if col in latest_data.columns]
        return latest_data[available_features]

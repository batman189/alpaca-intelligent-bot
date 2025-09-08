"""
TA-Lib Fallback Implementation
Provides fallback technical analysis functions when TA-Lib is not available
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Try to import TA-Lib, fall back to custom implementations
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("✅ TA-Lib imported successfully")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("⚠️ TA-Lib not available, using fallback implementations")

def SMA(prices: Union[pd.Series, np.ndarray], timeperiod: int = 20) -> pd.Series:
    """Simple Moving Average"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.SMA(prices.values if isinstance(prices, pd.Series) else prices, timeperiod))
    else:
        return pd.Series(prices).rolling(window=timeperiod).mean()

def EMA(prices: Union[pd.Series, np.ndarray], timeperiod: int = 20) -> pd.Series:
    """Exponential Moving Average"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.EMA(prices.values if isinstance(prices, pd.Series) else prices, timeperiod))
    else:
        return pd.Series(prices).ewm(span=timeperiod, adjust=False).mean()

def RSI(prices: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> pd.Series:
    """Relative Strength Index"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.RSI(prices.values if isinstance(prices, pd.Series) else prices, timeperiod))
    else:
        # Custom RSI implementation
        prices_series = pd.Series(prices)
        delta = prices_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

def MACD(prices: Union[pd.Series, np.ndarray], 
         fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence)"""
    if TALIB_AVAILABLE:
        macd, signal, hist = talib.MACD(prices.values if isinstance(prices, pd.Series) else prices, 
                                      fastperiod, slowperiod, signalperiod)
        return pd.Series(macd), pd.Series(signal), pd.Series(hist)
    else:
        # Custom MACD implementation
        prices_series = pd.Series(prices)
        exp1 = prices_series.ewm(span=fastperiod).mean()
        exp2 = prices_series.ewm(span=slowperiod).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signalperiod).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

def BBANDS(prices: Union[pd.Series, np.ndarray], timeperiod: int = 20, 
           nbdevup: float = 2, nbdevdn: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(prices.values if isinstance(prices, pd.Series) else prices, 
                                          timeperiod, nbdevup, nbdevdn)
        return pd.Series(upper), pd.Series(middle), pd.Series(lower)
    else:
        # Custom Bollinger Bands implementation
        prices_series = pd.Series(prices)
        middle = prices_series.rolling(window=timeperiod).mean()
        std = prices_series.rolling(window=timeperiod).std()
        upper = middle + (std * nbdevup)
        lower = middle - (std * nbdevdn)
        return upper, middle, lower

def STOCH(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
          close: Union[pd.Series, np.ndarray], fastk_period: int = 5, 
          slowk_period: int = 3, slowd_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    if TALIB_AVAILABLE:
        k, d = talib.STOCH(high.values if isinstance(high, pd.Series) else high,
                          low.values if isinstance(low, pd.Series) else low,
                          close.values if isinstance(close, pd.Series) else close,
                          fastk_period, slowk_period, slowd_period)
        return pd.Series(k), pd.Series(d)
    else:
        # Custom Stochastic implementation
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        lowest_low = low_series.rolling(window=fastk_period).min()
        highest_high = high_series.rolling(window=fastk_period).max()
        
        k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.rolling(window=slowk_period).mean()
        d_percent = k_percent.rolling(window=slowd_period).mean()
        
        return k_percent, d_percent

def WILLR(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
          close: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> pd.Series:
    """Williams %R"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.WILLR(high.values if isinstance(high, pd.Series) else high,
                                   low.values if isinstance(low, pd.Series) else low,
                                   close.values if isinstance(close, pd.Series) else close,
                                   timeperiod))
    else:
        # Custom Williams %R implementation
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        highest_high = high_series.rolling(window=timeperiod).max()
        lowest_low = low_series.rolling(window=timeperiod).min()
        
        willr = -100 * ((highest_high - close_series) / (highest_high - lowest_low))
        return willr

def ATR(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
        close: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> pd.Series:
    """Average True Range"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.ATR(high.values if isinstance(high, pd.Series) else high,
                                 low.values if isinstance(low, pd.Series) else low,
                                 close.values if isinstance(close, pd.Series) else close,
                                 timeperiod))
    else:
        # Custom ATR implementation
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift())
        tr3 = abs(low_series - close_series.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=timeperiod).mean()

def OBV(close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray]) -> pd.Series:
    """On Balance Volume"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.OBV(close.values if isinstance(close, pd.Series) else close,
                                 volume.values if isinstance(volume, pd.Series) else volume))
    else:
        # Custom OBV implementation
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)
        
        price_change = close_series.diff()
        obv = (volume_series * np.sign(price_change)).cumsum()
        return obv

def MOM(prices: Union[pd.Series, np.ndarray], timeperiod: int = 10) -> pd.Series:
    """Momentum"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.MOM(prices.values if isinstance(prices, pd.Series) else prices, timeperiod))
    else:
        # Custom momentum implementation
        prices_series = pd.Series(prices)
        return prices_series.diff(timeperiod)

def ROC(prices: Union[pd.Series, np.ndarray], timeperiod: int = 10) -> pd.Series:
    """Rate of Change"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.ROC(prices.values if isinstance(prices, pd.Series) else prices, timeperiod))
    else:
        # Custom ROC implementation
        prices_series = pd.Series(prices)
        return prices_series.pct_change(timeperiod) * 100

def CCI(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
        close: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> pd.Series:
    """Commodity Channel Index"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.CCI(high.values if isinstance(high, pd.Series) else high,
                                 low.values if isinstance(low, pd.Series) else low,
                                 close.values if isinstance(close, pd.Series) else close,
                                 timeperiod))
    else:
        # Custom CCI implementation
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        typical_price = (high_series + low_series + close_series) / 3
        sma_tp = typical_price.rolling(window=timeperiod).mean()
        mad = typical_price.rolling(window=timeperiod).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci

def MFI(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
        close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray], 
        timeperiod: int = 14) -> pd.Series:
    """Money Flow Index"""
    if TALIB_AVAILABLE:
        return pd.Series(talib.MFI(high.values if isinstance(high, pd.Series) else high,
                                 low.values if isinstance(low, pd.Series) else low,
                                 close.values if isinstance(close, pd.Series) else close,
                                 volume.values if isinstance(volume, pd.Series) else volume,
                                 timeperiod))
    else:
        # Custom MFI implementation
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)
        
        typical_price = (high_series + low_series + close_series) / 3
        money_flow = typical_price * volume_series
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(timeperiod).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(timeperiod).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

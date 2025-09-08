import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MomentumStrategy:
    def __init__(self):
        self.strong_trend_threshold = 0.03  # 3% move for strong trend
        self.min_volume_multiplier = 2.0    # 2x average volume
        
    def detect_strong_momentum(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect strong momentum moves that warrant aggressive trading
        Returns trade signal or None if no strong momentum
        """
        try:
            if data is None or len(data) < 10:
                return None
                
            # Calculate recent price change
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            price_change_pct = (current_price - prev_price) / prev_price
            
            # Calculate volume surge
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Strong upward momentum with volume confirmation
            if price_change_pct >= self.strong_trend_threshold and volume_ratio >= self.min_volume_multiplier:
                logger.info(f"🎯 STRONG MOMENTUM: {symbol} +{price_change_pct:.2%} on {volume_ratio:.1f}x volume")
                return {
                    'symbol': symbol,
                    'direction': 'long',
                    'strength': min(1.0, price_change_pct / self.strong_trend_threshold),
                    'confidence_boost': 1.5,  # 50% confidence boost for strong moves
                    'reason': f"strong_uptrend_{price_change_pct:.2%}_volume_{volume_ratio:.1f}x"
                }
                
            # Strong downward momentum with volume confirmation
            elif price_change_pct <= -self.strong_trend_threshold and volume_ratio >= self.min_volume_multiplier:
                logger.info(f"🎯 STRONG MOMENTUM: {symbol} {price_change_pct:.2%} on {volume_ratio:.1f}x volume")
                return {
                    'symbol': symbol,
                    'direction': 'short', 
                    'strength': min(1.0, abs(price_change_pct) / self.strong_trend_threshold),
                    'confidence_boost': 1.5,
                    'reason': f"strong_downtrend_{price_change_pct:.2%}_volume_{volume_ratio:.1f}x"
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error detecting momentum for {symbol}: {e}")
            return None
            
    def should_override_strategy(self, momentum_signal: Dict, current_confidence: float) -> bool:
        """
        Determine if momentum signal is strong enough to override normal strategy
        """
        if momentum_signal and current_confidence < 0.7:
            # Strong momentum with weak normal signal → override
            return True
        return False

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MomentumStrategy:
    def __init__(self):
        self.early_momentum_threshold = 0.008  # 0.8% move for early detection
        self.volume_multiplier = 1.5           # 1.5x average volume
        self.min_bars_for_trend = 3            # Need 3 consecutive bars
        
    def detect_early_momentum(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect EARLY momentum signs before large moves happen
        Looks for consistent buying pressure with increasing volume
        """
        try:
            if data is None or len(data) < 10:
                return None
                
            # Look at recent 5 bars for early momentum signs
            recent_data = data.iloc[-5:] if hasattr(data, 'iloc') else data[-5:]
            
            # Check for consistent direction
            price_changes = recent_data['close'].pct_change().dropna()
            if len(price_changes) < self.min_bars_for_trend:
                return None
                
            # Early upward momentum: consistent green bars with increasing volume
            if all(change > 0 for change in price_changes[-self.min_bars_for_trend:]):
                avg_gain = np.mean(price_changes[-self.min_bars_for_trend:])
                current_volume = recent_data['volume'].iloc[-1]
                avg_volume = recent_data['volume'].rolling(5).mean().iloc[-1]
                
                if avg_gain >= self.early_momentum_threshold and current_volume >= avg_volume * self.volume_multiplier:
                    return {
                        'symbol': symbol,
                        'direction': 'long',
                        'strength': min(1.0, avg_gain / self.early_momentum_threshold),
                        'reason': f"early_uptrend_{avg_gain:.2%}_volume_{current_volume/avg_volume:.1f}x"
                    }
            
            # Early downward momentum: consistent red bars with increasing volume  
            elif all(change < 0 for change in price_changes[-self.min_bars_for_trend:]):
                avg_loss = np.mean(price_changes[-self.min_bars_for_trend:])
                current_volume = recent_data['volume'].iloc[-1]
                avg_volume = recent_data['volume'].rolling(5).mean().iloc[-1]
                
                if abs(avg_loss) >= self.early_momentum_threshold and current_volume >= avg_volume * self.volume_multiplier:
                    return {
                        'symbol': symbol,
                        'direction': 'short',
                        'strength': min(1.0, abs(avg_loss) / self.early_momentum_threshold),
                        'reason': f"early_downtrend_{avg_loss:.2%}_volume_{current_volume/avg_volume:.1f}x"
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error detecting early momentum for {symbol}: {e}")
            return None
            
    def should_enhance_confidence(self, momentum_signal: Dict, current_confidence: float) -> bool:
        """
        Enhance confidence for early momentum signals, but don't override good signals
        """
        if momentum_signal and current_confidence < 0.7:
            # Enhance weaker signals with early momentum confirmation
            return True
        return False

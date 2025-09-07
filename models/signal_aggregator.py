"""
Multi-Source Signal Aggregator - Ultimate Trading Intelligence
Combines ALL detection methods with smart deduplication and ranking
COMPLETE VERSION - Fixed async/sync issues and improved error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SignalType(Enum):
    TECHNICAL = "technical"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    OPTIONS_FLOW = "options_flow"
    EARNINGS = "earnings"
    GAP = "gap"
    BREAKOUT = "breakout"
    SENTIMENT = "sentiment"
    PATTERN = "pattern"
    NEWS = "news"

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-100
    direction: str  # "bullish", "bearish", "neutral"
    timeframe: str
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    expected_move: Optional[float] = None
    catalyst: Optional[str] = None
    source_method: str = ""
    detected_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.expires_at is None:
            timeframe_expiry = {
                "1Min": timedelta(minutes=15),
                "5Min": timedelta(hours=1),
                "15Min": timedelta(hours=4),
                "1Hour": timedelta(hours=12),
                "1Day": timedelta(days=3)
            }
            self.expires_at = self.detected_at + timeframe_expiry.get(self.timeframe, timedelta(hours=2))
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at
    
    def get_score(self) -> float:
        """Calculate overall signal score (0-100)"""
        base_score = self.confidence
        strength_multiplier = self.strength.value / 5.0
        
        age_minutes = (datetime.now() - self.detected_at).total_seconds() / 60
        age_penalty = min(age_minutes * 0.5, 20)
        
        return max(0, base_score * strength_multiplier - age_penalty)

class MultiSourceSignalAggregator:
    def __init__(self):
        """Initialize the ultimate signal aggregation system"""
        
        self.active_signals: Dict[str, List[TradingSignal]] = defaultdict(list)
        self.signal_history: List[TradingSignal] = []
        self.dedupe_window = timedelta(minutes=5)
        
        # Signal source weights (how much to trust each detection method)
        self.source_weights = {
            "technical_analysis": 1.0,
            "volume_analysis": 1.2,  # Volume is very reliable
            "momentum_detection": 0.9,
            "options_flow": 1.3,  # Options flow is premium signal
            "earnings_detection": 1.1,
            "gap_analysis": 0.8,
            "breakout_detection": 1.0,
            "pattern_recognition": 0.9,
            "sentiment_analysis": 0.7,  # Sentiment can be noisy
            "news_impact": 0.8
        }
        
        # Lowered thresholds per upgrade recommendations
        self.min_confidence = 40  # Lowered from 75%
        self.min_signals_for_consensus = 2
        self.consensus_threshold = 0.6
        
        self.detection_stats = defaultdict(int)
        self.execution_stats = defaultdict(int)
        self.missed_opportunities = []
        
        logger.info("🎯 Multi-Source Signal Aggregator initialized")

    async def aggregate_signals(self, symbol: str, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Run ALL detection methods simultaneously and aggregate results"""
        all_signals = []
        
        # Run detection methods in parallel
        detection_tasks = [
            self._detect_technical_signals(symbol, market_data),
            self._detect_volume_anomalies(symbol, market_data),
            self._detect_momentum_patterns(symbol, market_data),
            self._detect_gap_setups(symbol, market_data),
            self._detect_breakout_patterns(symbol, market_data),
            self._detect_pattern_signals(symbol, market_data)
        ]
        
        async_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        for i, result in enumerate(async_results):
            if isinstance(result, Exception):
                method_name = detection_tasks[i].__name__ if hasattr(detection_tasks[i], '__name__') else f"method_{i}"
                logger.warning(f"⚠️  {method_name} failed: {result}")
                self.detection_stats[f"{method_name}_errors"] += 1
            elif result:
                all_signals.extend(result)
        
        # Add async detection methods
        try:
            earnings_signals = await self._detect_earnings_opportunities(symbol, market_data)
            all_signals.extend(earnings_signals)
        except Exception as e:
            logger.warning(f"⚠️  Earnings detection failed: {e}")
            
        try:
            options_signals = await self._detect_options_flow(symbol, market_data)
            all_signals.extend(options_signals)
        except Exception as e:
            logger.warning(f"⚠️  Options flow detection failed: {e}")
        
        # Deduplicate and rank signals
        filtered_signals = self._deduplicate_signals(all_signals)
        ranked_signals = self._rank_signals(filtered_signals)
        
        # Update statistics
        self.detection_stats["total_scans"] += 1
        self.detection_stats["signals_detected"] += len(ranked_signals)
        
        self.active_signals[symbol] = ranked_signals
        
        logger.info(f"🎯 {symbol}: Detected {len(ranked_signals)} signals from {len(all_signals)} raw signals")
        
        return ranked_signals

    async def _detect_technical_signals(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect technical analysis signals"""
        signals = []
        
        if len(data) < 20:
            return signals
            
        try:
            # RSI signals
            rsi = self._calculate_rsi(data['close'])
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:  # Oversold
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.TECHNICAL,
                    strength=SignalStrength.MODERATE,
                    confidence=75,
                    direction="bullish",
                    timeframe="15Min",
                    source_method="rsi_oversold",
                    metadata={"rsi": current_rsi}
                ))
            elif current_rsi > 70:  # Overbought
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.TECHNICAL,
                    strength=SignalStrength.MODERATE,
                    confidence=75,
                    direction="bearish",
                    timeframe="15Min",
                    source_method="rsi_overbought",
                    metadata={"rsi": current_rsi}
                ))
            
            # MACD signals
            macd_line, signal_line = self._calculate_macd(data['close'])
            if len(macd_line) >= 2:
                if (macd_line.iloc[-1] > signal_line.iloc[-1] and 
                    macd_line.iloc[-2] <= signal_line.iloc[-2]):
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.TECHNICAL,
                        strength=SignalStrength.STRONG,
                        confidence=80,
                        direction="bullish",
                        timeframe="1Hour",
                        source_method="macd_bullish_cross"
                    ))
            
            # Bollinger Bands squeeze
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['close'])
            current_price = data['close'].iloc[-1]
            
            if current_price <= bb_lower.iloc[-1]:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.TECHNICAL,
                    strength=SignalStrength.STRONG,
                    confidence=85,
                    direction="bullish",
                    timeframe="1Hour",
                    source_method="bollinger_oversold",
                    price_target=bb_middle.iloc[-1]
                ))
                
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            
        return signals

    async def _detect_volume_anomalies(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect volume-based signals"""
        signals = []
        
        if len(data) < 10:
            return signals
            
        try:
            volume_sma = data['volume'].rolling(20).mean()
            current_volume = data['volume'].iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            
            # High volume breakout
            if current_volume > avg_volume * 2:
                price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                
                if abs(price_change) > 0.02:
                    direction = "bullish" if price_change > 0 else "bearish"
                    confidence = min(95, 70 + (current_volume / avg_volume) * 5)
                    
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.VOLUME,
                        strength=SignalStrength.VERY_STRONG,
                        confidence=confidence,
                        direction=direction,
                        timeframe="5Min",
                        source_method="volume_breakout",
                        metadata={
                            "volume_ratio": current_volume / avg_volume,
                            "price_change": price_change
                        }
                    ))
            
            # Volume-price divergence
            recent_prices = data['close'].tail(5)
            recent_volumes = data['volume'].tail(5)
            
            if len(recent_prices) >= 5:
                price_trend = recent_prices.iloc[-1] > recent_prices.iloc[0]
                volume_trend = recent_volumes.mean() > volume_sma.iloc[-5]
                
                # Price going up but volume decreasing (bearish divergence)
                if price_trend and not volume_trend:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.VOLUME,
                        strength=SignalStrength.MODERATE,
                        confidence=65,
                        direction="bearish",
                        timeframe="15Min",
                        source_method="volume_price_divergence"
                    ))
                    
        except Exception as e:
            logger.error(f"Volume analysis error for {symbol}: {e}")
            
        return signals

    async def _detect_momentum_patterns(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect momentum-based signals"""
        signals = []
        
        if len(data) < 15:
            return signals
            
        try:
            returns = data['close'].pct_change()
            momentum_5 = returns.rolling(5).sum()
            momentum_10 = returns.rolling(10).sum()
            
            current_momentum_5 = momentum_5.iloc[-1]
            current_momentum_10 = momentum_10.iloc[-1]
            
            # Strong momentum signal
            if abs(current_momentum_5) > 0.05:
                direction = "bullish" if current_momentum_5 > 0 else "bearish"
                confidence = min(90, 60 + abs(current_momentum_5) * 500)
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.MOMENTUM,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    direction=direction,
                    timeframe="5Min",
                    source_method="momentum_acceleration",
                    metadata={"momentum_5": current_momentum_5}
                ))
            
            # Momentum reversal
            if (current_momentum_5 > 0 and current_momentum_10 < 0) or \
               (current_momentum_5 < 0 and current_momentum_10 > 0):
                direction = "bullish" if current_momentum_5 > 0 else "bearish"
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.MOMENTUM,
                    strength=SignalStrength.MODERATE,
                    confidence=70,
                    direction=direction,
                    timeframe="15Min",
                    source_method="momentum_reversal"
                ))
                
        except Exception as e:
            logger.error(f"Momentum analysis error for {symbol}: {e}")
            
        return signals

    async def _detect_gap_setups(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect gap trading opportunities"""
        signals = []
        
        if len(data) < 5:
            return signals
            
        try:
            current_open = data['open'].iloc[-1]
            previous_close = data['close'].iloc[-2]
            
            gap_percent = (current_open - previous_close) / previous_close
            
            if abs(gap_percent) > 0.02:
                direction = "bullish" if gap_percent > 0 else "bearish"
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.GAP,
                    strength=SignalStrength.MODERATE,
                    confidence=75,
                    direction=direction,
                    timeframe="1Hour",
                    source_method="gap_setup",
                    price_target=current_open * (1 + gap_percent * 0.5),
                    metadata={"gap_percent": gap_percent}
                ))
                
        except Exception as e:
            logger.error(f"Gap analysis error for {symbol}: {e}")
            
        return signals

    async def _detect_breakout_patterns(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect breakout patterns"""
        signals = []
        
        if len(data) < 20:
            return signals
            
        try:
            highs = data['high'].rolling(20).max()
            lows = data['low'].rolling(20).min()
            
            current_price = data['close'].iloc[-1]
            resistance = highs.iloc[-2]
            support = lows.iloc[-2]
            
            # Resistance breakout
            if current_price > resistance * 1.002:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BREAKOUT,
                    strength=SignalStrength.STRONG,
                    confidence=80,
                    direction="bullish",
                    timeframe="1Hour",
                    source_method="resistance_breakout",
                    price_target=resistance * 1.05,
                    stop_loss=resistance * 0.995
                ))
            
            # Support breakdown
            elif current_price < support * 0.998:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BREAKOUT,
                    strength=SignalStrength.STRONG,
                    confidence=80,
                    direction="bearish",
                    timeframe="1Hour",
                    source_method="support_breakdown",
                    price_target=support * 0.95,
                    stop_loss=support * 1.005
                ))
                
        except Exception as e:
            logger.error(f"Breakout analysis error for {symbol}: {e}")
            
        return signals

    async def _detect_pattern_signals(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect chart patterns"""
        signals = []
        
        if len(data) < 30:
            return signals
            
        try:
            recent_highs = data['high'].tail(10)
            recent_lows = data['low'].tail(10)
            
            # Triangle pattern
            if (recent_highs.is_monotonic_decreasing and 
                recent_lows.is_monotonic_increasing):
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.PATTERN,
                    strength=SignalStrength.MODERATE,
                    confidence=65,
                    direction="neutral",
                    timeframe="1Hour",
                    source_method="triangle_pattern"
                ))
            
            # Double bottom pattern (simplified)
            lows = data['low'].rolling(5).min()
            if len(lows) >= 20:
                low_1 = lows.iloc[-20:-10].min()
                low_2 = lows.iloc[-10:].min()
                
                if abs(low_1 - low_2) / low_1 < 0.02:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.PATTERN,
                        strength=SignalStrength.STRONG,
                        confidence=75,
                        direction="bullish",
                        timeframe="1Day",
                        source_method="double_bottom",
                        price_target=low_2 * 1.1
                    ))
                    
        except Exception as e:
            logger.error(f"Pattern analysis error for {symbol}: {e}")
            
        return signals

    async def _detect_earnings_opportunities(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect earnings-related opportunities (placeholder)"""
        signals = []
        # This would connect to earnings calendar API
        return signals

    async def _detect_options_flow(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect options flow signals (placeholder)"""
        signals = []
        # This would connect to options flow data
        return signals

    def _deduplicate_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Remove duplicate/overlapping signals"""
        if not signals:
            return signals
            
        grouped_signals = defaultdict(list)
        for signal in signals:
            key = f"{signal.symbol}_{signal.direction}_{signal.timeframe}"
            grouped_signals[key].append(signal)
        
        deduplicated = []
        for group_signals in grouped_signals.values():
            if len(group_signals) == 1:
                deduplicated.extend(group_signals)
            else:
                best_signal = max(group_signals, key=lambda s: s.get_score())
                deduplicated.append(best_signal)
        
        return deduplicated

    def _rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Rank signals by quality and strength"""
        if not signals:
            return signals
            
        # Apply source weights
        for signal in signals:
            weight = self.source_weights.get(signal.source_method, 1.0)
            signal.confidence = min(100, signal.confidence * weight)
        
        # Filter by minimum confidence
        filtered_signals = [s for s in signals if s.confidence >= self.min_confidence]
        
        # Sort by score (highest first)
        ranked_signals = sorted(filtered_signals, key=lambda s: s.get_score(), reverse=True)
        
        return ranked_signals

    def get_consensus_signals(self, symbol: str) -> List[TradingSignal]:
        """Get signals that have consensus from multiple methods"""
        symbol_signals = self.active_signals.get(symbol, [])
        
        if len(symbol_signals) < self.min_signals_for_consensus:
            return symbol_signals
        
        # Group by direction
        bullish_signals = [s for s in symbol_signals if s.direction == "bullish"]
        bearish_signals = [s for s in symbol_signals if s.direction == "bearish"]
        
        consensus_signals = []
        
        # Check bullish consensus
        if len(bullish_signals) / len(symbol_signals) >= self.consensus_threshold:
            best_bullish = max(bullish_signals, key=lambda s: s.get_score())
            consensus_signals.append(best_bullish)
        
        # Check bearish consensus
        if len(bearish_signals) / len(symbol_signals) >= self.consensus_threshold:
            best_bearish = max(bearish_signals, key=lambda s: s.get_score())
            consensus_signals.append(best_bearish)
        
        return consensus_signals

    def cleanup_expired_signals(self):
        """Remove expired signals"""
        current_time = datetime.now()
        
        for symbol in list(self.active_signals.keys()):
            active = [s for s in self.active_signals[symbol] if not s.is_expired()]
            if active:
                self.active_signals[symbol] = active
            else:
                del self.active_signals[symbol]

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get signal aggregation statistics"""
        total_active = sum(len(signals) for signals in self.active_signals.values())
        
        return {
            "total_active_signals": total_active,
            "symbols_with_signals": len(self.active_signals),
            "detection_stats": dict(self.detection_stats),
            "execution_stats": dict(self.execution_stats),
            "missed_opportunities_count": len(self.missed_opportunities)
        }

    # Technical indicator helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9).mean()
        return macd_line, signal_line

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band, sma

# Backward compatibility alias
SignalAggregator = MultiSourceSignalAggregator

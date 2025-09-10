"""
Multi-Timeframe Scanner - Simultaneous Analysis Across All Timeframes
Scalping, Swing, and Position opportunities detection in parallel
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeframeType(Enum):
    SCALP = "1Min"     # Ultra-short scalping
    SHORT = "5Min"     # Short-term momentum  
    SWING = "15Min"    # Swing trading
    POSITION = "1Hour" # Position trading
    DAILY = "1Day"     # Long-term trends

@dataclass
class TimeframeOpportunity:
    symbol: str
    timeframe: TimeframeType
    opportunity_type: str
    confidence: float
    direction: str  # bullish/bearish
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    expected_duration: Optional[timedelta] = None
    risk_reward_ratio: Optional[float] = None
    volume_confirmation: bool = False
    technical_strength: float = 0.0
    metadata: Dict[str, Any] = None

class MultiTimeframeScanner:
    def __init__(self):
        """Initialize multi-timeframe scanning engine"""
        
        # Timeframe configurations
        self.timeframes = {
            TimeframeType.SCALP: {
                "period": "1Min",
                "lookback": 50,
                "min_confidence": 60,
                "focus": ["momentum", "breakout", "volume"],
                "max_holding_time": timedelta(minutes=15)
            },
            TimeframeType.SHORT: {
                "period": "5Min", 
                "lookback": 100,
                "min_confidence": 65,
                "focus": ["technical", "momentum", "volume"],
                "max_holding_time": timedelta(hours=2)
            },
            TimeframeType.SWING: {
                "period": "15Min",
                "lookback": 200,
                "min_confidence": 70,
                "focus": ["technical", "pattern", "breakout"],
                "max_holding_time": timedelta(days=3)
            },
            TimeframeType.POSITION: {
                "period": "1Hour",
                "lookback": 300,
                "min_confidence": 75,
                "focus": ["technical", "pattern", "trend"],
                "max_holding_time": timedelta(weeks=2)
            },
            TimeframeType.DAILY: {
                "period": "1Day",
                "lookback": 100,
                "min_confidence": 80,
                "focus": ["pattern", "trend", "fundamentals"],
                "max_holding_time": timedelta(days=90)
            }
        }
        
        # Detection weights by timeframe (higher timeframes more reliable)
        self.timeframe_weights = {
            TimeframeType.SCALP: 0.8,
            TimeframeType.SHORT: 1.0,
            TimeframeType.SWING: 1.2,
            TimeframeType.POSITION: 1.3,
            TimeframeType.DAILY: 1.4
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Scanning statistics
        self.scan_stats = {
            "total_scans": 0,
            "opportunities_found": 0,
            "timeframe_breakdown": {tf.value: 0 for tf in TimeframeType}
        }
        
        logger.info("📊 Multi-Timeframe Scanner initialized")

    async def scan_all_timeframes(self, symbol: str, data_manager) -> Dict[TimeframeType, List[TimeframeOpportunity]]:
        """Scan all timeframes simultaneously for opportunities"""
        opportunities = {}
        
        # Create tasks for parallel timeframe analysis
        tasks = []
        for timeframe in TimeframeType:
            task = self._scan_single_timeframe(symbol, timeframe, data_manager)
            tasks.append((timeframe, task))
        
        # Execute all timeframe scans in parallel
        for timeframe, task in tasks:
            try:
                timeframe_opportunities = await task
                opportunities[timeframe] = timeframe_opportunities
                self.scan_stats["timeframe_breakdown"][timeframe.value] += len(timeframe_opportunities)
            except Exception as e:
                logger.error(f"❌ {timeframe.value} scan failed for {symbol}: {e}")
                opportunities[timeframe] = []
        
        # Update statistics
        self.scan_stats["total_scans"] += 1
        total_found = sum(len(opps) for opps in opportunities.values())
        self.scan_stats["opportunities_found"] += total_found
        
        if total_found > 0:
            logger.info(f"🎯 {symbol}: Found {total_found} opportunities across timeframes")
        
        return opportunities

    async def _scan_single_timeframe(self, symbol: str, timeframe: TimeframeType, 
                                   data_manager) -> List[TimeframeOpportunity]:
        """Scan a single timeframe for opportunities"""
        
        config = self.timeframes[timeframe]
        
        try:
            # Get market data for this timeframe
            market_data = await data_manager.get_market_data(
                symbol, config["period"], config["lookback"]
            )
            
            if market_data is None or len(market_data) < 20:
                return []
            
            opportunities = []
            
            # Run detection methods based on timeframe focus
            if "momentum" in config["focus"]:
                momentum_ops = await self._detect_momentum_opportunities(
                    symbol, timeframe, market_data
                )
                opportunities.extend(momentum_ops)
            
            if "breakout" in config["focus"]:
                breakout_ops = await self._detect_breakout_opportunities(
                    symbol, timeframe, market_data
                )
                opportunities.extend(breakout_ops)
            
            if "technical" in config["focus"]:
                technical_ops = await self._detect_technical_opportunities(
                    symbol, timeframe, market_data
                )
                opportunities.extend(technical_ops)
            
            if "pattern" in config["focus"]:
                pattern_ops = await self._detect_pattern_opportunities(
                    symbol, timeframe, market_data
                )
                opportunities.extend(pattern_ops)
            
            if "volume" in config["focus"]:
                volume_ops = await self._detect_volume_opportunities(
                    symbol, timeframe, market_data
                )
                opportunities.extend(volume_ops)
            
            if "trend" in config["focus"]:
                trend_ops = await self._detect_trend_opportunities(
                    symbol, timeframe, market_data
                )
                opportunities.extend(trend_ops)
            
            # Filter by minimum confidence
            filtered_opportunities = [
                op for op in opportunities 
                if op.confidence >= config["min_confidence"]
            ]
            
            # Apply timeframe weighting
            for op in filtered_opportunities:
                op.confidence *= self.timeframe_weights[timeframe]
                op.confidence = min(100, op.confidence)
            
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Single timeframe scan error ({timeframe.value}): {e}")
            return []

    async def _detect_momentum_opportunities(self, symbol: str, timeframe: TimeframeType,
                                          data: pd.DataFrame) -> List[TimeframeOpportunity]:
        """Detect momentum-based opportunities for this timeframe"""
        opportunities = []
        
        try:
            returns = data['close'].pct_change()
            
            # Adjust periods based on timeframe
            if timeframe in [TimeframeType.SCALP, TimeframeType.SHORT]:
                momentum_periods = [3, 5, 10]
            else:
                momentum_periods = [5, 10, 20]
            
            for period in momentum_periods:
                if len(data) >= period + 5:
                    momentum = returns.rolling(period).sum()
                    current_momentum = momentum.iloc[-1]
                    
                    # Threshold based on timeframe
                    threshold = 0.02 if timeframe == TimeframeType.SCALP else 0.03
                    if abs(current_momentum) > threshold:
                        
                        direction = "bullish" if current_momentum > 0 else "bearish"
                        confidence = min(85, 50 + abs(current_momentum) * 1000)
                        
                        # Calculate targets based on momentum
                        current_price = data['close'].iloc[-1]
                        momentum_extension = current_momentum * 0.5
                        target_price = current_price * (1 + momentum_extension)
                        stop_loss = current_price * (1 - abs(momentum_extension) * 0.3)
                        
                        opportunities.append(TimeframeOpportunity(
                            symbol=symbol,
                            timeframe=timeframe,
                            opportunity_type=f"momentum_{period}",
                            confidence=confidence,
                            direction=direction,
                            entry_price=current_price,
                            target_price=target_price,
                            stop_loss=stop_loss,
                            expected_duration=self._get_expected_duration(timeframe),
                            risk_reward_ratio=abs(momentum_extension) / (abs(momentum_extension) * 0.3),
                            metadata={"momentum_value": current_momentum, "period": period}
                        ))
        
        except Exception as e:
            logger.error(f"Momentum detection error: {e}")
        
        return opportunities

    async def _detect_breakout_opportunities(self, symbol: str, timeframe: TimeframeType,
                                           data: pd.DataFrame) -> List[TimeframeOpportunity]:
        """Detect breakout opportunities for this timeframe"""
        opportunities = []
        
        try:
            lookback = min(50, len(data) - 10)
            if lookback < 10:
                return opportunities
                
            highs = data['high'].rolling(lookback).max()
            lows = data['low'].rolling(lookback).min()
            
            current_price = data['close'].iloc[-1]
            resistance = highs.iloc[-2]
            support = lows.iloc[-2]
            
            # Volume confirmation
            volume_avg = data['volume'].rolling(20).mean()
            current_volume = data['volume'].iloc[-1]
            volume_confirmed = current_volume > volume_avg.iloc[-1] * 1.5
            
            # Breakout threshold based on timeframe
            breakout_threshold = 0.001 if timeframe == TimeframeType.SCALP else 0.002
            
            # Resistance breakout
            if current_price > resistance * (1 + breakout_threshold):
                confidence = 70
                if volume_confirmed:
                    confidence += 10
                    
                # Adjust confidence based on timeframe
                if timeframe in [TimeframeType.POSITION, TimeframeType.DAILY]:
                    confidence += 5
                
                target_distance = (resistance - support) * 0.618  # Fibonacci extension
                target_price = resistance + target_distance
                stop_loss = resistance * 0.998
                
                opportunities.append(TimeframeOpportunity(
                    symbol=symbol,
                    timeframe=timeframe,
                    opportunity_type="resistance_breakout",
                    confidence=confidence,
                    direction="bullish",
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_duration=self._get_expected_duration(timeframe),
                    volume_confirmation=volume_confirmed,
                    metadata={
                        "resistance_level": resistance,
                        "support_level": support,
                        "volume_ratio": current_volume / volume_avg.iloc[-1] if volume_avg.iloc[-1] > 0 else 1
                    }
                ))
            
            # Support breakdown
            elif current_price < support * (1 - breakout_threshold):
                confidence = 70
                if volume_confirmed:
                    confidence += 10
                    
                if timeframe in [TimeframeType.POSITION, TimeframeType.DAILY]:
                    confidence += 5
                
                target_distance = (resistance - support) * 0.618
                target_price = support - target_distance
                stop_loss = support * 1.002
                
                opportunities.append(TimeframeOpportunity(
                    symbol=symbol,
                    timeframe=timeframe,
                    opportunity_type="support_breakdown",
                    confidence=confidence,
                    direction="bearish",
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_duration=self._get_expected_duration(timeframe),
                    volume_confirmation=volume_confirmed,
                    metadata={
                        "resistance_level": resistance,
                        "support_level": support,
                        "volume_ratio": current_volume / volume_avg.iloc[-1] if volume_avg.iloc[-1] > 0 else 1
                    }
                ))
                
        except Exception as e:
            logger.error(f"Breakout detection error: {e}")
        
        return opportunities

    async def _detect_technical_opportunities(self, symbol: str, timeframe: TimeframeType,
                                            data: pd.DataFrame) -> List[TimeframeOpportunity]:
        """Detect technical indicator opportunities"""
        opportunities = []
        
        try:
            # RSI opportunities
            rsi = self._calculate_rsi(data['close'])
            if len(rsi) > 0:
                current_rsi = rsi.iloc[-1]
                
                # Oversold condition (adjust threshold for scalping)
                oversold_threshold = 25 if timeframe == TimeframeType.SCALP else 30
                if current_rsi < oversold_threshold:
                    opportunities.append(TimeframeOpportunity(
                        symbol=symbol,
                        timeframe=timeframe,
                        opportunity_type="rsi_oversold",
                        confidence=75,
                        direction="bullish",
                        entry_price=data['close'].iloc[-1],
                        expected_duration=self._get_expected_duration(timeframe),
                        technical_strength=oversold_threshold - current_rsi,
                        metadata={"rsi": current_rsi}
                    ))
                
                # Overbought condition  
                overbought_threshold = 75 if timeframe == TimeframeType.SCALP else 70
                if current_rsi > overbought_threshold:
                    opportunities.append(TimeframeOpportunity(
                        symbol=symbol,
                        timeframe=timeframe,
                        opportunity_type="rsi_overbought",
                        confidence=75,
                        direction="bearish",
                        entry_price=data['close'].iloc[-1],
                        expected_duration=self._get_expected_duration(timeframe),
                        technical_strength=current_rsi - overbought_threshold,
                        metadata={"rsi": current_rsi}
                    ))
            
            # MACD opportunities
            if len(data) >= 26:
                macd_line, signal_line = self._calculate_macd(data['close'])
                
                if len(macd_line) >= 2:
                    # Bullish crossover
                    if (macd_line.iloc[-1] > signal_line.iloc[-1] and 
                        macd_line.iloc[-2] <= signal_line.iloc[-2]):
                        
                        opportunities.append(TimeframeOpportunity(
                            symbol=symbol,
                            timeframe=timeframe,
                            opportunity_type="macd_bullish",
                            confidence=80,
                            direction="bullish",
                            entry_price=data['close'].iloc[-1],
                            expected_duration=self._get_expected_duration(timeframe),
                            technical_strength=macd_line.iloc[-1] - signal_line.iloc[-1],
                            metadata={
                                "macd": macd_line.iloc[-1],
                                "signal": signal_line.iloc[-1]
                            }
                        ))
                    
                    # Bearish crossover
                    elif (macd_line.iloc[-1] < signal_line.iloc[-1] and 
                          macd_line.iloc[-2] >= signal_line.iloc[-2]):
                        
                        opportunities.append(TimeframeOpportunity(
                            symbol=symbol,
                            timeframe=timeframe,
                            opportunity_type="macd_bearish",
                            confidence=80,
                            direction="bearish",
                            entry_price=data['close'].iloc[-1],
                            expected_duration=self._get_expected_duration(timeframe),
                            technical_strength=signal_line.iloc[-1] - macd_line.iloc[-1],
                            metadata={
                                "macd": macd_line.iloc[-1],
                                "signal": signal_line.iloc[-1]
                            }
                        ))
        
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
        
        return opportunities

    async def _detect_pattern_opportunities(self, symbol: str, timeframe: TimeframeType,
                                          data: pd.DataFrame) -> List[TimeframeOpportunity]:
        """Detect chart pattern opportunities"""
        opportunities = []
        
        try:
            if len(data) < 30:
                return opportunities
            
            recent_highs = data['high'].tail(15)
            recent_lows = data['low'].tail(15)
            
            # Ascending triangle (resistance + rising support)
            if (recent_lows.is_monotonic_increasing and 
                recent_highs.tail(10).std() < recent_highs.mean() * 0.01):
                
                opportunities.append(TimeframeOpportunity(
                    symbol=symbol,
                    timeframe=timeframe,
                    opportunity_type="ascending_triangle",
                    confidence=65,
                    direction="bullish",
                    entry_price=data['close'].iloc[-1],
                    target_price=recent_highs.max() * 1.02,
                    stop_loss=recent_lows.iloc[-1] * 0.98,
                    expected_duration=self._get_expected_duration(timeframe),
                    metadata={
                        "pattern": "ascending_triangle",
                        "resistance": recent_highs.max(),
                        "support_trend": "rising"
                    }
                ))
            
            # Descending triangle (support + falling resistance)
            elif (recent_highs.tail(10).is_monotonic_decreasing and
                  recent_lows.tail(10).std() < recent_lows.mean() * 0.01):
                
                opportunities.append(TimeframeOpportunity(
                    symbol=symbol,
                    timeframe=timeframe,
                    opportunity_type="descending_triangle",
                    confidence=65,
                    direction="bearish", 
                    entry_price=data['close'].iloc[-1],
                    target_price=recent_lows.min() * 0.98,
                    stop_loss=recent_highs.iloc[-1] * 1.02,
                    expected_duration=self._get_expected_duration(timeframe),
                    metadata={
                        "pattern": "descending_triangle",
                        "resistance_trend": "falling",
                        "support": recent_lows.min()
                    }
                ))
        
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
        
        return opportunities

    async def _detect_volume_opportunities(self, symbol: str, timeframe: TimeframeType,
                                         data: pd.DataFrame) -> List[TimeframeOpportunity]:
        """Detect volume-based opportunities"""
        opportunities = []
        
        try:
            if len(data) < 20:
                return opportunities
            
            volume_avg = data['volume'].rolling(20).mean()
            current_volume = data['volume'].iloc[-1]
            
            # Volume spike analysis
            volume_ratio = current_volume / volume_avg.iloc[-1] if volume_avg.iloc[-1] > 0 else 1
            if volume_ratio > 2.0:  # 2x average volume
                
                price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                
                if abs(price_change) > 0.015:  # 1.5% price move with volume
                    direction = "bullish" if price_change > 0 else "bearish"
                    confidence = min(90, 60 + volume_ratio * 10)
                    
                    opportunities.append(TimeframeOpportunity(
                        symbol=symbol,
                        timeframe=timeframe,
                        opportunity_type="volume_breakout",
                        confidence=confidence,
                        direction=direction,
                        entry_price=data['close'].iloc[-1],
                        expected_duration=self._get_expected_duration(timeframe),
                        volume_confirmation=True,
                        metadata={
                            "volume_ratio": volume_ratio,
                            "price_change": price_change
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Volume analysis error: {e}")
        
        return opportunities

    async def _detect_trend_opportunities(self, symbol: str, timeframe: TimeframeType,
                                        data: pd.DataFrame) -> List[TimeframeOpportunity]:
        """Detect trend-following opportunities"""
        opportunities = []
        
        try:
            if len(data) < 50:
                return opportunities
            
            # Calculate moving averages for trend detection
            short_ma = data['close'].rolling(10).mean()
            long_ma = data['close'].rolling(50).mean()
            
            if len(short_ma) < 50 or len(long_ma) < 50:
                return opportunities
            
            # Trend strength
            current_short = short_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            previous_short = short_ma.iloc[-5]
            previous_long = long_ma.iloc[-5]
            
            # Strong uptrend
            if (current_short > current_long and 
                current_short > previous_short and
                current_long > previous_long):
                
                trend_strength = (current_short - current_long) / current_long
                confidence = min(85, 60 + trend_strength * 1000)
                
                opportunities.append(TimeframeOpportunity(
                    symbol=symbol,
                    timeframe=timeframe,
                    opportunity_type="uptrend_continuation",
                    confidence=confidence,
                    direction="bullish",
                    entry_price=data['close'].iloc[-1],
                    target_price=data['close'].iloc[-1] * (1 + trend_strength),
                    stop_loss=current_long * 0.98,
                    expected_duration=self._get_expected_duration(timeframe, multiplier=2),
                    metadata={
                        "trend_strength": trend_strength,
                        "short_ma": current_short,
                        "long_ma": current_long
                    }
                ))
            
            # Strong downtrend
            elif (current_short < current_long and
                  current_short < previous_short and
                  current_long < previous_long):
                
                trend_strength = (current_long - current_short) / current_long
                confidence = min(85, 60 + trend_strength * 1000)
                
                opportunities.append(TimeframeOpportunity(
                    symbol=symbol,
                    timeframe=timeframe,
                    opportunity_type="downtrend_continuation",
                    confidence=confidence,
                    direction="bearish",
                    entry_price=data['close'].iloc[-1],
                    target_price=data['close'].iloc[-1] * (1 - trend_strength),
                    stop_loss=current_long * 1.02,
                    expected_duration=self._get_expected_duration(timeframe, multiplier=2),
                    metadata={
                        "trend_strength": trend_strength,
                        "short_ma": current_short,
                        "long_ma": current_long
                    }
                ))
        
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
        
        return opportunities

    def get_best_opportunities_across_timeframes(self, all_opportunities: Dict[TimeframeType, List[TimeframeOpportunity]],
                                               max_per_symbol: int = 3) -> List[TimeframeOpportunity]:
        """Get the best opportunities across all timeframes"""
        
        # Flatten all opportunities
        flat_opportunities = []
        for timeframe_ops in all_opportunities.values():
            flat_opportunities.extend(timeframe_ops)
        
        if not flat_opportunities:
            return []
        
        # Sort by confidence (highest first)
        sorted_opportunities = sorted(flat_opportunities, key=lambda op: op.confidence, reverse=True)
        
        # Take top N opportunities
        return sorted_opportunities[:max_per_symbol]

    def get_timeframe_consensus(self, all_opportunities: Dict[TimeframeType, List[TimeframeOpportunity]]) -> Dict[str, Any]:
        """Analyze consensus across timeframes"""
        
        if not all_opportunities:
            return {"consensus": "none", "strength": 0}
        
        # Count bullish vs bearish signals across timeframes
        bullish_count = 0
        bearish_count = 0
        total_confidence = 0
        total_signals = 0
        
        for timeframe_ops in all_opportunities.values():
            for op in timeframe_ops:
                total_signals += 1
                total_confidence += op.confidence
                
                if op.direction == "bullish":
                    bullish_count += 1
                elif op.direction == "bearish":
                    bearish_count += 1
        
        if total_signals == 0:
            return {"consensus": "none", "strength": 0}
        
        avg_confidence = total_confidence / total_signals
        
        # Determine consensus
        if bullish_count > bearish_count * 2:
            consensus = "bullish"
            strength = (bullish_count / total_signals) * (avg_confidence / 100)
        elif bearish_count > bullish_count * 2:
            consensus = "bearish" 
            strength = (bearish_count / total_signals) * (avg_confidence / 100)
        else:
            consensus = "mixed"
            strength = 0.5
        
        return {
            "consensus": consensus,
            "strength": strength,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "total_signals": total_signals,
            "avg_confidence": avg_confidence
        }

    def _get_expected_duration(self, timeframe: TimeframeType, multiplier: int = 1) -> timedelta:
        """Get expected holding duration for timeframe"""
        base_duration = self.timeframes[timeframe]["max_holding_time"]
        return base_duration * multiplier

    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scanning statistics"""
        return {
            "total_scans": self.scan_stats["total_scans"],
            "total_opportunities": self.scan_stats["opportunities_found"],
            "avg_opportunities_per_scan": (
                self.scan_stats["opportunities_found"] / 
                max(self.scan_stats["total_scans"], 1)
            ),
            "timeframe_breakdown": dict(self.scan_stats["timeframe_breakdown"]),
            "active_timeframes": len([tf for tf in TimeframeType])
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

if __name__ == "__main__":
    # Test the multi-timeframe scanner
    async def test_scanner():
        scanner = MultiTimeframeScanner()
        
        print("🧪 Testing Multi-Timeframe Scanner...")
        
        # REMOVED: MockDataManager - NO FAKE DATA ALLOWED
        # This scanner now requires real data manager to function
        print("❌ Multi-Timeframe Scanner test requires real data manager")
        print("❌ Fake data generation removed for safety")
        return None
        
        # Scan all timeframes
        opportunities = await scanner.scan_all_timeframes("TEST", data_manager)
        
        total_ops = sum(len(ops) for ops in opportunities.values())
        print(f"✅ Found {total_ops} opportunities across all timeframes")
        
        for timeframe, ops in opportunities.items():
            if ops:
                print(f"📊 {timeframe.value}: {len(ops)} opportunities")
                for op in ops[:2]:  # Show first 2
                    print(f"  - {op.opportunity_type}: {op.direction} "
                          f"(confidence: {op.confidence:.1f}%)")
        
        # Test consensus analysis
        consensus = scanner.get_timeframe_consensus(opportunities)
        print(f"🎯 Consensus: {consensus['consensus']} (strength: {consensus['strength']:.2f})")
        
        # Test statistics
        stats = scanner.get_scan_statistics()
        print(f"📈 Scan stats: {stats['avg_opportunities_per_scan']:.1f} opportunities per scan")
    
    asyncio.run(test_scanner())
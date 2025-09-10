"""
Market Regime Detector - Adaptive Strategy Selection
Automatically detect Bull/Bear/Sideways/Volatile markets and adapt strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNCERTAIN = "uncertain"

class RegimeStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4

@dataclass
class RegimeAnalysis:
    regime: MarketRegime
    strength: RegimeStrength
    confidence: float
    duration_days: int
    key_indicators: Dict[str, float]
    recommended_adjustments: Dict[str, Any]
    detected_at: datetime

class MarketRegimeDetector:
    def __init__(self):
        """Initialize market regime detection system"""
        
        # Regime detection parameters
        self.trend_lookback = 20  # Days for trend analysis
        self.volatility_window = 10  # Days for volatility calculation
        self.volume_window = 15  # Days for volume analysis
        
        # Regime thresholds
        self.bull_threshold = 0.15  # 15% trend strength for bull
        self.bear_threshold = -0.15  # -15% trend strength for bear
        self.volatility_threshold = 0.25  # 25% volatility for volatile regime
        self.sideways_threshold = 0.05  # 5% max trend for sideways
        
        # Regime history tracking
        self.regime_history = deque(maxlen=100)  # Track last 100 regime changes
        self.current_regime = None
        self.regime_start_time = None
        
        # Market indicators cache
        self.indicator_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=1)
        
        # Strategy adjustments for each regime
        self.regime_strategies = {
            MarketRegime.BULL: {
                "position_size_multiplier": 1.2,
                "risk_tolerance": 1.3,
                "preferred_strategies": ["momentum", "breakout", "trend_following"],
                "avoid_strategies": ["mean_reversion"],
                "stop_loss_adjustment": 0.9,  # Wider stops in bull markets
                "confidence_threshold": 0.65  # Lower threshold in bull markets
            },
            MarketRegime.BEAR: {
                "position_size_multiplier": 0.7,
                "risk_tolerance": 0.8,
                "preferred_strategies": ["short_selling", "mean_reversion", "defensive"],
                "avoid_strategies": ["momentum", "breakout"],
                "stop_loss_adjustment": 1.2,  # Tighter stops in bear markets
                "confidence_threshold": 0.80  # Higher threshold in bear markets
            },
            MarketRegime.SIDEWAYS: {
                "position_size_multiplier": 1.0,
                "risk_tolerance": 1.1,
                "preferred_strategies": ["mean_reversion", "range_trading", "options_selling"],
                "avoid_strategies": ["trend_following", "momentum"],
                "stop_loss_adjustment": 1.0,
                "confidence_threshold": 0.75
            },
            MarketRegime.VOLATILE: {
                "position_size_multiplier": 0.6,
                "risk_tolerance": 0.7,
                "preferred_strategies": ["scalping", "volatility_trading", "straddles"],
                "avoid_strategies": ["position_trading", "buy_and_hold"],
                "stop_loss_adjustment": 1.3,  # Much tighter stops
                "confidence_threshold": 0.85  # Much higher threshold
            },
            MarketRegime.UNCERTAIN: {
                "position_size_multiplier": 0.5,
                "risk_tolerance": 0.6,
                "preferred_strategies": ["conservative", "cash"],
                "avoid_strategies": ["aggressive"],
                "stop_loss_adjustment": 1.5,
                "confidence_threshold": 0.90  # Highest threshold
            }
        }
        
        logger.info("🧠 Market Regime Detector initialized")

    async def detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> RegimeAnalysis:
        """
        Detect current market regime using multiple market indicators
        
        Args:
            market_data: Dictionary containing data for major indices (SPY, QQQ, IWM, etc.)
        """
        
        try:
            # Calculate multiple regime indicators
            trend_analysis = self._analyze_trend_strength(market_data)
            volatility_analysis = self._analyze_volatility_regime(market_data)
            volume_analysis = self._analyze_volume_patterns(market_data)
            correlation_analysis = self._analyze_market_correlation(market_data)
            breadth_analysis = self._analyze_market_breadth(market_data)
            
            # Combine indicators to determine regime
            regime_scores = self._calculate_regime_scores(
                trend_analysis, volatility_analysis, volume_analysis,
                correlation_analysis, breadth_analysis
            )
            
            # Determine primary regime
            primary_regime = max(regime_scores, key=regime_scores.get)
            regime_confidence = regime_scores[primary_regime]
            
            # Determine regime strength
            strength = self._calculate_regime_strength(regime_confidence, trend_analysis, volatility_analysis)
            
            # Calculate regime duration
            duration = self._calculate_regime_duration(primary_regime)
            
            # Get recommended strategy adjustments
            adjustments = self.regime_strategies.get(primary_regime, {})
            
            # Create regime analysis
            analysis = RegimeAnalysis(
                regime=primary_regime,
                strength=strength,
                confidence=regime_confidence,
                duration_days=duration,
                key_indicators={
                    "trend_strength": trend_analysis.get("overall_trend", 0),
                    "volatility_level": volatility_analysis.get("avg_volatility", 0),
                    "volume_strength": volume_analysis.get("volume_trend", 0),
                    "market_correlation": correlation_analysis.get("avg_correlation", 0),
                    "market_breadth": breadth_analysis.get("breadth_score", 0)
                },
                recommended_adjustments=adjustments,
                detected_at=datetime.now()
            )
            
            # Update regime tracking
            self._update_regime_tracking(analysis)
            
            logger.info(f"🧠 Market regime detected: {primary_regime.value.upper()} "
                       f"(confidence: {regime_confidence:.1f}%, strength: {strength.name})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            
            # Return uncertain regime as fallback
            return RegimeAnalysis(
                regime=MarketRegime.UNCERTAIN,
                strength=RegimeStrength.WEAK,
                confidence=50.0,
                duration_days=0,
                key_indicators={},
                recommended_adjustments=self.regime_strategies[MarketRegime.UNCERTAIN],
                detected_at=datetime.now()
            )

    def _analyze_trend_strength(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze overall market trend strength"""
        trend_signals = []
        
        for symbol, data in market_data.items():
            if data is None or len(data) < self.trend_lookback:
                continue
                
            try:
                # Calculate multiple trend indicators
                close_prices = data['close']
                
                # Simple trend (price change over lookback period)
                simple_trend = (close_prices.iloc[-1] - close_prices.iloc[-self.trend_lookback]) / close_prices.iloc[-self.trend_lookback]
                
                # Moving average trend
                short_ma = close_prices.rolling(5).mean()
                long_ma = close_prices.rolling(self.trend_lookback).mean()
                ma_trend = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
                
                # Trend consistency (how many days in same direction)
                daily_changes = close_prices.pct_change().tail(self.trend_lookback)
                positive_days = (daily_changes > 0).sum()
                trend_consistency = (positive_days / len(daily_changes)) * 2 - 1  # Scale to -1 to 1
                
                # Combine trend measures
                symbol_trend = (simple_trend + ma_trend + trend_consistency * 0.1) / 3
                trend_signals.append(symbol_trend)
                
            except Exception as e:
                logger.warning(f"Trend analysis failed for {symbol}: {e}")
        
        if not trend_signals:
            return {"overall_trend": 0, "trend_consistency": 0}
        
        overall_trend = np.mean(trend_signals)
        trend_consistency = 1 - np.std(trend_signals)  # Lower std = higher consistency
        
        return {
            "overall_trend": overall_trend,
            "trend_consistency": trend_consistency,
            "symbols_analyzed": len(trend_signals)
        }

    def _analyze_volatility_regime(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze market volatility patterns"""
        volatility_measures = []
        
        for symbol, data in market_data.items():
            if data is None or len(data) < self.volatility_window:
                continue
                
            try:
                # Calculate realized volatility
                returns = data['close'].pct_change()
                realized_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
                current_vol = realized_vol.iloc[-1]
                
                # Calculate volatility trend
                vol_trend = (realized_vol.iloc[-1] - realized_vol.iloc[-self.volatility_window//2]) / realized_vol.iloc[-self.volatility_window//2]
                
                # High-low volatility
                hl_vol = ((data['high'] - data['low']) / data['close']).rolling(self.volatility_window).mean()
                
                volatility_measures.append({
                    "realized_vol": current_vol,
                    "vol_trend": vol_trend,
                    "hl_vol": hl_vol.iloc[-1]
                })
                
            except Exception as e:
                logger.warning(f"Volatility analysis failed for {symbol}: {e}")
        
        if not volatility_measures:
            return {"avg_volatility": 0, "vol_trend": 0}
        
        avg_volatility = np.mean([vm["realized_vol"] for vm in volatility_measures])
        avg_vol_trend = np.mean([vm["vol_trend"] for vm in volatility_measures])
        avg_hl_vol = np.mean([vm["hl_vol"] for vm in volatility_measures])
        
        return {
            "avg_volatility": avg_volatility,
            "vol_trend": avg_vol_trend,
            "hl_volatility": avg_hl_vol,
            "is_high_vol": avg_volatility > self.volatility_threshold
        }

    def _analyze_volume_patterns(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze market volume patterns"""
        volume_signals = []
        
        for symbol, data in market_data.items():
            if data is None or len(data) < self.volume_window or 'volume' not in data.columns:
                continue
                
            try:
                volume = data['volume']
                volume_ma = volume.rolling(self.volume_window).mean()
                
                # Volume trend
                current_avg = volume_ma.iloc[-5:].mean()
                historical_avg = volume_ma.iloc[-self.volume_window:-5].mean()
                volume_trend = (current_avg - historical_avg) / historical_avg
                
                # Volume-price relationship
                price_changes = data['close'].pct_change()
                volume_changes = volume.pct_change()
                
                # Correlation between volume and price changes
                vol_price_corr = price_changes.rolling(self.volume_window).corr(volume_changes).iloc[-1]
                
                volume_signals.append({
                    "volume_trend": volume_trend,
                    "vol_price_corr": vol_price_corr if not np.isnan(vol_price_corr) else 0
                })
                
            except Exception as e:
                logger.warning(f"Volume analysis failed for {symbol}: {e}")
        
        if not volume_signals:
            return {"volume_trend": 0, "vol_price_correlation": 0}
        
        avg_volume_trend = np.mean([vs["volume_trend"] for vs in volume_signals])
        avg_vol_price_corr = np.mean([vs["vol_price_corr"] for vs in volume_signals])
        
        return {
            "volume_trend": avg_volume_trend,
            "vol_price_correlation": avg_vol_price_corr,
            "volume_strength": abs(avg_volume_trend)
        }

    def _analyze_market_correlation(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze correlation patterns between market indices"""
        if len(market_data) < 2:
            return {"avg_correlation": 0.5}
        
        try:
            # Create returns matrix
            returns_data = {}
            for symbol, data in market_data.items():
                if data is not None and len(data) >= 20:
                    returns_data[symbol] = data['close'].pct_change().dropna()
            
            if len(returns_data) < 2:
                return {"avg_correlation": 0.5}
            
            # Align data and calculate correlation matrix
            returns_df = pd.DataFrame(returns_data).dropna()
            corr_matrix = returns_df.corr()
            
            # Calculate average correlation (excluding diagonal)
            correlations = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    correlations.append(corr_matrix.iloc[i, j])
            
            avg_correlation = np.mean(correlations) if correlations else 0.5
            
            return {
                "avg_correlation": avg_correlation,
                "correlation_strength": abs(avg_correlation - 0.5) * 2,  # Distance from random
                "market_cohesion": "high" if avg_correlation > 0.7 else "low"
            }
            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
            return {"avg_correlation": 0.5}

    def _analyze_market_breadth(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze market breadth indicators"""
        advancing_count = 0
        declining_count = 0
        total_symbols = 0
        
        for symbol, data in market_data.items():
            if data is None or len(data) < 5:
                continue
                
            try:
                # Check if symbol is advancing or declining over last 5 days
                recent_change = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
                
                total_symbols += 1
                if recent_change > 0.01:  # 1% threshold
                    advancing_count += 1
                elif recent_change < -0.01:
                    declining_count += 1
                    
            except Exception as e:
                logger.warning(f"Breadth analysis failed for {symbol}: {e}")
        
        if total_symbols == 0:
            return {"breadth_score": 0}
        
        advance_decline_ratio = (advancing_count - declining_count) / total_symbols
        breadth_score = advance_decline_ratio  # -1 (all declining) to 1 (all advancing)
        
        return {
            "breadth_score": breadth_score,
            "advancing_count": advancing_count,
            "declining_count": declining_count,
            "total_analyzed": total_symbols
        }

    def _calculate_regime_scores(self, trend_analysis: Dict, volatility_analysis: Dict,
                                volume_analysis: Dict, correlation_analysis: Dict,
                                breadth_analysis: Dict) -> Dict[MarketRegime, float]:
        """Calculate probability scores for each market regime"""
        
        scores = {regime: 0.0 for regime in MarketRegime}
        
        # Trend-based scoring
        trend_strength = trend_analysis.get("overall_trend", 0)
        
        if trend_strength > self.bull_threshold:
            scores[MarketRegime.BULL] += 40 * (trend_strength / self.bull_threshold)
        elif trend_strength < self.bear_threshold:
            scores[MarketRegime.BEAR] += 40 * abs(trend_strength / self.bear_threshold)
        else:
            scores[MarketRegime.SIDEWAYS] += 30
        
        # Volatility-based scoring
        if volatility_analysis.get("is_high_vol", False):
            scores[MarketRegime.VOLATILE] += 35
            scores[MarketRegime.UNCERTAIN] += 15
        else:
            # Low volatility favors trending markets
            if trend_strength > 0:
                scores[MarketRegime.BULL] += 20
            elif trend_strength < 0:
                scores[MarketRegime.BEAR] += 20
            else:
                scores[MarketRegime.SIDEWAYS] += 25
        
        # Volume confirmation
        volume_trend = volume_analysis.get("volume_trend", 0)
        if abs(volume_trend) > 0.1:  # Significant volume change
            if trend_strength > 0 and volume_trend > 0:
                scores[MarketRegime.BULL] += 15
            elif trend_strength < 0 and volume_trend > 0:
                scores[MarketRegime.BEAR] += 15
        
        # Breadth confirmation
        breadth_score = breadth_analysis.get("breadth_score", 0)
        if breadth_score > 0.3:
            scores[MarketRegime.BULL] += 15
        elif breadth_score < -0.3:
            scores[MarketRegime.BEAR] += 15
        else:
            scores[MarketRegime.SIDEWAYS] += 10
        
        # Correlation effects
        avg_correlation = correlation_analysis.get("avg_correlation", 0.5)
        if avg_correlation > 0.8:  # High correlation suggests regime clarity
            # Boost the leading regime
            leading_regime = max(scores, key=scores.get)
            scores[leading_regime] += 10
        elif avg_correlation < 0.3:  # Low correlation suggests uncertainty
            scores[MarketRegime.UNCERTAIN] += 20
            scores[MarketRegime.VOLATILE] += 15
        
        # Normalize scores to percentages
        total_score = sum(scores.values())
        if total_score > 0:
            for regime in scores:
                scores[regime] = (scores[regime] / total_score) * 100
        else:
            # Default to uncertain if no clear signals
            scores[MarketRegime.UNCERTAIN] = 100
        
        return scores

    def _calculate_regime_strength(self, confidence: float, trend_analysis: Dict,
                                 volatility_analysis: Dict) -> RegimeStrength:
        """Calculate the strength of the detected regime"""
        
        trend_strength = abs(trend_analysis.get("overall_trend", 0))
        volatility_level = volatility_analysis.get("avg_volatility", 0)
        
        # Combine confidence with indicator strength
        combined_strength = (confidence / 100) * (1 + trend_strength + volatility_level)
        
        if combined_strength >= 0.8:
            return RegimeStrength.EXTREME
        elif combined_strength >= 0.6:
            return RegimeStrength.STRONG
        elif combined_strength >= 0.4:
            return RegimeStrength.MODERATE
        else:
            return RegimeStrength.WEAK

    def _calculate_regime_duration(self, regime: MarketRegime) -> int:
        """Calculate how long the current regime has been active"""
        if self.current_regime == regime:
            if self.regime_start_time:
                duration = datetime.now() - self.regime_start_time
                return duration.days
        return 0

    def _update_regime_tracking(self, analysis: RegimeAnalysis):
        """Update regime tracking history"""
        
        # Check if regime has changed
        if self.current_regime != analysis.regime:
            # Regime change detected
            logger.info(f"🔄 Regime change: {self.current_regime} -> {analysis.regime.value}")
            self.current_regime = analysis.regime
            self.regime_start_time = datetime.now()
        
        # Add to history
        self.regime_history.append(analysis)

    def get_strategy_adjustments(self, base_strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy adjustments based on current market regime"""
        
        if not self.current_regime:
            return base_strategy_params
        
        adjustments = self.regime_strategies.get(self.current_regime, {})
        adjusted_params = base_strategy_params.copy()
        
        # Apply regime-specific adjustments
        if "position_size_multiplier" in adjustments:
            adjusted_params["position_size"] = (
                base_strategy_params.get("position_size", 1.0) * 
                adjustments["position_size_multiplier"]
            )
        
        if "risk_tolerance" in adjustments:
            adjusted_params["risk_tolerance"] = (
                base_strategy_params.get("risk_tolerance", 1.0) * 
                adjustments["risk_tolerance"]
            )
        
        if "stop_loss_adjustment" in adjustments:
            adjusted_params["stop_loss_multiplier"] = adjustments["stop_loss_adjustment"]
        
        if "confidence_threshold" in adjustments:
            adjusted_params["min_confidence"] = adjustments["confidence_threshold"]
        
        adjusted_params["preferred_strategies"] = adjustments.get("preferred_strategies", [])
        adjusted_params["avoid_strategies"] = adjustments.get("avoid_strategies", [])
        
        return adjusted_params

    def get_regime_history(self, days: int = 30) -> List[RegimeAnalysis]:
        """Get regime history for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            analysis for analysis in self.regime_history 
            if analysis.detected_at >= cutoff_date
        ]

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection performance"""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for analysis in self.regime_history:
            regime = analysis.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        current_duration = self._calculate_regime_duration(self.current_regime) if self.current_regime else 0
        
        return {
            "current_regime": self.current_regime.value if self.current_regime else None,
            "current_regime_duration_days": current_duration,
            "regime_distribution": regime_counts,
            "total_regime_changes": len(self.regime_history),
            "avg_regime_confidence": np.mean([a.confidence for a in self.regime_history]) if self.regime_history else 0
        }

if __name__ == "__main__":
    # Test the market regime detector
    detector = MarketRegimeDetector()
    
    print("🧪 Testing Market Regime Detector...")
    
    # REMOVED: Fake sample data generation - only real market data allowed
    # This detector now requires real market data from data manager
    
    print("⚠️ Market Regime Detector requires real market data to function")
    print("❌ Fake data generation removed for safety")
    print("✅ Use with real data manager for proper operation")
    
    async def test_detection():
        print("❌ Test requires real market data - no fake data available")
        return None
        print(f"🎯 Key indicators: {analysis.key_indicators}")
        
        # Test strategy adjustments
        base_params = {"position_size": 1.0, "risk_tolerance": 1.0}
        adjusted = detector.get_strategy_adjustments(base_params)
        print(f"⚙️  Strategy adjustments: {adjusted}")
    
    import asyncio
    asyncio.run(test_detection())
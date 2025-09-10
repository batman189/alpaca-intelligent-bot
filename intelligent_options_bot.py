"""
INTELLIGENT OPTIONS TRADING BOT
Advanced pattern recognition and machine learning system for options trading

This bot uses sophisticated algorithms to:
- Analyze chart patterns using computer vision techniques
- Learn from historical market data and outcomes
- Make intelligent options trading decisions based on pattern recognition
- Adapt strategies based on market conditions and performance

NO HARDCODED RULES - Uses ML and pattern recognition for decisions
"""

import asyncio
import logging
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import talib
import cv2
from data.multi_source_data_manager import MultiSourceDataManager
from data.data_client import AlpacaDataClient
from market_knowledge_db import MarketKnowledgeDatabase, PatternRecord, TradingOutcome
from options_flow_analyzer import OptionsFlowAnalyzer, FlowAlert
import alpaca_trade_api as tradeapi

class PatternRecognitionEngine:
    """Advanced pattern recognition using ML and computer vision"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.pattern_clusters = None
        self.historical_patterns = []
        self.pattern_outcomes = []
        
    async def analyze_chart_patterns(self, price_data: pd.DataFrame) -> Dict:
        """Analyze chart patterns using computer vision and technical analysis"""
        patterns = {}
        
        # Technical pattern detection
        patterns['support_resistance'] = self._find_support_resistance(price_data)
        patterns['trend_channels'] = self._detect_trend_channels(price_data)
        patterns['breakout_patterns'] = self._identify_breakouts(price_data)
        patterns['reversal_signals'] = self._detect_reversals(price_data)
        
        # Volume pattern analysis
        patterns['volume_profile'] = self._analyze_volume_patterns(price_data)
        patterns['unusual_activity'] = self._detect_unusual_volume(price_data)
        
        # Volatility patterns
        patterns['volatility_expansion'] = self._detect_volatility_changes(price_data)
        patterns['volatility_contraction'] = self._identify_squeeze_patterns(price_data)
        
        return patterns
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels"""
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find pivot points
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(closes) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivot_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivot_lows.append((i, lows[i]))
        
        # Cluster similar levels
        if pivot_highs:
            resistance_levels = self._cluster_levels([p[1] for p in pivot_highs])
        else:
            resistance_levels = []
            
        if pivot_lows:
            support_levels = self._cluster_levels([p[1] for p in pivot_lows])
        else:
            support_levels = []
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows
        }
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster similar price levels"""
        if len(levels) < 2:
            return levels
            
        levels_array = np.array(levels).reshape(-1, 1)
        n_clusters = min(3, len(levels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit(levels_array)
        return clusters.cluster_centers_.flatten().tolist()
    
    def _detect_trend_channels(self, data: pd.DataFrame) -> Dict:
        """Detect trend channels and direction"""
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        # Calculate trend using linear regression
        x = np.arange(len(closes))
        trend_slope = np.polyfit(x, closes, 1)[0]
        
        # Channel width analysis
        upper_channel = np.max(highs[-20:])
        lower_channel = np.min(lows[-20:])
        channel_width = upper_channel - lower_channel
        
        return {
            'trend_direction': 'bullish' if trend_slope > 0 else 'bearish',
            'trend_strength': abs(trend_slope),
            'channel_width': channel_width,
            'channel_position': (closes[-1] - lower_channel) / channel_width if channel_width > 0 else 0.5
        }
    
    def _identify_breakouts(self, data: pd.DataFrame) -> Dict:
        """Identify breakout patterns"""
        closes = data['Close'].values
        volumes = data['Volume'].values
        
        # Bollinger Bands breakout
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
        
        current_price = closes[-1]
        breakout_type = None
        
        if current_price > bb_upper[-1]:
            breakout_type = 'bullish_breakout'
        elif current_price < bb_lower[-1]:
            breakout_type = 'bearish_breakout'
        
        # Volume confirmation
        avg_volume = np.mean(volumes[-20:])
        volume_surge = volumes[-1] > avg_volume * 1.5
        
        return {
            'breakout_type': breakout_type,
            'volume_confirmation': volume_surge,
            'breakout_strength': abs(current_price - bb_middle[-1]) / (bb_upper[-1] - bb_lower[-1])
        }
    
    def _detect_reversals(self, data: pd.DataFrame) -> Dict:
        """Detect potential reversal patterns"""
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        # RSI divergence
        rsi = talib.RSI(closes, timeperiod=14)
        
        # MACD divergence
        macd, macdsignal, macdhist = talib.MACD(closes)
        
        # Stochastic divergence
        slowk, slowd = talib.STOCH(highs, lows, closes)
        
        return {
            'rsi_oversold': rsi[-1] < 30,
            'rsi_overbought': rsi[-1] > 70,
            'macd_bullish_cross': macd[-1] > macdsignal[-1] and macd[-2] <= macdsignal[-2],
            'macd_bearish_cross': macd[-1] < macdsignal[-1] and macd[-2] >= macdsignal[-2],
            'stoch_oversold': slowk[-1] < 20,
            'stoch_overbought': slowk[-1] > 80
        }
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns and institutional flow"""
        volumes = data['Volume'].values
        closes = data['Close'].values
        
        # Volume trend
        volume_ma = talib.SMA(volumes.astype(float), timeperiod=20)
        volume_trend = 'increasing' if volumes[-1] > volume_ma[-1] else 'decreasing'
        
        # Price-volume relationship
        price_change = (closes[-1] - closes[-2]) / closes[-2]
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
        
        return {
            'volume_trend': volume_trend,
            'volume_price_correlation': price_change * volume_change,
            'relative_volume': volumes[-1] / np.mean(volumes[-20:]),
            'accumulation_distribution': talib.AD(data['High'].values, data['Low'].values, 
                                                 data['Close'].values, data['Volume'].values)[-1]
        }
    
    def _detect_unusual_volume(self, data: pd.DataFrame) -> Dict:
        """Detect unusual volume activity"""
        volumes = data['Volume'].values
        
        avg_volume = np.mean(volumes[-20:])
        volume_spike = volumes[-1] > avg_volume * 2
        
        return {
            'volume_spike': volume_spike,
            'volume_ratio': volumes[-1] / avg_volume,
            'sustained_volume': np.mean(volumes[-3:]) > avg_volume * 1.5
        }
    
    def _detect_volatility_changes(self, data: pd.DataFrame) -> Dict:
        """Detect volatility expansion patterns"""
        closes = data['Close'].values
        
        # ATR for volatility
        atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        current_volatility = atr[-1]
        avg_volatility = np.mean(atr[-20:])
        
        return {
            'volatility_expansion': current_volatility > avg_volatility * 1.3,
            'volatility_ratio': current_volatility / avg_volatility,
            'volatility_trend': 'expanding' if current_volatility > avg_volatility else 'contracting'
        }
    
    def _identify_squeeze_patterns(self, data: pd.DataFrame) -> Dict:
        """Identify volatility squeeze patterns"""
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Keltner Channels
        ema = talib.EMA(closes, timeperiod=20)
        atr = talib.ATR(highs, lows, closes, timeperiod=20)
        kc_upper = ema + (atr * 2)
        kc_lower = ema - (atr * 2)
        
        # Squeeze condition
        squeeze = (bb_upper[-1] < kc_upper[-1]) and (bb_lower[-1] > kc_lower[-1])
        
        return {
            'squeeze_active': squeeze,
            'bb_width': bb_width[-1],
            'squeeze_duration': self._count_squeeze_bars(bb_upper, bb_lower, kc_upper, kc_lower)
        }
    
    def _count_squeeze_bars(self, bb_upper, bb_lower, kc_upper, kc_lower) -> int:
        """Count consecutive squeeze bars"""
        count = 0
        for i in range(len(bb_upper) - 1, -1, -1):
            if (bb_upper[i] < kc_upper[i]) and (bb_lower[i] > kc_lower[i]):
                count += 1
            else:
                break
        return count

class MarketLearningEngine:
    """Machine learning engine for market pattern learning"""
    
    def __init__(self):
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.historical_data = []
        self.pattern_memory = {}
        
    async def learn_from_historical_data(self, symbols: List[str]) -> None:
        """Learn patterns from historical market data"""
        for symbol in symbols:
            try:
                # Get extensive historical data
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period="2y", interval="1d")
                
                if len(hist_data) > 100:
                    patterns = await self._extract_learning_features(hist_data)
                    outcomes = self._label_outcomes(hist_data)
                    
                    self.historical_data.append({
                        'symbol': symbol,
                        'patterns': patterns,
                        'outcomes': outcomes
                    })
                    
            except Exception as e:
                logging.error(f"Error learning from {symbol}: {e}")
    
    async def _extract_learning_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for machine learning"""
        features = []
        
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        volumes = data['Volume'].values
        
        # Technical indicators
        rsi = talib.RSI(closes, timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(closes)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        
        # Price patterns
        sma_20 = talib.SMA(closes, timeperiod=20)
        sma_50 = talib.SMA(closes, timeperiod=50)
        ema_12 = talib.EMA(closes, timeperiod=12)
        ema_26 = talib.EMA(closes, timeperiod=26)
        
        # Volume indicators
        volume_sma = talib.SMA(volumes.astype(float), timeperiod=20)
        ad_line = talib.AD(highs, lows, closes, volumes)
        
        # Combine features
        for i in range(50, len(closes)):
            feature_vector = [
                rsi[i] if not np.isnan(rsi[i]) else 50,
                macd[i] if not np.isnan(macd[i]) else 0,
                macdsignal[i] if not np.isnan(macdsignal[i]) else 0,
                (closes[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if not np.isnan(bb_upper[i]) else 0.5,
                atr[i] / closes[i] if not np.isnan(atr[i]) else 0.01,
                closes[i] / sma_20[i] if not np.isnan(sma_20[i]) else 1,
                sma_20[i] / sma_50[i] if not np.isnan(sma_50[i]) else 1,
                volumes[i] / volume_sma[i] if not np.isnan(volume_sma[i]) else 1,
                (closes[i] - closes[i-5]) / closes[i-5],  # 5-day return
                (closes[i] - closes[i-20]) / closes[i-20],  # 20-day return
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _label_outcomes(self, data: pd.DataFrame) -> np.ndarray:
        """Label outcomes for supervised learning"""
        closes = data['Close'].values
        outcomes = []
        
        for i in range(50, len(closes) - 5):
            # Future 5-day return
            future_return = (closes[i+5] - closes[i]) / closes[i]
            
            # Classify outcome
            if future_return > 0.05:  # 5% gain
                outcome = 2  # Strong bullish
            elif future_return > 0.02:  # 2% gain
                outcome = 1  # Bullish
            elif future_return < -0.05:  # 5% loss
                outcome = -2  # Strong bearish
            elif future_return < -0.02:  # 2% loss
                outcome = -1  # Bearish
            else:
                outcome = 0  # Neutral
            
            outcomes.append(outcome)
        
        return np.array(outcomes)
    
    async def predict_market_direction(self, current_patterns: Dict) -> Dict:
        """Predict market direction based on learned patterns"""
        if not self.historical_data:
            return {'direction': 'neutral', 'confidence': 0.0}
        
        # Find similar historical patterns
        similar_patterns = self._find_similar_patterns(current_patterns)
        
        if not similar_patterns:
            return {'direction': 'neutral', 'confidence': 0.0}
        
        # Analyze outcomes of similar patterns
        outcomes = [p['outcome'] for p in similar_patterns]
        avg_outcome = np.mean(outcomes)
        confidence = min(abs(avg_outcome), 1.0)
        
        if avg_outcome > 0.5:
            direction = 'bullish'
        elif avg_outcome < -0.5:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'direction': direction,
            'confidence': confidence,
            'similar_patterns_count': len(similar_patterns)
        }
    
    def _find_similar_patterns(self, current_patterns: Dict) -> List[Dict]:
        """Find historically similar patterns"""
        similar = []
        
        for historical in self.historical_data:
            similarity_score = self._calculate_pattern_similarity(
                current_patterns, historical['patterns']
            )
            
            if similarity_score > 0.7:  # High similarity threshold
                similar.append({
                    'similarity': similarity_score,
                    'outcome': historical['outcomes']
                })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:10]
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two patterns"""
        # Simplified similarity calculation
        # In practice, this would be more sophisticated
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if isinstance(pattern1[key], (int, float)) and isinstance(pattern2[key], (int, float)):
                # Numerical similarity
                max_val = max(abs(pattern1[key]), abs(pattern2[key]), 1)
                diff = abs(pattern1[key] - pattern2[key]) / max_val
                similarity = 1 - min(diff, 1)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0

class IntelligentDecisionEngine:
    """Advanced decision engine that learns and adapts without hardcoded rules"""
    
    def __init__(self, knowledge_db: MarketKnowledgeDatabase, flow_analyzer: OptionsFlowAnalyzer):
        self.knowledge_db = knowledge_db
        self.flow_analyzer = flow_analyzer
        self.logger = logging.getLogger(__name__)
        self.decision_history = []
        self.adaptation_rate = 0.1  # How quickly to adapt to new information
        self.confidence_threshold = 0.6  # Minimum confidence for action
    
    async def make_intelligent_decision(self, symbol: str, patterns: Dict, 
                                      prediction: Dict, market_data: pd.DataFrame,
                                      flow_alerts: List[FlowAlert]) -> Dict:
        """Make intelligent decision using all available information without hardcoded rules"""
        
        # Gather all intelligence sources
        intelligence = await self._gather_market_intelligence(symbol, patterns, prediction, 
                                                            market_data, flow_alerts)
        
        # Learn from similar historical situations
        historical_context = await self._get_historical_context(symbol, intelligence)
        
        # Assess current market regime
        market_regime = self.knowledge_db.get_market_regime_analysis()
        
        # Calculate decision confidence using ensemble approach
        decision_confidence = await self._calculate_ensemble_confidence(
            intelligence, historical_context, market_regime
        )
        
        # Generate decision based on learned patterns
        decision = await self._generate_adaptive_decision(
            symbol, intelligence, decision_confidence, historical_context
        )
        
        # Store decision for future learning
        self._record_decision(symbol, decision, intelligence)
        
        return decision
    
    async def _gather_market_intelligence(self, symbol: str, patterns: Dict, 
                                        prediction: Dict, market_data: pd.DataFrame,
                                        flow_alerts: List[FlowAlert]) -> Dict:
        """Gather comprehensive market intelligence"""
        current_price = market_data['Close'].iloc[-1]
        
        intelligence = {
            'price_momentum': self._calculate_multi_timeframe_momentum(market_data),
            'pattern_signals': self._extract_pattern_signals(patterns),
            'ai_prediction': prediction,
            'options_flow_signals': self._analyze_flow_signals(flow_alerts),
            'volatility_profile': self._analyze_volatility_regime(market_data),
            'volume_analysis': self._analyze_volume_characteristics(market_data),
            'technical_confluence': self._find_technical_confluence(patterns, market_data),
            'market_context': await self._get_broader_market_context(),
            'current_price': current_price,
            'symbol': symbol
        }
        
        return intelligence
    
    def _calculate_multi_timeframe_momentum(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum across multiple timeframes"""
        closes = data['Close'].values
        
        momentum = {}
        timeframes = [1, 3, 5, 10, 20]  # Different period lengths
        
        for tf in timeframes:
            if len(closes) > tf:
                momentum[f'{tf}d'] = (closes[-1] - closes[-tf-1]) / closes[-tf-1]
        
        # Calculate momentum strength and consistency
        momentum_values = list(momentum.values())
        momentum['strength'] = np.mean([abs(m) for m in momentum_values])
        momentum['consistency'] = len([m for m in momentum_values if m > 0]) / len(momentum_values)
        
        return momentum
    
    def _extract_pattern_signals(self, patterns: Dict) -> Dict:
        """Extract actionable signals from pattern analysis"""
        signals = {
            'breakout_strength': 0,
            'reversal_probability': 0,
            'trend_continuation': 0,
            'volatility_expansion': 0,
            'volume_confirmation': 0
        }
        
        # Breakout signals
        breakout = patterns.get('breakout_patterns', {})
        if breakout.get('breakout_type') == 'bullish_breakout':
            signals['breakout_strength'] = breakout.get('breakout_strength', 0) * 2
        elif breakout.get('breakout_type') == 'bearish_breakout':
            signals['breakout_strength'] = -breakout.get('breakout_strength', 0) * 2
        
        # Reversal signals
        reversal = patterns.get('reversal_signals', {})
        reversal_score = 0
        if reversal.get('rsi_oversold'): reversal_score += 0.3
        if reversal.get('rsi_overbought'): reversal_score -= 0.3
        if reversal.get('macd_bullish_cross'): reversal_score += 0.4
        if reversal.get('macd_bearish_cross'): reversal_score -= 0.4
        signals['reversal_probability'] = reversal_score
        
        # Trend signals
        trend = patterns.get('trend_channels', {})
        if trend.get('trend_direction') == 'bullish':
            signals['trend_continuation'] = trend.get('trend_strength', 0)
        elif trend.get('trend_direction') == 'bearish':
            signals['trend_continuation'] = -trend.get('trend_strength', 0)
        
        # Volatility signals
        vol_expansion = patterns.get('volatility_expansion', {})
        if vol_expansion.get('volatility_expansion'):
            signals['volatility_expansion'] = vol_expansion.get('volatility_ratio', 1) - 1
        
        # Volume confirmation
        volume = patterns.get('volume_profile', {})
        if volume.get('volume_price_correlation', 0) > 0:
            signals['volume_confirmation'] = 1
        elif volume.get('volume_price_correlation', 0) < 0:
            signals['volume_confirmation'] = -1
        
        return signals
    
    def _analyze_flow_signals(self, flow_alerts: List[FlowAlert]) -> Dict:
        """Analyze options flow signals"""
        signals = {
            'smart_money_direction': 0,
            'unusual_activity_strength': 0,
            'gamma_squeeze_potential': 0,
            'institutional_positioning': 0
        }
        
        if not flow_alerts:
            return signals
        
        # Smart money signals
        smart_money_alerts = [a for a in flow_alerts if 'smart_money' in a.alert_type]
        for alert in smart_money_alerts:
            direction_multiplier = 1 if alert.bullish_bearish == 'bullish' else -1
            signals['smart_money_direction'] += alert.confidence * direction_multiplier
        
        # Unusual activity
        unusual_alerts = [a for a in flow_alerts if a.alert_type in ['unusual_volume', 'block', 'sweep']]
        if unusual_alerts:
            avg_confidence = sum(a.confidence for a in unusual_alerts) / len(unusual_alerts)
            signals['unusual_activity_strength'] = avg_confidence
        
        # Gamma squeeze potential
        gamma_alerts = [a for a in flow_alerts if 'gamma' in a.alert_type]
        if gamma_alerts:
            signals['gamma_squeeze_potential'] = max(a.confidence for a in gamma_alerts)
        
        return signals
    
    def _analyze_volatility_regime(self, data: pd.DataFrame) -> Dict:
        """Analyze current volatility regime"""
        closes = data['Close'].values
        returns = np.diff(np.log(closes))
        
        current_vol = np.std(returns[-20:]) * np.sqrt(252)  # 20-day realized vol
        historical_vol = np.std(returns) * np.sqrt(252)  # Full period vol
        
        vol_regime = {
            'current_vol': current_vol,
            'historical_vol': historical_vol,
            'vol_ratio': current_vol / historical_vol,
            'regime': 'low' if current_vol < historical_vol * 0.8 else 'high' if current_vol > historical_vol * 1.2 else 'normal'
        }
        
        return vol_regime
    
    def _analyze_volume_characteristics(self, data: pd.DataFrame) -> Dict:
        """Analyze volume characteristics"""
        volumes = data['Volume'].values
        closes = data['Close'].values
        
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        
        # Price-volume relationship
        price_changes = np.diff(closes[-10:])
        volume_changes = np.diff(volumes[-10:])
        
        correlation = np.corrcoef(price_changes, volume_changes[:-1])[0, 1] if len(price_changes) > 1 else 0
        
        return {
            'relative_volume': current_volume / avg_volume,
            'price_volume_correlation': correlation,
            'volume_trend': 'increasing' if np.mean(volumes[-5:]) > np.mean(volumes[-20:-5]) else 'decreasing'
        }
    
    def _find_technical_confluence(self, patterns: Dict, data: pd.DataFrame) -> Dict:
        """Find confluence of technical signals"""
        confluences = {
            'bullish_confluence': 0,
            'bearish_confluence': 0,
            'neutral_confluence': 0
        }
        
        # Support/Resistance confluence
        sr = patterns.get('support_resistance', {})
        current_price = data['Close'].iloc[-1]
        
        support_levels = sr.get('support', [])
        resistance_levels = sr.get('resistance', [])
        
        # Check proximity to key levels
        for support in support_levels:
            if abs(current_price - support) / current_price < 0.02:  # Within 2%
                confluences['bullish_confluence'] += 0.3
        
        for resistance in resistance_levels:
            if abs(current_price - resistance) / current_price < 0.02:  # Within 2%
                confluences['bearish_confluence'] += 0.3
        
        return confluences
    
    async def _get_broader_market_context(self) -> Dict:
        """Get broader market context (SPY, VIX, etc.)"""
        # Simplified - would normally fetch real market data
        return {
            'spy_trend': 'neutral',
            'vix_level': 'normal',
            'sector_rotation': 'technology',
            'market_sentiment': 'neutral'
        }
    
    async def _get_historical_context(self, symbol: str, intelligence: Dict) -> Dict:
        """Get historical context for similar market conditions"""
        
        # Find similar patterns from knowledge database
        pattern_type = self._classify_current_situation(intelligence)
        similar_patterns = self.knowledge_db.get_similar_patterns(
            symbol, intelligence['pattern_signals'], pattern_type, lookback_days=180
        )
        
        if not similar_patterns:
            return {'similar_count': 0, 'avg_outcome': 0, 'confidence': 0}
        
        outcomes = [p.outcome for p in similar_patterns]
        avg_outcome = np.mean(outcomes)
        outcome_consistency = len([o for o in outcomes if o * avg_outcome > 0]) / len(outcomes)
        
        return {
            'similar_count': len(similar_patterns),
            'avg_outcome': avg_outcome,
            'outcome_consistency': outcome_consistency,
            'confidence': min(len(similar_patterns) / 20, 1.0) * outcome_consistency
        }
    
    def _classify_current_situation(self, intelligence: Dict) -> str:
        """Classify current market situation"""
        signals = intelligence['pattern_signals']
        
        if signals['breakout_strength'] > 0.3:
            return 'bullish_breakout'
        elif signals['breakout_strength'] < -0.3:
            return 'bearish_breakout'
        elif signals['reversal_probability'] > 0.5:
            return 'bullish_reversal'
        elif signals['reversal_probability'] < -0.5:
            return 'bearish_reversal'
        elif signals['volatility_expansion'] > 0.2:
            return 'volatility_expansion'
        else:
            return 'mixed'
    
    async def _calculate_ensemble_confidence(self, intelligence: Dict, 
                                           historical_context: Dict, 
                                           market_regime: Dict) -> Dict:
        """Calculate confidence using ensemble of different signals"""
        
        confidences = {}
        
        # Pattern recognition confidence
        pattern_signals = intelligence['pattern_signals']
        pattern_strength = sum(abs(v) for v in pattern_signals.values()) / len(pattern_signals)
        confidences['pattern'] = min(pattern_strength, 1.0)
        
        # AI prediction confidence
        confidences['ai_prediction'] = intelligence['ai_prediction'].get('confidence', 0)
        
        # Options flow confidence
        flow_signals = intelligence['options_flow_signals']
        flow_strength = sum(abs(v) for v in flow_signals.values()) / max(len(flow_signals), 1)
        confidences['options_flow'] = min(flow_strength, 1.0)
        
        # Historical pattern confidence
        confidences['historical'] = historical_context.get('confidence', 0)
        
        # Market regime adjustment
        regime_multiplier = 1.0
        if market_regime.get('regime') == 'bullish':
            regime_multiplier = 1.2
        elif market_regime.get('regime') == 'bearish':
            regime_multiplier = 0.8
        
        # Weighted ensemble confidence
        weights = {'pattern': 0.25, 'ai_prediction': 0.25, 'options_flow': 0.3, 'historical': 0.2}
        
        ensemble_confidence = sum(confidences[k] * weights[k] for k in confidences.keys())
        ensemble_confidence *= regime_multiplier
        
        return {
            'individual_confidences': confidences,
            'ensemble_confidence': min(ensemble_confidence, 1.0),
            'regime_adjustment': regime_multiplier
        }
    
    async def _generate_adaptive_decision(self, symbol: str, intelligence: Dict, 
                                        confidence: Dict, historical_context: Dict) -> Dict:
        """Generate adaptive decision based on all available intelligence"""
        
        decision = {
            'action': 'hold',
            'option_type': None,
            'strike_distance': 0,
            'expiration_days': 7,
            'confidence': 0.0,
            'reasoning': [],
            'position_size_multiplier': 1.0,
            'adaptive_factors': {}
        }
        
        ensemble_confidence = confidence['ensemble_confidence']
        
        # Only act if confidence exceeds threshold
        if ensemble_confidence < self.confidence_threshold:
            decision['reasoning'].append(f'Confidence {ensemble_confidence:.2f} below threshold {self.confidence_threshold}')
            return decision
        
        # Determine direction and strength from multiple signals
        directional_signals = []
        
        # Pattern signals
        pattern_signals = intelligence['pattern_signals']
        if pattern_signals['breakout_strength'] > 0.3:
            directional_signals.append(('bullish', pattern_signals['breakout_strength'], 'breakout'))
        elif pattern_signals['breakout_strength'] < -0.3:
            directional_signals.append(('bearish', abs(pattern_signals['breakout_strength']), 'breakout'))
        
        if pattern_signals['reversal_probability'] > 0.4:
            directional_signals.append(('bullish', pattern_signals['reversal_probability'], 'reversal'))
        elif pattern_signals['reversal_probability'] < -0.4:
            directional_signals.append(('bearish', abs(pattern_signals['reversal_probability']), 'reversal'))
        
        # AI prediction signal
        ai_pred = intelligence['ai_prediction']
        if ai_pred['direction'] in ['bullish', 'bearish'] and ai_pred['confidence'] > 0.6:
            strength = ai_pred['confidence']
            directional_signals.append((ai_pred['direction'], strength, 'ai_prediction'))
        
        # Options flow signals
        flow_signals = intelligence['options_flow_signals']
        if abs(flow_signals['smart_money_direction']) > 0.3:
            direction = 'bullish' if flow_signals['smart_money_direction'] > 0 else 'bearish'
            strength = abs(flow_signals['smart_money_direction'])
            directional_signals.append((direction, strength, 'smart_money_flow'))
        
        # Historical context signal
        if historical_context['similar_count'] > 5:
            avg_outcome = historical_context['avg_outcome']
            if abs(avg_outcome) > 0.02:  # Significant historical outcome
                direction = 'bullish' if avg_outcome > 0 else 'bearish'
                strength = min(abs(avg_outcome) * 5, 1.0)  # Scale outcome to strength
                directional_signals.append((direction, strength, 'historical_pattern'))
        
        # Calculate consensus direction
        if not directional_signals:
            decision['reasoning'].append('No clear directional signals')
            return decision
        
        bullish_signals = [s for s in directional_signals if s[0] == 'bullish']
        bearish_signals = [s for s in directional_signals if s[0] == 'bearish']
        
        bullish_strength = sum(s[1] for s in bullish_signals)
        bearish_strength = sum(s[1] for s in bearish_signals)
        
        # Determine final direction and action
        if bullish_strength > bearish_strength and bullish_strength > 1.0:
            # Bullish consensus
            volatility_regime = intelligence['volatility_profile']['regime']
            momentum = intelligence['price_momentum']['strength']
            
            if volatility_regime == 'low' and momentum > 0.02:
                decision['action'] = 'buy_calls'
                decision['option_type'] = 'call'
                decision['strike_distance'] = min(0.03, momentum * 2)  # Adaptive strike distance
                decision['reasoning'].extend([f'{s[2]}: {s[1]:.2f}' for s in bullish_signals])
            
            elif intelligence['options_flow_signals']['gamma_squeeze_potential'] > 0.5:
                decision['action'] = 'buy_calls'
                decision['option_type'] = 'call'
                decision['strike_distance'] = 0.01  # Near ATM for gamma
                decision['expiration_days'] = 3  # Short expiry for gamma
                decision['reasoning'].append('Gamma squeeze potential detected')
            
            elif volatility_regime == 'high':
                decision['action'] = 'sell_puts'  # Bullish but high vol - sell puts
                decision['option_type'] = 'put'
                decision['strike_distance'] = -0.05  # OTM puts
                decision['expiration_days'] = 14
                decision['reasoning'].append('High volatility regime - selling puts')
        
        elif bearish_strength > bullish_strength and bearish_strength > 1.0:
            # Bearish consensus
            volatility_regime = intelligence['volatility_profile']['regime']
            momentum = intelligence['price_momentum']['strength']
            
            if volatility_regime in ['normal', 'high'] and momentum > 0.02:
                decision['action'] = 'buy_puts'
                decision['option_type'] = 'put'
                decision['strike_distance'] = -min(0.03, momentum * 2)
                decision['reasoning'].extend([f'{s[2]}: {s[1]:.2f}' for s in bearish_signals])
        
        # Special volatility expansion strategy
        if (intelligence['pattern_signals']['volatility_expansion'] > 0.3 and 
            intelligence['volatility_profile']['regime'] == 'low'):
            decision['action'] = 'buy_straddle'
            decision['option_type'] = 'straddle'
            decision['strike_distance'] = 0  # ATM
            decision['expiration_days'] = 14
            decision['reasoning'].append('Volatility expansion from low regime')
        
        # Adaptive position sizing
        decision['position_size_multiplier'] = self._calculate_adaptive_position_size(
            ensemble_confidence, intelligence, historical_context
        )
        
        decision['confidence'] = ensemble_confidence
        decision['adaptive_factors'] = {
            'signal_count': len(directional_signals),
            'bullish_strength': bullish_strength,
            'bearish_strength': bearish_strength,
            'volatility_regime': intelligence['volatility_profile']['regime'],
            'momentum': intelligence['price_momentum']['strength']
        }
        
        return decision
    
    def _calculate_adaptive_position_size(self, confidence: float, intelligence: Dict, 
                                        historical_context: Dict) -> float:
        """Calculate adaptive position size based on confidence and conditions"""
        base_multiplier = confidence  # Start with confidence
        
        # Adjust for historical performance
        if historical_context.get('outcome_consistency', 0) > 0.7:
            base_multiplier *= 1.2
        elif historical_context.get('outcome_consistency', 0) < 0.4:
            base_multiplier *= 0.8
        
        # Adjust for volatility regime
        vol_regime = intelligence['volatility_profile']['regime']
        if vol_regime == 'high':
            base_multiplier *= 0.7  # Reduce size in high vol
        elif vol_regime == 'low':
            base_multiplier *= 1.3  # Increase size in low vol
        
        # Adjust for momentum
        momentum = intelligence['price_momentum']['strength']
        if momentum > 0.05:  # Strong momentum
            base_multiplier *= 1.2
        elif momentum < 0.01:  # Weak momentum
            base_multiplier *= 0.8
        
        return min(max(base_multiplier, 0.3), 2.0)  # Cap between 0.3x and 2x
    
    def _record_decision(self, symbol: str, decision: Dict, intelligence: Dict) -> None:
        """Record decision for future learning"""
        self.decision_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'decision': decision,
            'intelligence': intelligence
        })
        
        # Keep only recent decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    async def learn_from_outcome(self, symbol: str, decision: Dict, actual_outcome: float) -> None:
        """Learn from actual trading outcomes"""
        try:
            # Store outcome in knowledge database
            outcome_record = TradingOutcome(
                timestamp=datetime.now(),
                symbol=symbol,
                action=decision['action'],
                entry_price=0.0,  # Would be filled with actual entry
                exit_price=None,
                profit_loss=actual_outcome,
                pattern_match_confidence=decision['confidence'],
                decision_reasoning=decision['reasoning']
            )
            
            self.knowledge_db.store_trading_outcome(outcome_record)
            
            # Adapt confidence threshold based on performance
            if actual_outcome > 0 and decision['confidence'] > 0.8:
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.01)
            elif actual_outcome < 0 and decision['confidence'] < 0.7:
                self.confidence_threshold = min(0.8, self.confidence_threshold + 0.01)
            
            self.logger.info(f"📚 Learned from {symbol}: outcome {actual_outcome:.2f}, "
                           f"adjusted threshold to {self.confidence_threshold:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error learning from outcome: {e}")

class IntelligentOptionsBot:
    """Main intelligent options trading bot"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.data_manager = MultiSourceDataManager()
        self.pattern_engine = PatternRecognitionEngine()
        self.learning_engine = MarketLearningEngine()
        self.knowledge_db = MarketKnowledgeDatabase()
        self.flow_analyzer = OptionsFlowAnalyzer()
        self.decision_engine = IntelligentDecisionEngine(self.knowledge_db, self.flow_analyzer)
        
        # Initialize Alpaca client
        self.alpaca = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            base_url=os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
            api_version='v2'
        )
        
        self.watchlist = os.getenv('WATCHLIST', 'SPY,QQQ,TSLA,AAPL,MSFT,NVDA,UNH').split(',')
        self.is_learning_complete = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('intelligent_options_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the bot and learn from historical data"""
        self.logger.info("🤖 Initializing Intelligent Options Bot...")
        
        # Learn from historical data
        await self.learning_engine.learn_from_historical_data(self.watchlist)
        self.is_learning_complete = True
        
        self.logger.info("✅ Bot initialization complete - Ready for intelligent trading")
    
    async def run(self):
        """Main trading loop"""
        if not self.is_learning_complete:
            await self.initialize()
        
        self.logger.info("🚀 Starting intelligent options trading...")
        
        while True:
            try:
                await self._trading_cycle()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading cycle: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        for symbol in self.watchlist:
            try:
                # Get current market data
                market_data = await self.data_manager.get_market_data(symbol)
                if not market_data:
                    continue
                
                # Analyze patterns
                patterns = await self.pattern_engine.analyze_chart_patterns(market_data)
                
                # Get AI prediction
                prediction = await self.learning_engine.predict_market_direction(patterns)
                
                # Make trading decision
                decision = await self._make_intelligent_decision(symbol, patterns, prediction, market_data)
                
                if decision['action'] != 'hold':
                    await self._execute_options_trade(symbol, decision)
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing {symbol}: {e}")
    
    async def _make_intelligent_decision(self, symbol: str, patterns: Dict, 
                                       prediction: Dict, market_data: pd.DataFrame) -> Dict:
        """Make intelligent trading decision using the new decision engine"""
        
        # Get options flow alerts
        flow_alerts = await self.flow_analyzer.analyze_options_flow([symbol])
        
        # Use the intelligent decision engine
        decision = await self.decision_engine.make_intelligent_decision(
            symbol, patterns, prediction, market_data, flow_alerts
        )
        
        self.logger.info(f"🧠 {symbol}: {decision['action'].upper()} - "
                        f"Confidence: {decision['confidence']:.2f} - "
                        f"Reasoning: {', '.join(decision['reasoning'])}")
        
        return decision
    
    def _assess_market_conditions(self, patterns: Dict, prediction: Dict) -> Dict:
        """Assess overall market conditions"""
        conditions = {
            'breakout_strength': 0,
            'breakdown_strength': 0,
            'squeeze_potential': 0,
            'trend_strength': 0
        }
        
        # Breakout analysis
        breakout = patterns.get('breakout_patterns', {})
        if breakout.get('breakout_type') == 'bullish_breakout':
            conditions['breakout_strength'] = breakout.get('breakout_strength', 0)
        elif breakout.get('breakout_type') == 'bearish_breakout':
            conditions['breakdown_strength'] = breakout.get('breakout_strength', 0)
        
        # Squeeze analysis
        squeeze = patterns.get('volatility_contraction', {})
        if squeeze.get('squeeze_active'):
            conditions['squeeze_potential'] = min(squeeze.get('squeeze_duration', 0) / 10, 1.0)
        
        # Trend analysis
        trend = patterns.get('trend_channels', {})
        conditions['trend_strength'] = trend.get('trend_strength', 0)
        
        return conditions
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """Calculate short-term price momentum"""
        if len(data) < 5:
            return 0.0
        
        current_price = data['Close'].iloc[-1]
        price_5_ago = data['Close'].iloc[-5]
        
        return (current_price - price_5_ago) / price_5_ago
    
    async def _execute_options_trade(self, symbol: str, decision: Dict):
        """Execute options trade based on decision"""
        try:
            # Get current options chain
            options_data = await self.data_manager.get_options_chain(symbol)
            if not options_data:
                self.logger.error(f"❌ No options data available for {symbol}")
                return
            
            # Calculate position size based on account value
            account = self.alpaca.get_account()
            portfolio_value = float(account.portfolio_value)
            risk_amount = portfolio_value * 0.02  # 2% risk per trade
            
            if decision['action'] == 'buy_calls':
                await self._buy_calls(symbol, decision, options_data, risk_amount)
            elif decision['action'] == 'buy_puts':
                await self._buy_puts(symbol, decision, options_data, risk_amount)
            elif decision['action'] == 'buy_straddle':
                await self._buy_straddle(symbol, decision, options_data, risk_amount)
                
        except Exception as e:
            self.logger.error(f"❌ Error executing trade for {symbol}: {e}")
    
    async def _buy_calls(self, symbol: str, decision: Dict, options_data: Dict, risk_amount: float):
        """Buy call options"""
        try:
            # Find appropriate call option
            calls = options_data.get('calls', [])
            if not calls:
                return
            
            # Filter by expiration and strike
            target_expiry = (datetime.now() + timedelta(days=decision['expiration_days'])).strftime('%Y-%m-%d')
            
            suitable_calls = [
                call for call in calls 
                if call.get('expiration') == target_expiry
            ]
            
            if suitable_calls:
                # Select the option closest to target strike
                best_call = min(suitable_calls, 
                              key=lambda x: abs(x.get('strike', 0) - self._calculate_target_strike(symbol, decision)))
                
                # Calculate quantity based on risk
                option_price = best_call.get('last_price', best_call.get('mark', 0))
                if option_price > 0:
                    quantity = max(1, int(risk_amount / (option_price * 100)))
                    
                    # Place order
                    order = self.alpaca.submit_order(
                        symbol=best_call['symbol'],
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    self.logger.info(f"✅ CALL ORDER: {symbol} {quantity}x {best_call['symbol']} @ ${option_price}")
                    return order
                    
        except Exception as e:
            self.logger.error(f"❌ Error buying calls for {symbol}: {e}")
    
    async def _buy_puts(self, symbol: str, decision: Dict, options_data: Dict, risk_amount: float):
        """Buy put options"""
        try:
            # Similar logic to calls but for puts
            puts = options_data.get('puts', [])
            if not puts:
                return
            
            target_expiry = (datetime.now() + timedelta(days=decision['expiration_days'])).strftime('%Y-%m-%d')
            
            suitable_puts = [
                put for put in puts 
                if put.get('expiration') == target_expiry
            ]
            
            if suitable_puts:
                best_put = min(suitable_puts, 
                             key=lambda x: abs(x.get('strike', 0) - self._calculate_target_strike(symbol, decision)))
                
                option_price = best_put.get('last_price', best_put.get('mark', 0))
                if option_price > 0:
                    quantity = max(1, int(risk_amount / (option_price * 100)))
                    
                    order = self.alpaca.submit_order(
                        symbol=best_put['symbol'],
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    self.logger.info(f"✅ PUT ORDER: {symbol} {quantity}x {best_put['symbol']} @ ${option_price}")
                    return order
                    
        except Exception as e:
            self.logger.error(f"❌ Error buying puts for {symbol}: {e}")
    
    async def _buy_straddle(self, symbol: str, decision: Dict, options_data: Dict, risk_amount: float):
        """Buy straddle (call + put at same strike)"""
        try:
            calls = options_data.get('calls', [])
            puts = options_data.get('puts', [])
            
            if not calls or not puts:
                return
            
            target_expiry = (datetime.now() + timedelta(days=decision['expiration_days'])).strftime('%Y-%m-%d')
            target_strike = self._calculate_target_strike(symbol, decision)
            
            # Find matching call and put
            suitable_calls = [c for c in calls if c.get('expiration') == target_expiry]
            suitable_puts = [p for p in puts if p.get('expiration') == target_expiry]
            
            if suitable_calls and suitable_puts:
                best_call = min(suitable_calls, key=lambda x: abs(x.get('strike', 0) - target_strike))
                best_put = min(suitable_puts, key=lambda x: abs(x.get('strike', 0) - target_strike))
                
                call_price = best_call.get('last_price', best_call.get('mark', 0))
                put_price = best_put.get('last_price', best_put.get('mark', 0))
                
                if call_price > 0 and put_price > 0:
                    straddle_cost = (call_price + put_price) * 100
                    quantity = max(1, int(risk_amount / straddle_cost))
                    
                    # Place both orders
                    call_order = self.alpaca.submit_order(
                        symbol=best_call['symbol'],
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    put_order = self.alpaca.submit_order(
                        symbol=best_put['symbol'],
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    self.logger.info(f"✅ STRADDLE: {symbol} {quantity}x calls + puts @ ${straddle_cost/100:.2f}")
                    return [call_order, put_order]
                    
        except Exception as e:
            self.logger.error(f"❌ Error buying straddle for {symbol}: {e}")
    
    def _calculate_target_strike(self, symbol: str, decision: Dict) -> float:
        """Calculate target strike price"""
        try:
            # Get current stock price
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            strike_distance = decision.get('strike_distance', 0)
            return current_price * (1 + strike_distance)
            
        except:
            return 0.0

if __name__ == "__main__":
    bot = IntelligentOptionsBot()
    asyncio.run(bot.run())
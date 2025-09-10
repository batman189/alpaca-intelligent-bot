"""
Dynamic Watchlist Manager - Smart Symbol Discovery
Automatically expand beyond static watchlists with trending stocks and volume leaders
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WatchlistCategory(Enum):
    BASE = "base"                    # Core static watchlist
    TRENDING = "trending"            # Trending stocks
    VOLUME_LEADERS = "volume_leaders" # High volume stocks
    EARNINGS = "earnings"            # Earnings plays
    SECTOR_LEADERS = "sector_leaders" # Sector momentum
    BREAKOUTS = "breakouts"          # Technical breakouts
    NEWS_DRIVEN = "news_driven"      # News-driven moves
    MOMENTUM = "momentum"            # Momentum stocks

@dataclass
class WatchlistEntry:
    symbol: str
    category: WatchlistCategory
    score: float
    added_at: datetime
    metadata: Dict[str, Any]
    last_updated: datetime
    expiry: Optional[datetime] = None

class DynamicWatchlistManager:
    def __init__(self, base_symbols: List[str] = None):
        """Initialize dynamic watchlist management system"""
        
        # ADD THIS LINE TO FIX THE MISSING NAME ATTRIBUTE
        self.name = "DynamicWatchlistManager"
    
        # Base static watchlist
        self.base_symbols = set(base_symbols or [
            "SPY", "QQQ", "IWM", "AAPL", "GOOGL", "MSFT", "TSLA", 
            "AMZN", "NVDA", "META", "NFLX", "AMD", "UNH", "JNJ", "PFE"
        ])
        
        # Dynamic watchlists by category
        self.dynamic_watchlists: Dict[WatchlistCategory, Dict[str, WatchlistEntry]] = {
            category: {} for category in WatchlistCategory
        }
        
        # Watchlist management parameters
        self.max_symbols_per_category = {
            WatchlistCategory.BASE: 50,
            WatchlistCategory.TRENDING: 20,
            WatchlistCategory.VOLUME_LEADERS: 15,
            WatchlistCategory.EARNINGS: 10,
            WatchlistCategory.SECTOR_LEADERS: 25,
            WatchlistCategory.BREAKOUTS: 15,
            WatchlistCategory.NEWS_DRIVEN: 10,
            WatchlistCategory.MOMENTUM: 20
        }
        
        # Scoring weights for different factors
        self.scoring_weights = {
            "volume_score": 0.25,
            "price_change": 0.20,
            "trend_strength": 0.20,
            "volatility": 0.15,
            "technical_score": 0.10,
            "news_sentiment": 0.10
        }
        
        # Update frequencies (in minutes)
        self.update_frequencies = {
            WatchlistCategory.BASE: 1440,        # Daily
            WatchlistCategory.TRENDING: 60,      # Hourly
            WatchlistCategory.VOLUME_LEADERS: 30, # Every 30 minutes
            WatchlistCategory.EARNINGS: 1440,    # Daily
            WatchlistCategory.SECTOR_LEADERS: 240, # Every 4 hours
            WatchlistCategory.BREAKOUTS: 15,     # Every 15 minutes
            WatchlistCategory.NEWS_DRIVEN: 30,   # Every 30 minutes
            WatchlistCategory.MOMENTUM: 60       # Hourly
        }
        
        # Expiry times for dynamic entries
        self.entry_expiry = {
            WatchlistCategory.TRENDING: timedelta(days=3),
            WatchlistCategory.VOLUME_LEADERS: timedelta(hours=6),
            WatchlistCategory.BREAKOUTS: timedelta(hours=12),
            WatchlistCategory.NEWS_DRIVEN: timedelta(hours=24),
            WatchlistCategory.MOMENTUM: timedelta(days=2)
        }
        
        # Market data cache
        self.market_data_cache = {}
        self.last_updates = {category: datetime.min for category in WatchlistCategory}
        
        # Statistics tracking
        self.discovery_stats = defaultdict(int)
        
        # Initialize base watchlist
        for symbol in self.base_symbols:
            self.dynamic_watchlists[WatchlistCategory.BASE][symbol] = WatchlistEntry(
                symbol=symbol,
                category=WatchlistCategory.BASE,
                score=100.0,
                added_at=datetime.now(),
                metadata={"source": "base_watchlist"},
                last_updated=datetime.now()
            )
        
        logger.info(f"📈 Dynamic Watchlist Manager initialized with {len(self.base_symbols)} base symbols")

    async def update_watchlists(self, market_data_manager) -> Dict[WatchlistCategory, int]:
        """Update all dynamic watchlists"""
        
        update_counts = {}
        current_time = datetime.now()
        
        for category in WatchlistCategory:
            if category == WatchlistCategory.BASE:
                continue  # Base watchlist doesn't need updates
                
            # Check if update is needed
            time_since_update = current_time - self.last_updates[category]
            update_interval = timedelta(minutes=self.update_frequencies[category])
            
            if time_since_update >= update_interval:
                try:
                    count = await self._update_category_watchlist(category, market_data_manager)
                    update_counts[category] = count
                    self.last_updates[category] = current_time
                    logger.info(f"📊 Updated {category.value} watchlist: {count} symbols")
                except Exception as e:
                    logger.error(f"❌ Failed to update {category.value} watchlist: {e}")
                    update_counts[category] = 0
        
        # Clean expired entries
        self._cleanup_expired_entries()
        
        return update_counts

    async def _update_category_watchlist(self, category: WatchlistCategory, 
                                       data_manager) -> int:
        """Update a specific category watchlist"""
        
        if category == WatchlistCategory.TRENDING:
            return await self._update_trending_stocks(data_manager)
        elif category == WatchlistCategory.VOLUME_LEADERS:
            return await self._update_volume_leaders(data_manager)
        elif category == WatchlistCategory.EARNINGS:
            return await self._update_earnings_plays(data_manager)
        elif category == WatchlistCategory.SECTOR_LEADERS:
            return await self._update_sector_leaders(data_manager)
        elif category == WatchlistCategory.BREAKOUTS:
            return await self._update_breakout_stocks(data_manager)
        elif category == WatchlistCategory.NEWS_DRIVEN:
            return await self._update_news_driven_stocks(data_manager)
        elif category == WatchlistCategory.MOMENTUM:
            return await self._update_momentum_stocks(data_manager)
        
        return 0

    async def _update_trending_stocks(self, data_manager) -> int:
        """Update trending stocks watchlist"""
        
        # Sample trending candidates (would normally come from financial APIs)
        trending_candidates = [
            "ROKU", "PLTR", "NIO", "LCID", "RIVN", "SOFI", "HOOD", "COIN",
            "SHOP", "SQ", "PYPL", "UBER", "LYFT", "SNAP", "TWTR", "PINS"
        ]
        
        added_count = 0
        
        for symbol in trending_candidates[:10]:  # Limit to top 10
            try:
                # Get market data
                data = await data_manager.get_market_data(symbol, "1Day", 30)
                if data is None or len(data) < 10:
                    continue
                
                # Calculate trending score
                score = self._calculate_trending_score(symbol, data)
                
                if score > 60:  # Threshold for inclusion
                    self._add_or_update_symbol(
                        symbol=symbol,
                        category=WatchlistCategory.TRENDING,
                        score=score,
                        metadata={
                            "trending_score": score,
                            "source": "trending_discovery",
                            "price_change_30d": (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
                        }
                    )
                    added_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process trending candidate {symbol}: {e}")
        
        return added_count

    async def _update_volume_leaders(self, data_manager) -> int:
        """Update volume leaders watchlist"""
        
        # Sample volume leaders - would typically come from market screener API
        volume_candidates = [
            "AAPL", "TSLA", "SPY", "QQQ", "AMD", "NVDA", "F", "BAC", 
            "MSFT", "AMZN", "META", "GOOGL"
        ]
        
        added_count = 0
        
        for symbol in volume_candidates:
            try:
                data = await data_manager.get_market_data(symbol, "1Hour", 50)
                if data is None or len(data) < 20:
                    continue
                
                # Calculate volume score
                score = self._calculate_volume_score(symbol, data)
                
                if score > 70:  # High volume threshold
                    self._add_or_update_symbol(
                        symbol=symbol,
                        category=WatchlistCategory.VOLUME_LEADERS,
                        score=score,
                        metadata={
                            "volume_score": score,
                            "source": "volume_discovery",
                            "avg_volume_ratio": data['volume'].tail(5).mean() / data['volume'].mean()
                        }
                    )
                    added_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process volume candidate {symbol}: {e}")
        
        return added_count

    async def _update_earnings_plays(self, data_manager) -> int:
        """Update earnings plays watchlist"""
        
        # Sample earnings candidates (would connect to earnings calendar API)
        earnings_candidates = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NFLX", "NVDA"
        ]
        
        added_count = 0
        
        for symbol in earnings_candidates[:5]:  # Limit to 5
            try:
                data = await data_manager.get_market_data(symbol, "1Day", 20)
                if data is None:
                    continue
                
                # Calculate earnings score (volatility + volume)
                score = self._calculate_earnings_score(symbol, data)
                
                if score > 65:
                    self._add_or_update_symbol(
                        symbol=symbol,
                        category=WatchlistCategory.EARNINGS,
                        score=score,
                        metadata={
                            "earnings_score": score,
                            "source": "earnings_discovery",
                            "estimated_earnings_date": "TBD"  # Would be real date from API
                        }
                    )
                    added_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process earnings candidate {symbol}: {e}")
        
        return added_count

    async def _update_sector_leaders(self, data_manager) -> int:
        """Update sector leaders watchlist"""
        
        # Sector ETFs and leaders
        sector_candidates = {
            "XLK": ["AAPL", "MSFT", "NVDA", "GOOGL"],  # Technology
            "XLF": ["JPM", "BAC", "WFC", "GS"],        # Financials  
            "XLE": ["XOM", "CVX", "COP", "EOG"],       # Energy
            "XLV": ["JNJ", "PFE", "UNH", "ABBV"],      # Healthcare
            "XLI": ["CAT", "BA", "GE", "MMM"]          # Industrials
        }
        
        added_count = 0
        
        for etf, stocks in sector_candidates.items():
            try:
                # Check sector ETF performance
                etf_data = await data_manager.get_market_data(etf, "1Day", 10)
                if etf_data is None:
                    continue
                
                sector_performance = (etf_data['close'].iloc[-1] - etf_data['close'].iloc[-5]) / etf_data['close'].iloc[-5]
                
                # If sector is performing well, add its leaders
                if sector_performance > 0.02:  # 2% sector gain in 5 days
                    for symbol in stocks[:2]:  # Top 2 from each strong sector
                        try:
                            stock_data = await data_manager.get_market_data(symbol, "1Day", 10)
                            if stock_data is None:
                                continue
                                
                            score = self._calculate_sector_leader_score(symbol, stock_data, sector_performance)
                            
                            if score > 70:
                                self._add_or_update_symbol(
                                    symbol=symbol,
                                    category=WatchlistCategory.SECTOR_LEADERS,
                                    score=score,
                                    metadata={
                                        "sector_leader_score": score,
                                        "sector_etf": etf,
                                        "sector_performance": sector_performance,
                                        "source": "sector_discovery"
                                    }
                                )
                                added_count += 1
                                
                        except Exception as e:
                            logger.warning(f"Failed to process sector leader {symbol}: {e}")
                            
            except Exception as e:
                logger.warning(f"Failed to process sector {etf}: {e}")
        
        return added_count

    async def _update_breakout_stocks(self, data_manager) -> int:
        """Update breakout stocks watchlist"""
        
        # Sample breakout candidates
        breakout_candidates = [
            "ROKU", "SHOP", "SQ", "PYPL", "UBER", "LYFT", "SNAP", "PINS",
            "PLTR", "NIO", "LCID", "RIVN", "SOFI", "HOOD"
        ]
        
        added_count = 0
        
        for symbol in breakout_candidates:
            try:
                data = await data_manager.get_market_data(symbol, "1Hour", 100)
                if data is None or len(data) < 50:
                    continue
                
                # Calculate breakout score
                score = self._calculate_breakout_score(symbol, data)
                
                if score > 75:  # High breakout probability
                    self._add_or_update_symbol(
                        symbol=symbol,
                        category=WatchlistCategory.BREAKOUTS,
                        score=score,
                        metadata={
                            "breakout_score": score,
                            "source": "breakout_discovery",
                            "breakout_type": "resistance" if data['close'].iloc[-1] > data['high'].rolling(20).max().iloc[-2] else "volume"
                        }
                    )
                    added_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process breakout candidate {symbol}: {e}")
        
        return added_count

    async def _update_news_driven_stocks(self, data_manager) -> int:
        """Update news-driven stocks watchlist (placeholder)"""
        # This would integrate with news APIs to find stocks with significant news
        return 0

    async def _update_momentum_stocks(self, data_manager) -> int:
        """Update momentum stocks watchlist"""
        
        momentum_candidates = [
            "TSLA", "NVDA", "AMD", "COIN", "ROKU", "PLTR", "NIO", "LCID"
        ]
        
        added_count = 0
        
        for symbol in momentum_candidates:
            try:
                data = await data_manager.get_market_data(symbol, "1Day", 20)
                if data is None:
                    continue
                
                score = self._calculate_momentum_score(symbol, data)
                
                if score > 70:
                    self._add_or_update_symbol(
                        symbol=symbol,
                        category=WatchlistCategory.MOMENTUM,
                        score=score,
                        metadata={
                            "momentum_score": score,
                            "source": "momentum_discovery",
                            "momentum_period": "20d"
                        }
                    )
                    added_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process momentum candidate {symbol}: {e}")
        
        return added_count

    def _calculate_trending_score(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate trending score for a symbol"""
        try:
            # Price momentum over different periods
            price_5d = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
            price_10d = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
            
            # Volume trend
            volume_avg_recent = data['volume'].tail(5).mean()
            volume_avg_historical = data['volume'].iloc[-20:-5].mean()
            volume_trend = (volume_avg_recent - volume_avg_historical) / volume_avg_historical if volume_avg_historical > 0 else 0
            
            # Combine factors
            trending_score = (
                abs(price_5d) * 30 +      # Recent price movement
                abs(price_10d) * 20 +     # Medium-term movement
                volume_trend * 25 +       # Volume increase
                25                        # Base score
            )
            
            return min(100, max(0, trending_score))
            
        except Exception as e:
            logger.warning(f"Trending score calculation failed for {symbol}: {e}")
            return 0

    def _calculate_volume_score(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate volume score for a symbol"""
        try:
            if 'volume' not in data.columns:
                return 0
            
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Recent volume trend
            recent_volume_avg = data['volume'].tail(5).mean()
            volume_trend = (recent_volume_avg - avg_volume) / avg_volume if avg_volume > 0 else 0
            
            volume_score = (
                min(volume_ratio * 20, 50) +     # Current volume ratio (capped)
                min(volume_trend * 30, 30) +     # Volume trend
                20                               # Base score
            )
            
            return min(100, max(0, volume_score))
            
        except Exception as e:
            logger.warning(f"Volume score calculation failed for {symbol}: {e}")
            return 0

    def _calculate_earnings_score(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate earnings play score"""
        try:
            # Volatility (higher is better for earnings plays)
            returns = data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Volume increase
            volume_ratio = data['volume'].tail(5).mean() / data['volume'].mean() if data['volume'].mean() > 0 else 1
            
            earnings_score = (
                min(volatility * 100, 40) +      # Volatility component
                min(volume_ratio * 30, 40) +     # Volume component
                20                               # Base score
            )
            
            return min(100, max(0, earnings_score))
            
        except Exception as e:
            logger.warning(f"Earnings score calculation failed for {symbol}: {e}")
            return 0

    def _calculate_sector_leader_score(self, symbol: str, data: pd.DataFrame, 
                                     sector_performance: float) -> float:
        """Calculate sector leader score"""
        try:
            # Individual stock performance
            stock_performance = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
            
            # Relative strength vs sector
            relative_strength = stock_performance - sector_performance
            
            sector_score = (
                stock_performance * 30 +         # Absolute performance
                relative_strength * 40 +         # Relative performance
                sector_performance * 20 +        # Sector momentum
                10                               # Base score
            )
            
            return min(100, max(0, sector_score * 100))
            
        except Exception as e:
            logger.warning(f"Sector leader score calculation failed for {symbol}: {e}")
            return 0

    def _calculate_breakout_score(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate breakout score"""
        try:
            current_price = data['close'].iloc[-1]
            
            # Resistance breakout
            resistance_20 = data['high'].rolling(20).max().iloc[-2]
            resistance_50 = data['high'].rolling(50).max().iloc[-2] if len(data) >= 50 else resistance_20
            
            resistance_break = 0
            if current_price > resistance_20:
                resistance_break += 30
            if current_price > resistance_50:
                resistance_break += 20
            
            # Volume confirmation
            volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1] if data['volume'].rolling(20).mean().iloc[-1] > 0 else 1
            volume_score = min(volume_ratio * 20, 30)
            
            # Price momentum
            momentum = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]
            momentum_score = min(abs(momentum) * 200, 20)
            
            breakout_score = resistance_break + volume_score + momentum_score
            
            return min(100, max(0, breakout_score))
            
        except Exception as e:
            logger.warning(f"Breakout score calculation failed for {symbol}: {e}")
            return 0

    def _calculate_momentum_score(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            # Multiple timeframe momentum
            momentum_5d = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
            momentum_10d = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
            momentum_20d = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20] if len(data) >= 20 else momentum_10d
            
            # Momentum consistency (same direction across timeframes)
            momentum_consistency = 0
            if momentum_5d > 0 and momentum_10d > 0 and momentum_20d > 0:
                momentum_consistency = 20
            elif momentum_5d < 0 and momentum_10d < 0 and momentum_20d < 0:
                momentum_consistency = 20
            
            momentum_score = (
                abs(momentum_5d) * 30 +
                abs(momentum_10d) * 25 +
                abs(momentum_20d) * 15 +
                momentum_consistency +
                10  # Base score
            )
            
            return min(100, max(0, momentum_score * 100))
            
        except Exception as e:
            logger.warning(f"Momentum score calculation failed for {symbol}: {e}")
            return 0

    def _add_or_update_symbol(self, symbol: str, category: WatchlistCategory, 
                             score: float, metadata: Dict[str, Any]):
        """Add or update a symbol in the watchlist"""
        
        # Check if we're at capacity
        current_count = len(self.dynamic_watchlists[category])
        max_count = self.max_symbols_per_category[category]
        
        if symbol in self.dynamic_watchlists[category]:
            # Update existing entry
            entry = self.dynamic_watchlists[category][symbol]
            entry.score = score
            entry.metadata.update(metadata)
            entry.last_updated = datetime.now()
        else:
            # Add new entry
            if current_count >= max_count:
                # Remove lowest scoring entry
                lowest_symbol = min(
                    self.dynamic_watchlists[category].keys(),
                    key=lambda s: self.dynamic_watchlists[category][s].score
                )
                if self.dynamic_watchlists[category][lowest_symbol].score < score:
                    del self.dynamic_watchlists[category][lowest_symbol]
                else:
                    return  # New symbol doesn't qualify
            
            # Create new entry
            expiry = None
            if category in self.entry_expiry:
                expiry = datetime.now() + self.entry_expiry[category]
            
            self.dynamic_watchlists[category][symbol] = WatchlistEntry(
                symbol=symbol,
                category=category,
                score=score,
                added_at=datetime.now(),
                metadata=metadata,
                last_updated=datetime.now(),
                expiry=expiry
            )
            
            self.discovery_stats[f"{category.value}_added"] += 1

    def _cleanup_expired_entries(self):
        """Remove expired watchlist entries"""
        current_time = datetime.now()
        expired_count = 0
        
        for category, entries in self.dynamic_watchlists.items():
            if category == WatchlistCategory.BASE:
                continue  # Base watchlist entries don't expire
                
            expired_symbols = []
            for symbol, entry in entries.items():
                if entry.expiry and current_time > entry.expiry:
                    expired_symbols.append(symbol)
            
            for symbol in expired_symbols:
                del entries[symbol]
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"🧹 Cleaned up {expired_count} expired watchlist entries")

    def get_all_symbols(self) -> Set[str]:
        """Get all symbols from all watchlists"""
        all_symbols = set()
        
        for category_watchlist in self.dynamic_watchlists.values():
            all_symbols.update(category_watchlist.keys())
        
        return all_symbols

    def get_symbols_by_category(self, category: WatchlistCategory) -> List[str]:
        """Get symbols from a specific category"""
        return list(self.dynamic_watchlists[category].keys())

    def get_top_symbols(self, count: int = 50) -> List[Tuple[str, float, WatchlistCategory]]:
        """Get top symbols across all categories by score"""
        all_entries = []
        
        for category, entries in self.dynamic_watchlists.items():
            for symbol, entry in entries.items():
                all_entries.append((symbol, entry.score, category))
        
        # Sort by score (highest first)
        all_entries.sort(key=lambda x: x[1], reverse=True)
        
        return all_entries[:count]

    def get_watchlist_statistics(self) -> Dict[str, Any]:
        """Get watchlist statistics"""
        total_symbols = len(self.get_all_symbols())
        category_counts = {}
        
        for category, entries in self.dynamic_watchlists.items():
            category_counts[category.value] = len(entries)
        
        return {
            "total_unique_symbols": total_symbols,
            "category_breakdown": category_counts,
            "discovery_stats": dict(self.discovery_stats),
            "base_symbols_count": len(self.base_symbols),
            "dynamic_symbols_count": total_symbols - len(self.base_symbols)
        }

if __name__ == "__main__":
    # Test the dynamic watchlist manager
    async def test_watchlist():
        manager = DynamicWatchlistManager()
        
        print("🧪 Testing Dynamic Watchlist Manager...")
        
        # REMOVED: MockDataManager - NO FAKE DATA ALLOWED  
        # This manager now requires real data manager to function
        print("❌ Dynamic Watchlist Manager test requires real data manager")
        print("❌ Fake data generation removed for safety")
        return None
        
        total_updates = sum(updates.values())
        print(f"✅ Updated watchlists: {total_updates} new symbols")
        
        # Get statistics
        stats = manager.get_watchlist_statistics()
        print(f"📊 Total symbols: {stats['total_unique_symbols']}")
        print(f"📈 Category breakdown: {stats['category_breakdown']}")
        
        # Get top symbols
        top_symbols = manager.get_top_symbols(10)
        print("🏆 Top symbols:")
        for symbol, score, category in top_symbols[:5]:
            print(f"  {symbol}: {score:.1f} ({category.value})")
    
    import asyncio
    asyncio.run(test_watchlist())

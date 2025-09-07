"""
Multi-Source Data Manager - Failure-Resistant Data Layer
Redundant data sources with automatic fallback for maximum uptime
UPDATED VERSION - Fixed async/sync issues and improved error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import yfinance as yf
from dataclasses import dataclass
from enum import Enum
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataSource(Enum):
    ALPACA = "alpaca"
    IEX = "iex"
    YAHOO = "yahoo"
    BACKUP = "backup"

@dataclass
class DataSourceConfig:
    name: str
    priority: int
    is_active: bool
    last_error: Optional[str] = None
    error_count: int = 0
    last_success: Optional[datetime] = None

class MultiSourceDataManager:
    def __init__(self, alpaca_key: str = None, alpaca_secret: str = None):
        """Initialize multi-source data manager with fallback capabilities"""
        
        # Data source configurations (priority order)
        self.data_sources = {
            DataSource.ALPACA: DataSourceConfig("Alpaca", 1, True),
            DataSource.IEX: DataSourceConfig("IEX", 2, True),
            DataSource.YAHOO: DataSourceConfig("Yahoo Finance", 3, True),
            DataSource.BACKUP: DataSourceConfig("Backup Cache", 4, True)
        }
        
        # Initialize Alpaca client
        self.alpaca_client = None
        if alpaca_key and alpaca_secret:
            try:
                self.alpaca_client = tradeapi.REST(
                    alpaca_key, alpaca_secret, 
                    base_url='https://paper-api.alpaca.markets'
                )
                logger.info("✅ Alpaca client initialized")
            except Exception as e:
                logger.error(f"❌ Alpaca initialization failed: {e}")
                self.data_sources[DataSource.ALPACA].is_active = False
        
        # Data cache for backup
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Thread safety
        self.cache_lock = threading.Lock()
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 1
        self.max_errors_before_disable = 5
        
        # Connection pooling
        self.connection_pool = {}
        self.pool_lock = threading.Lock()
        
        # Semaphore for rate limiting
        self.api_semaphore = asyncio.Semaphore(10)
        
        logger.info("🔄 Multi-source data manager initialized")

    async def get_market_data(self, symbol: str, timeframe: str = "1Min", 
                            limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data with automatic fallover between sources"""
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{limit}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            logger.debug(f"📦 Using cached data for {symbol}")
            return cached_data
        
        # Try each data source in priority order
        for source in sorted(self.data_sources.keys(), 
                           key=lambda x: self.data_sources[x].priority):
            
            if not self.data_sources[source].is_active:
                continue
                
            try:
                async with self.api_semaphore:  # Rate limiting
                    data = await self._fetch_from_source_async(source, symbol, timeframe, limit)
                    
                if data is not None and not data.empty:
                    self._mark_success(source)
                    self._cache_data(cache_key, data)
                    logger.debug(f"✅ Data fetched from {source.value}")
                    return data
                    
            except Exception as e:
                self._handle_error(source, str(e))
                logger.warning(f"⚠️  {source.value} failed: {e}")
                continue
        
        # Last resort: try cache again (maybe expired cache is better than nothing)
        cached_data = self._get_cached_data(cache_key, ignore_expiry=True)
        if cached_data is not None:
            logger.info("📦 Using expired cached data as fallback")
            return cached_data
            
        logger.error(f"❌ All data sources failed for {symbol}")
        return None

    async def _fetch_from_source_async(self, source: DataSource, symbol: str, 
                                     timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from specific source with proper async handling"""
        
        if source == DataSource.ALPACA:
            return await self._fetch_alpaca_data_async(symbol, timeframe, limit)
        elif source == DataSource.IEX:
            return await self._fetch_iex_data_async(symbol, timeframe, limit)
        elif source == DataSource.YAHOO:
            return await self._fetch_yahoo_data_async(symbol, timeframe, limit)
        elif source == DataSource.BACKUP:
            return self._get_cached_data(f"{symbol}_{timeframe}_{limit}", ignore_expiry=True)
        
        return None

    async def _fetch_alpaca_data_async(self, symbol: str, timeframe: str, 
                                     limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca with async wrapper"""
        if not self.alpaca_client:
            raise Exception("Alpaca client not initialized")
        
        # Run Alpaca API call in thread pool since it's synchronous
        loop = asyncio.get_event_loop()
        
        def fetch_alpaca_sync():
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=5)
                
                bars = self.alpaca_client.get_bars(
                    symbol, timeframe, 
                    start=start_time.isoformat(),
                    end=end_time.isoformat(),
                    limit=limit
                )
                
                if not bars:
                    return None
                    
                data = []
                for bar in bars:
                    data.append({
                        'timestamp': bar.timestamp,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume)
                    })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                return df
                
            except Exception as e:
                raise Exception(f"Alpaca API error: {e}")
        
        # Execute in thread pool
        with ThreadPoolExecutor(max_workers=5) as executor:
            future = loop.run_in_executor(executor, fetch_alpaca_sync)
            return await future

    async def _fetch_iex_data_async(self, symbol: str, timeframe: str, 
                                  limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from IEX Cloud (placeholder - needs API token)"""
        # This would need IEX Cloud implementation
        raise Exception("IEX Cloud not implemented - add your IEX token")

    async def _fetch_yahoo_data_async(self, symbol: str, timeframe: str, 
                                    limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance with async wrapper"""
        
        loop = asyncio.get_event_loop()
        
        def fetch_yahoo_sync():
            try:
                period_map = {
                    "1Min": "5d", "5Min": "1mo", "15Min": "3mo",
                    "1Hour": "6mo", "1Day": "1y"
                }
                
                interval_map = {
                    "1Min": "1m", "5Min": "5m", "15Min": "15m", 
                    "1Hour": "1h", "1Day": "1d"
                }
                
                period = period_map.get(timeframe, "1mo")
                interval = interval_map.get(timeframe, "5m")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    return None
                    
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                data.reset_index(inplace=True)
                
                # Ensure we have timestamp column
                if 'datetime' in data.columns:
                    data['timestamp'] = data['datetime']
                elif 'date' in data.columns:
                    data['timestamp'] = data['date']
                else:
                    data['timestamp'] = data.index
                    
                data.set_index('timestamp', inplace=True)
                
                # Limit to requested number of rows
                if len(data) > limit:
                    data = data.tail(limit)
                    
                return data
                
            except Exception as e:
                raise Exception(f"Yahoo Finance error: {e}")
        
        # Execute in thread pool
        with ThreadPoolExecutor(max_workers=5) as executor:
            future = loop.run_in_executor(executor, fetch_yahoo_sync)
            return await future

    async def get_multiple_symbols_data(self, symbols: List[str], 
                                      timeframe: str = "1Min") -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently with proper async"""
        results = {}
        
        # Create tasks for parallel execution
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self.get_market_data(symbol, timeframe),
                name=f"fetch_{symbol}"
            )
            tasks.append((symbol, task))
        
        # Execute all tasks concurrently
        for symbol, task in tasks:
            try:
                data = await task
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return results

    def _cache_data(self, key: str, data: pd.DataFrame):
        """Cache data with expiry (thread-safe)"""
        with self.cache_lock:
            self.data_cache[key] = data.copy()
            self.cache_expiry[key] = datetime.now() + self.cache_duration

    def _get_cached_data(self, key: str, ignore_expiry: bool = False) -> Optional[pd.DataFrame]:
        """Get cached data if not expired (thread-safe)"""
        with self.cache_lock:
            if key not in self.data_cache:
                return None
            
            if not ignore_expiry and datetime.now() > self.cache_expiry.get(key, datetime.min):
                # Data expired, remove it
                del self.data_cache[key]
                if key in self.cache_expiry:
                    del self.cache_expiry[key]
                return None
                
            return self.data_cache[key].copy()

    def _mark_success(self, source: DataSource):
        """Mark successful data fetch"""
        config = self.data_sources[source]
        config.last_success = datetime.now()
        config.error_count = 0
        config.last_error = None
        
        if not config.is_active:
            config.is_active = True
            logger.info(f"✅ Re-enabled {source.value} after successful fetch")

    def _handle_error(self, source: DataSource, error: str):
        """Handle data source error"""
        config = self.data_sources[source]
        config.error_count += 1
        config.last_error = error
        
        if config.error_count >= self.max_errors_before_disable:
            config.is_active = False
            logger.warning(f"⚠️  Disabled {source.value} after {config.error_count} errors")

    def get_data_source_status(self) -> Dict[str, Dict]:
        """Get status of all data sources"""
        status = {}
        for source, config in self.data_sources.items():
            status[source.value] = {
                'active': config.is_active,
                'priority': config.priority,
                'error_count': config.error_count,
                'last_error': config.last_error,
                'last_success': config.last_success.isoformat() if config.last_success else None
            }
        return status

    def reset_data_source(self, source: DataSource):
        """Reset a data source (re-enable and clear errors)"""
        config = self.data_sources[source]
        config.is_active = True
        config.error_count = 0
        config.last_error = None
        logger.info(f"🔄 Reset {source.value}")

    def get_health_score(self) -> float:
        """Calculate overall data system health (0-100)"""
        active_sources = sum(1 for config in self.data_sources.values() 
                           if config.is_active)
        total_sources = len(self.data_sources)
        
        base_score = (active_sources / total_sources) * 100
        recent_errors = sum(config.error_count for config in self.data_sources.values())
        error_penalty = min(recent_errors * 5, 30)
        
        return max(0, base_score - error_penalty)

    async def start_health_monitor(self):
        """Start background health monitoring"""
        while True:
            try:
                # Test each data source health
                for source in self.data_sources:
                    if self.data_sources[source].is_active:
                        try:
                            # Quick health check with a simple symbol
                            await self._fetch_from_source_async(source, "SPY", "1Day", 1)
                        except Exception as e:
                            logger.warning(f"Health check failed for {source.value}: {e}")
                
                # Clean up old cache entries
                self._cleanup_cache()
                
                # Wait 5 minutes before next health check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)

    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        with self.cache_lock:
            current_time = datetime.now()
            expired_keys = [
                key for key, expiry_time in self.cache_expiry.items()
                if current_time > expiry_time
            ]
            
            for key in expired_keys:
                if key in self.data_cache:
                    del self.data_cache[key]
                del self.cache_expiry[key]
            
            if expired_keys:
                logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

    async def test_all_sources(self) -> Dict[str, bool]:
        """Test all data sources and return their status"""
        test_results = {}
        
        for source in self.data_sources:
            try:
                data = await self._fetch_from_source_async(source, "SPY", "1Day", 1)
                test_results[source.value] = data is not None and not data.empty
            except Exception as e:
                logger.error(f"Test failed for {source.value}: {e}")
                test_results[source.value] = False
        
        return test_results

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        with self.cache_lock:
            total_entries = len(self.data_cache)
            total_size = sum(
                data.memory_usage(deep=True).sum() 
                for data in self.data_cache.values()
            ) / (1024 * 1024)  # MB
            
            return {
                "total_entries": total_entries,
                "total_size_mb": round(total_size, 2),
                "cache_hit_rate": self._calculate_cache_hit_rate()
            }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would need proper hit/miss tracking in production
        return 0.75  # Placeholder

# Test function
async def test_data_manager():
    """Test the multi-source data manager"""
    manager = MultiSourceDataManager()
    
    print("🧪 Testing Multi-Source Data Manager...")
    
    # Test single symbol
    data = await manager.get_market_data("AAPL", "5Min", 50)
    if data is not None:
        print(f"✅ AAPL data: {len(data)} rows")
        print(data.head())
    else:
        print("❌ Failed to get AAPL data")
    
    # Test multiple symbols
    symbols = ["SPY", "QQQ", "AAPL"]
    multi_data = await manager.get_multiple_symbols_data(symbols, "1Day")
    print(f"📊 Multi-symbol data: {len(multi_data)} symbols fetched")
    
    # Check health
    health = manager.get_health_score()
    print(f"📊 Data system health: {health}%")
    
    # Check status
    status = manager.get_data_source_status()
    for source, info in status.items():
        print(f"📡 {source}: {'🟢' if info['active'] else '🔴'} Active")

if __name__ == "__main__":
    asyncio.run(test_data_manager())

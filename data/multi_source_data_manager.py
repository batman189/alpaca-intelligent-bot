"""
Multi-Source Data Manager - Failure-Resistant Data Layer
Redundant data sources with automatic fallback for maximum uptime
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
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 1
        self.max_errors_before_disable = 5
        
        logger.info("🔄 Multi-source data manager initialized")

    async def get_market_data(self, symbol: str, timeframe: str = "1Min", 
                            limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data with automatic fallback between sources"""
        for source in sorted(self.data_sources.keys(), 
                           key=lambda x: self.data_sources[x].priority):
            
            if not self.data_sources[source].is_active:
                continue
                
            try:
                data = await self._fetch_from_source(source, symbol, timeframe, limit)
                if data is not None and not data.empty:
                    self._mark_success(source)
                    self._cache_data(f"{symbol}_{timeframe}", data)
                    logger.debug(f"✅ Data fetched from {source.value}")
                    return data
                    
            except Exception as e:
                self._handle_error(source, str(e))
                logger.warning(f"⚠️  {source.value} failed: {e}")
                continue
        
        # Last resort: try cache
        cached_data = self._get_cached_data(f"{symbol}_{timeframe}")
        if cached_data is not None:
            logger.info("📦 Using cached data as fallback")
            return cached_data
            
        logger.error(f"❌ All data sources failed for {symbol}")
        return None

    async def _fetch_from_source(self, source: DataSource, symbol: str, 
                                timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from specific source"""
        
        if source == DataSource.ALPACA:
            return await self._fetch_alpaca_data(symbol, timeframe, limit)
        elif source == DataSource.IEX:
            return await self._fetch_iex_data(symbol, timeframe, limit)
        elif source == DataSource.YAHOO:
            return await self._fetch_yahoo_data(symbol, timeframe, limit)
        elif source == DataSource.BACKUP:
            return self._get_cached_data(f"{symbol}_{timeframe}")
        
        return None

    async def _fetch_alpaca_data(self, symbol: str, timeframe: str, 
                               limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca"""
        if not self.alpaca_client:
            raise Exception("Alpaca client not initialized")
            
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df

    async def _fetch_iex_data(self, symbol: str, timeframe: str, 
                            limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from IEX Cloud (placeholder - needs API token)"""
        raise Exception("IEX token not configured")

    async def _fetch_yahoo_data(self, symbol: str, timeframe: str, 
                              limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
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

    def get_multiple_symbols_data(self, symbols: List[str], 
                                 timeframe: str = "1Min") -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently"""
        results = {}
        
        async def fetch_all():
            tasks = []
            for symbol in symbols:
                task = self.get_market_data(symbol, timeframe)
                tasks.append((symbol, task))
            
            for symbol, task in tasks:
                try:
                    data = await task
                    if data is not None:
                        results[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
        
        # Run async tasks
        try:
            asyncio.run(fetch_all())
        except Exception as e:
            logger.error(f"Async fetch error: {e}")
            
        return results

    def _cache_data(self, key: str, data: pd.DataFrame):
        """Cache data with expiry"""
        self.data_cache[key] = data.copy()
        self.cache_expiry[key] = datetime.now() + self.cache_duration

    def _get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if not expired"""
        if key not in self.data_cache:
            return None
            
        if datetime.now() > self.cache_expiry.get(key, datetime.min):
            del self.data_cache[key]
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

if __name__ == "__main__":
    # Test the multi-source data manager
    async def test_data_manager():
        manager = MultiSourceDataManager()
        
        print("🧪 Testing Multi-Source Data Manager...")
        
        # Test single symbol
        data = await manager.get_market_data("AAPL", "5Min", 50)
        if data is not None:
            print(f"✅ AAPL data: {len(data)} rows")
            print(data.head())
        else:
            print("❌ Failed to get AAPL data")
        
        # Check health
        health = manager.get_health_score()
        print(f"📊 Data system health: {health}%")
        
        # Check status
        status = manager.get_data_source_status()
        for source, info in status.items():
            print(f"📡 {source}: {'🟢' if info['active'] else '🔴'} Active")
    
    asyncio.run(test_data_manager())
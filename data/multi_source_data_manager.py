"""
Multi-Source Data Manager - FIXED VERSION
Robust data collection with proper fallbacks and error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from datetime import datetime, timedelta
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import time

# Safe imports with fallbacks
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CriticalDataFailureError(Exception):
    """Raised when all real data sources fail"""
    pass

class TradingHaltManager:
    """Manages trading halts due to data source failures"""
    def __init__(self):
        self.is_halted = False
        self.halt_reason = None
        self.halt_start_time = None
        self.consecutive_failures = 0
        
    def should_halt_trading(self, active_real_sources: int, total_failures: int) -> bool:
        """Determine if trading should be halted"""
        # Halt if no real data sources are available
        if active_real_sources == 0:
            return True
        # Halt if we have multiple consecutive failures
        if total_failures >= 5:
            return True
        return False
    
    def halt_trading(self, reason: str):
        """Halt all trading operations"""
        if not self.is_halted:
            self.is_halted = True
            self.halt_reason = reason
            self.halt_start_time = datetime.now()
            logger.critical(f"🚨 TRADING HALTED: {reason}")
            logger.critical("🚨 ALL TRADING OPERATIONS SUSPENDED FOR SAFETY")
    
    def resume_trading(self, reason: str = "Data sources recovered"):
        """Resume trading operations"""
        if self.is_halted:
            halt_duration = datetime.now() - self.halt_start_time
            logger.info(f"✅ TRADING RESUMED: {reason} (Halted for {halt_duration})")
            self.is_halted = False
            self.halt_reason = None
            self.halt_start_time = None
            self.consecutive_failures = 0

class DataSourceAlert:
    """Alert system for critical data source failures"""
    def __init__(self):
        self.alert_threshold = 3
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes
        
    def should_send_alert(self, source: str, current_failures: int) -> bool:
        """Check if we should send an alert"""
        if current_failures < self.alert_threshold:
            return False
            
        last_alert = self.last_alert_time.get(source, 0)
        if time.time() - last_alert < self.alert_cooldown:
            return False
            
        return True
    
    def send_critical_alert(self, source: str, error_msg: str, symbol: str = None):
        """Send critical alert for data source failure"""
        try:
            alert_msg = f"🚨 CRITICAL DATA SOURCE FAILURE: {source}"
            if symbol:
                alert_msg += f" for symbol {symbol}"
            alert_msg += f"\nError: {error_msg}\nTime: {datetime.now()}"
            alert_msg += "\n⚠️  TRADING MAY BE COMPROMISED - MANUAL INTERVENTION REQUIRED"
            
            logger.critical(alert_msg)
            self.last_alert_time[source] = time.time()
            
            # TODO: Integrate with email/SMS/Slack alerts
            
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")

class DataSource(Enum):
    ALPACA = "alpaca"
    YAHOO = "yahoo"
    # REMOVED: MOCK = "mock"  # No mock data in production!

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
        """Initialize multi-source data manager with SAFE fallbacks (NO MOCK DATA)"""
        
        # Safety systems
        self.alert_system = DataSourceAlert()
        self.trading_halt_manager = TradingHaltManager()
        
        # Data source configurations (REAL DATA ONLY)
        self.data_sources = {
            DataSource.ALPACA: DataSourceConfig("Alpaca", 1, ALPACA_AVAILABLE),
            DataSource.YAHOO: DataSourceConfig("Yahoo Finance", 2, YAHOO_AVAILABLE)
            # REMOVED: Mock data - never use fake data for trading!
        }
        
        # Initialize Alpaca client
        self.alpaca_client = None
        if alpaca_key and alpaca_secret and ALPACA_AVAILABLE:
            try:
                # Use live market data API for your $100/month plan
                # Paper API only gives you paper trading data, not real market data
                import os
                base_url = os.getenv('APCA_API_BASE_URL', 'https://api.alpaca.markets')  # Live API for real market data
                
                self.alpaca_client = tradeapi.REST(
                    alpaca_key, alpaca_secret, 
                    base_url=base_url
                )
                logger.info("[OK] Alpaca client initialized")
            except Exception as e:
                logger.error(f"[ERROR] Alpaca initialization failed: {e}")
                self.data_sources[DataSource.ALPACA].is_active = False
        elif not ALPACA_AVAILABLE:
            logger.warning("[WARN] Alpaca Trade API not available")
            self.data_sources[DataSource.ALPACA].is_active = False
        
        # Data cache for fallback
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Thread safety
        self.cache_lock = threading.Lock()
        
        # Error handling
        self.max_retries = 2
        self.retry_delay = 1
        self.max_errors_before_disable = 3
        
        # Rate limiting
        self.api_semaphore = asyncio.Semaphore(5)
        
        logger.info("🔄 Multi-source data manager initialized")

    async def get_market_data(self, symbol: str, timeframe: str = "15Min", 
                            limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data with automatic fallback between sources"""
        
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
                    logger.debug(f"✅ Data fetched from {source.value} for {symbol}")
                    return data
                    
            except Exception as e:
                self._handle_error(source, str(e))
                logger.warning(f"⚠️ {source.value} failed for {symbol}: {e}")
                continue
        
        # 🚨 CRITICAL: ALL REAL DATA SOURCES HAVE FAILED!
        logger.critical(f"🚨 CRITICAL FAILURE: All data sources failed for {symbol}")
        logger.critical("🚨 NO REAL MARKET DATA AVAILABLE - CANNOT CONTINUE SAFELY")
        
        # Count active real sources and total failures
        active_real_sources = sum(1 for config in self.data_sources.values() if config.is_active)
        total_failures = sum(config.error_count for config in self.data_sources.values())
        
        # Send critical alert
        self.alert_system.send_critical_alert(
            "ALL_SOURCES", 
            f"All data sources failed for {symbol}", 
            symbol
        )
        
        # Check if trading should be halted
        if self.trading_halt_manager.should_halt_trading(active_real_sources, total_failures):
            self.trading_halt_manager.halt_trading(
                f"All data sources failed - no real data available"
            )
        
        # NEVER GENERATE FAKE DATA - Raise exception instead
        raise CriticalDataFailureError(
            f"All real data sources failed for {symbol}. Trading halted for safety."
        )

    async def _fetch_from_source_async(self, source: DataSource, symbol: str, 
                                     timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from specific source with proper async handling"""
        
        if source == DataSource.ALPACA:
            return await self._fetch_alpaca_data_async(symbol, timeframe, limit)
        elif source == DataSource.YAHOO:
            return await self._fetch_yahoo_data_async(symbol, timeframe, limit)
        
        # NO MOCK DATA - only real sources allowed
        logger.error(f"Unknown data source: {source}")
        return None

    async def _fetch_alpaca_data_async(self, symbol: str, timeframe: str, 
                                     limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca with proper date formatting"""
        if not self.alpaca_client:
            raise Exception("Alpaca client not initialized")
        
        loop = asyncio.get_event_loop()
        
        def fetch_alpaca_sync():
            try:
                # Fix the date formatting issue - use proper format for Alpaca API
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)  # Get enough historical data
                
                # Format dates properly for Alpaca API (ISO format without microseconds)
                start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Map timeframes to Alpaca format
                timeframe_map = {
                    "1Min": "1Min", "5Min": "5Min", "15Min": "15Min", 
                    "1Hour": "1Hour", "1Day": "1Day"
                }
                alpaca_timeframe = timeframe_map.get(timeframe, "15Min")
                
                bars = self.alpaca_client.get_bars(
                    symbol, 
                    alpaca_timeframe,
                    start=start_str,
                    end=end_str,
                    limit=limit
                )
                
                if not bars:
                    return None
                    
                data = []
                for bar in bars:
                    data.append({
                        'timestamp': bar.t,
                        'open': float(bar.o),
                        'high': float(bar.h),
                        'low': float(bar.l),
                        'close': float(bar.c),
                        'volume': int(bar.v)
                    })
                
                if not data:
                    return None
                
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Limit to requested number of rows
                if len(df) > limit:
                    df = df.tail(limit)
                
                return df
                
            except Exception as e:
                raise Exception(f"Alpaca API error: {e}")
        
        # Execute in thread pool
        with ThreadPoolExecutor(max_workers=3) as executor:
            future = loop.run_in_executor(executor, fetch_alpaca_sync)
            return await future

    async def _fetch_yahoo_data_async(self, symbol: str, timeframe: str, 
                                    limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance with better error handling"""
        
        if not YAHOO_AVAILABLE:
            raise Exception("Yahoo Finance not available")
        
        loop = asyncio.get_event_loop()
        
        def fetch_yahoo_sync():
            try:
                # Map timeframes and periods for Yahoo Finance
                # Updated to respect Yahoo Finance limitations
                period_map = {
                    "1Min": "5d",    # 1-minute data: max 7 days
                    "5Min": "1mo",   # 5-minute data: max 60 days  
                    "15Min": "2mo",  # 15-minute data: max 60 days (reduced from 3mo)
                    "1Hour": "6mo",  # 1-hour data: max 730 days
                    "1Day": "1y"     # 1-day data: max many years
                }
                
                interval_map = {
                    "1Min": "1m", "5Min": "5m", "15Min": "15m", 
                    "1Hour": "1h", "1Day": "1d"
                }
                
                period = period_map.get(timeframe, "3mo")
                interval = interval_map.get(timeframe, "15m")
                
                # Create ticker object with timeout and better error handling
                ticker = yf.Ticker(symbol)
                
                # Try to get data with error handling
                data = ticker.history(
                    period=period, 
                    interval=interval,
                    prepost=False,  # No pre/post market data
                    auto_adjust=True,  # Adjust for splits/dividends
                    back_adjust=False,
                    repair=False,
                    keepna=False,
                    actions=False
                )
                
                if data.empty:
                    # Try alternative approach for stubborn symbols
                    data = yf.download(
                        symbol, 
                        period=period, 
                        interval=interval,
                        progress=False
                        # Removed show_errors - deprecated in newer yfinance versions
                    )
                
                if data.empty:
                    return None
                    
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                data.reset_index(inplace=True)
                
                # Handle different date column names
                date_col = None
                for col in ['datetime', 'date', 'timestamp']:
                    if col in data.columns:
                        date_col = col
                        break
                
                if date_col:
                    data['timestamp'] = pd.to_datetime(data[date_col])
                else:
                    data['timestamp'] = data.index
                    
                data.set_index('timestamp', inplace=True)
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in data.columns:
                        # Try alternative column names
                        alt_names = {
                            'open': ['Open'],
                            'high': ['High'], 
                            'low': ['Low'],
                            'close': ['Close', 'Adj Close'],
                            'volume': ['Volume']
                        }
                        found = False
                        for alt_name in alt_names.get(col, []):
                            if alt_name.lower() in data.columns:
                                data[col] = data[alt_name.lower()]
                                found = True
                                break
                        if not found:
                            raise Exception(f"Missing required column: {col}")
                
                # Clean the data
                data = data[required_cols].dropna()
                
                # Limit to requested number of rows
                if len(data) > limit:
                    data = data.tail(limit)
                    
                return data
                
            except Exception as e:
                raise Exception(f"Yahoo Finance error: {e}")
        
        # Execute in thread pool with timeout
        with ThreadPoolExecutor(max_workers=3) as executor:
            future = loop.run_in_executor(executor, fetch_yahoo_sync)
            try:
                return await asyncio.wait_for(future, timeout=30)  # 30 second timeout
            except asyncio.TimeoutError:
                raise Exception("Yahoo Finance request timeout")

    # REMOVED: _generate_mock_data() - No fake data allowed in production trading!
    # Mock data generation was dangerous and could lead to trading on invalid data.
    # All data must come from real market sources (Alpaca, Yahoo Finance, etc.)

    async def get_multiple_symbols_data(self, symbols: List[str], 
                                      timeframe: str = "15Min") -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently"""
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
                    logger.debug(f"✅ Got data for {symbol}: {len(data)} bars")
                else:
                    logger.warning(f"⚠️ No data for {symbol}")
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
        """Handle data source error with enhanced safety measures"""
        config = self.data_sources[source]
        config.error_count += 1
        config.last_error = error
        
        # Send alert if threshold reached
        if self.alert_system.should_send_alert(source.value, config.error_count):
            self.alert_system.send_critical_alert(source.value, error)
        
        # Disable source if too many errors
        if config.error_count >= self.max_errors_before_disable:
            config.is_active = False
            logger.critical(f"🚨 DISABLED {source.value} after {config.error_count} errors")
            
            # Check if this leaves us with no active sources
            active_sources = sum(1 for c in self.data_sources.values() if c.is_active)
            if active_sources == 0:
                logger.critical("🚨 CRITICAL: NO DATA SOURCES REMAIN ACTIVE!")
                self.trading_halt_manager.halt_trading("All data sources disabled")
                self.alert_system.send_critical_alert(
                    "SYSTEM", 
                    "All data sources disabled - no real data available"
                )

    def get_data_source_status(self) -> Dict[str, Dict]:
        """Get status of all data sources with safety information"""
        status = {}
        for source, config in self.data_sources.items():
            status[source.value] = {
                'active': config.is_active,
                'priority': config.priority,
                'error_count': config.error_count,
                'last_error': config.last_error,
                'last_success': config.last_success.isoformat() if config.last_success else None,
                'is_real_data': True  # All sources are real data now
            }
        
        # Add system safety status
        status['system'] = {
            'trading_halted': self.trading_halt_manager.is_halted,
            'halt_reason': self.trading_halt_manager.halt_reason,
            'halt_start_time': self.trading_halt_manager.halt_start_time.isoformat() if self.trading_halt_manager.halt_start_time else None,
            'health_score': self.get_health_score(),
            'active_sources': sum(1 for c in self.data_sources.values() if c.is_active),
            'total_sources': len(self.data_sources),
            'uses_real_data_only': True
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
        """Calculate overall data system health (0-100) - real data sources only"""
        active_sources = sum(1 for config in self.data_sources.values() if config.is_active)
        total_sources = len(self.data_sources)
        
        if total_sources == 0:
            return 0
        
        base_score = (active_sources / total_sources) * 100
        recent_errors = sum(config.error_count for config in self.data_sources.values())
        error_penalty = min(recent_errors * 10, 50)  # Increased penalty for errors
        
        final_score = max(0, base_score - error_penalty)
        
        # Log critical health warnings
        if final_score < 50:
            logger.critical(f"🚨 CRITICAL: Data health at {final_score}% - Trading safety compromised!")
        elif final_score < 70:
            logger.warning(f"⚠️ Warning: Data health at {final_score}% - Monitor closely")
        
        return final_score

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
                "cache_duration_minutes": self.cache_duration.total_seconds() / 60
            }

# Test function
async def test_data_manager():
    """Test the multi-source data manager"""
    manager = MultiSourceDataManager()
    
    print("🧪 Testing Multi-Source Data Manager...")
    
    # Test single symbol
    data = await manager.get_market_data("AAPL", "15Min", 50)
    if data is not None:
        print(f"✅ AAPL data: {len(data)} rows")
        print(data.head())
    else:
        print("❌ Failed to get AAPL data")
    
    # Test multiple symbols
    symbols = ["SPY", "QQQ", "AAPL"]
    multi_data = await manager.get_multiple_symbols_data(symbols, "15Min")
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

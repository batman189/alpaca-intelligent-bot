"""
Enhanced Data Client for Professional Trading Bot
Provides multi-timeframe data aggregation, real-time streaming, and market data caching
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import time
import talib
from alpaca_trade_api import REST, Stream
from alpaca_trade_api.common import URL
import websocket
import json
import queue
import concurrent.futures
import gc  # Phase 3 Memory Optimization

@dataclass
class MarketData:
    """Container for market data with metadata"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

@dataclass
class RealtimeQuote:
    """Real-time quote data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask - self.bid

@dataclass
class MultiTimeframeData:
    """Multi-timeframe data container"""
    symbol: str
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    
    def get_latest(self, timeframe: str) -> Optional[MarketData]:
        """Get latest data point for timeframe"""
        if timeframe in self.data and not self.data[timeframe].empty:
            row = self.data[timeframe].iloc[-1]
            return MarketData(
                symbol=self.symbol,
                timestamp=row.name,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timeframe=timeframe,
                vwap=row.get('vwap'),
                trade_count=row.get('trade_count')
            )
        return None

class EnhancedDataClient:
    """
    Professional-grade data client with multi-timeframe support
    Handles real-time data, historical data, and market analysis
    """
    
    SUPPORTED_TIMEFRAMES = ['1Min', '5Min', '15Min', '1Hour', '1Day']
    # Phase 3 Memory Optimization - Dynamic cache sizing based on timeframe
    CACHE_SIZES = {
        '1Min': 500,    # ~8 hours of 1-min data
        '5Min': 300,    # ~1 day of 5-min data  
        '15Min': 200,   # ~2 days of 15-min data
        '1Hour': 168,   # ~1 week of hourly data
        '1Day': 100     # ~3 months of daily data
    }
    MAX_CACHE_SIZE = 500  # Fallback for compatibility
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        
        # Initialize Alpaca REST API
        self.api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )
        
        # Initialize streaming with your $100/month plan
        self.stream = Stream(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            data_feed='sip'  # Use SIP for your paid plan (more comprehensive than IEX)
        )
        
        # Data storage
        self.market_data: Dict[str, MultiTimeframeData] = {}
        self.realtime_quotes: Dict[str, RealtimeQuote] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Caching and performance
        self.data_lock = threading.RLock()
        self.cache_timestamps: Dict[str, datetime] = {}
        self.update_intervals = {
            '1Min': timedelta(seconds=30),
            '5Min': timedelta(minutes=2),
            '15Min': timedelta(minutes=7),
            '1Hour': timedelta(minutes=30),
            '1Day': timedelta(hours=4)
        }
        
        # Performance monitoring
        self.data_stats = {
            'updates_per_minute': deque(maxlen=60),
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0
        }
        
        # Background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.background_tasks = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Data Client initialized")
    
    async def start(self, symbols: List[str]):
        """Start the data client with real-time streaming"""
        self.running = True
        self.logger.info(f"Starting data client for symbols: {symbols}")
        
        # Initialize market data containers
        for symbol in symbols:
            self.market_data[symbol] = MultiTimeframeData(symbol=symbol)
        
        # Start background data collection
        background_task = asyncio.create_task(self._background_data_collector(symbols))
        self.background_tasks.append(background_task)
        
        # Start real-time streaming
        await self._start_realtime_stream(symbols)
        
        self.logger.info("Data client started successfully")
    
    async def stop(self):
        """Stop all data collection and streaming"""
        self.running = False
        
        # Stop streaming
        if hasattr(self.stream, 'stop'):
            await self.stream.stop()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Data client stopped")
    
    async def _start_realtime_stream(self, symbols: List[str]):
        """Start real-time data streaming"""
        try:
            # Subscribe to quotes
            for symbol in symbols:
                self.stream.subscribe_quotes(self._handle_quote, symbol)
                self.stream.subscribe_trades(self._handle_trade, symbol)
            
            # Start streaming in background
            stream_task = asyncio.create_task(self.stream._run_forever())
            self.background_tasks.append(stream_task)
            
        except Exception as e:
            self.logger.error(f"Error starting real-time stream: {e}")
    
    def _handle_quote(self, quote):
        """Handle real-time quote updates"""
        try:
            symbol = quote.symbol
            realtime_quote = RealtimeQuote(
                symbol=symbol,
                timestamp=quote.timestamp,
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                bid_size=int(quote.bid_size),
                ask_size=int(quote.ask_size)
            )
            
            with self.data_lock:
                self.realtime_quotes[symbol] = realtime_quote
            
            # Notify subscribers
            self._notify_subscribers(f"quote_{symbol}", realtime_quote)
            
        except Exception as e:
            self.logger.error(f"Error handling quote: {e}")
    
    def _handle_trade(self, trade):
        """Handle real-time trade updates"""
        try:
            # Update minute bar with trade data
            self._update_minute_bar(trade.symbol, trade.price, trade.size, trade.timestamp)
            
        except Exception as e:
            self.logger.error(f"Error handling trade: {e}")
    
    def _update_minute_bar(self, symbol: str, price: float, volume: int, timestamp):
        """Update minute-level bars with trade data"""
        try:
            minute_timestamp = timestamp.replace(second=0, microsecond=0)
            
            with self.data_lock:
                if symbol not in self.market_data:
                    self.market_data[symbol] = MultiTimeframeData(symbol=symbol)
                
                # Initialize 1Min data if needed
                if '1Min' not in self.market_data[symbol].data:
                    self.market_data[symbol].data['1Min'] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                
                df = self.market_data[symbol].data['1Min']
                
                # Update current minute bar
                if minute_timestamp in df.index:
                    # Update existing bar
                    df.loc[minute_timestamp, 'high'] = max(df.loc[minute_timestamp, 'high'], price)
                    df.loc[minute_timestamp, 'low'] = min(df.loc[minute_timestamp, 'low'], price)
                    df.loc[minute_timestamp, 'close'] = price
                    df.loc[minute_timestamp, 'volume'] += volume
                else:
                    # Create new bar
                    new_row = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume
                    }
                    df.loc[minute_timestamp] = new_row
                
                # Phase 3 Memory Optimization - Use dynamic cache sizing
                cache_size = self.CACHE_SIZES.get('1Min', self.MAX_CACHE_SIZE)
                if len(df) > cache_size:
                    df = df.tail(cache_size)
                    self.market_data[symbol].data['1Min'] = df
                
                # Update aggregated timeframes
                self._update_aggregated_timeframes(symbol)
                
        except Exception as e:
            self.logger.error(f"Error updating minute bar: {e}")
    
    def _update_aggregated_timeframes(self, symbol: str):
        """Update higher timeframe data from minute data"""
        try:
            minute_data = self.market_data[symbol].data.get('1Min')
            if minute_data is None or minute_data.empty:
                return
            
            # Update 5Min bars
            five_min_data = minute_data.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            cache_size = self.CACHE_SIZES.get('5Min', self.MAX_CACHE_SIZE)
            self.market_data[symbol].data['5Min'] = five_min_data.tail(cache_size)
            
            # Update 15Min bars
            fifteen_min_data = minute_data.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            cache_size = self.CACHE_SIZES.get('15Min', self.MAX_CACHE_SIZE)
            self.market_data[symbol].data['15Min'] = fifteen_min_data.tail(cache_size)
            
            # Update 1Hour bars
            hour_data = minute_data.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            cache_size = self.CACHE_SIZES.get('1Hour', self.MAX_CACHE_SIZE)
            self.market_data[symbol].data['1Hour'] = hour_data.tail(cache_size)
            
        except Exception as e:
            self.logger.error(f"Error updating aggregated timeframes: {e}")
    
    async def _background_data_collector(self, symbols: List[str]):
        """Background task to collect and update historical data"""
        while self.running:
            try:
                for symbol in symbols:
                    await self._update_historical_data(symbol)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in background data collector: {e}")
                await asyncio.sleep(60)
    
    async def _update_historical_data(self, symbol: str):
        """Update historical data for a symbol"""
        try:
            current_time = datetime.now()
            
            for timeframe in self.SUPPORTED_TIMEFRAMES:
                cache_key = f"{symbol}_{timeframe}"
                
                # Check if update is needed
                last_update = self.cache_timestamps.get(cache_key, datetime.min)
                update_interval = self.update_intervals.get(timeframe, timedelta(minutes=5))
                
                if current_time - last_update < update_interval:
                    continue
                
                # Get historical data
                await self._fetch_historical_data(symbol, timeframe)
                self.cache_timestamps[cache_key] = current_time
                
        except Exception as e:
            self.logger.error(f"Error updating historical data for {symbol}: {e}")
    
    async def _fetch_historical_data(self, symbol: str, timeframe: str):
        """Fetch historical data from Alpaca API"""
        try:
            # Map timeframe to Alpaca format
            alpaca_timeframe = self._map_timeframe(timeframe)
            
            # Calculate start date and format for Alpaca API
            if timeframe == '1Min':
                start_date = datetime.now() - timedelta(days=5)
            elif timeframe in ['5Min', '15Min']:
                start_date = datetime.now() - timedelta(days=30)
            elif timeframe == '1Hour':
                start_date = datetime.now() - timedelta(days=90)
            else:  # 1Day
                start_date = datetime.now() - timedelta(days=365)
            
            # Format dates for Alpaca API (ISO format)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch bars
            bars = self.api.get_bars(
                symbol,
                alpaca_timeframe,
                start=start_str,
                end=end_str,
                limit=self.MAX_CACHE_SIZE
            )
            
            if not bars:
                return
            
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.t,
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v),
                    'vwap': float(bar.vw) if bar.vw else None,
                    'trade_count': int(bar.n) if bar.n else None
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            with self.data_lock:
                if symbol not in self.market_data:
                    self.market_data[symbol] = MultiTimeframeData(symbol=symbol)
                
                self.market_data[symbol].data[timeframe] = df
                self.market_data[symbol].last_update = datetime.now()
            
            self.data_stats['api_calls'] += 1
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol} {timeframe}: {e}")
    
    def _map_timeframe(self, timeframe: str) -> str:
        """Map internal timeframe to Alpaca API format"""
        mapping = {
            '1Min': '1Min',
            '5Min': '5Min',
            '15Min': '15Min',
            '1Hour': '1Hour',
            '1Day': '1Day'
        }
        return mapping.get(timeframe, '1Min')
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            if len(df) < 20:
                return df
            
            # Moving averages
            df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            df['ema_20'] = talib.EMA(df['close'].values, timeperiod=20)
            
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(df['volume'].values.astype(float), timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def get_data(self, symbol: str, timeframe: str, periods: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data for symbol and timeframe"""
        try:
            with self.data_lock:
                if symbol not in self.market_data:
                    self.data_stats['cache_misses'] += 1
                    return None
                
                if timeframe not in self.market_data[symbol].data:
                    self.data_stats['cache_misses'] += 1
                    return None
                
                df = self.market_data[symbol].data[timeframe]
                self.data_stats['cache_hits'] += 1
                
                return df.tail(periods) if len(df) > periods else df
                
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol} {timeframe}: {e}")
            return None
    
    def get_latest_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Get latest real-time quote"""
        with self.data_lock:
            return self.realtime_quotes.get(symbol)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from latest quote or bar"""
        quote = self.get_latest_quote(symbol)
        if quote:
            return (quote.bid + quote.ask) / 2
        
        # Fallback to latest bar
        latest_data = self.get_latest_data(symbol, '1Min')
        if latest_data:
            return latest_data.close
        
        return None
    
    def get_latest_data(self, symbol: str, timeframe: str) -> Optional[MarketData]:
        """Get latest market data for symbol and timeframe"""
        with self.data_lock:
            if symbol in self.market_data:
                return self.market_data[symbol].get_latest(timeframe)
        return None
    
    def subscribe_to_updates(self, event_type: str, callback: Callable):
        """Subscribe to data updates"""
        self.subscribers[event_type].append(callback)
    
    def _notify_subscribers(self, event_type: str, data):
        """Notify subscribers of data updates"""
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in subscriber callback: {e}")
    
    def get_multi_timeframe_analysis(self, symbol: str) -> Dict:
        """Get comprehensive multi-timeframe analysis"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'timeframes': {}
        }
        
        for timeframe in self.SUPPORTED_TIMEFRAMES:
            data = self.get_data(symbol, timeframe, 50)
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                analysis['timeframes'][timeframe] = {
                    'price': latest['close'],
                    'change_pct': ((latest['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close']) * 100 if len(data) > 1 else 0,
                    'volume_ratio': latest.get('volume_ratio', 1),
                    'rsi': latest.get('rsi'),
                    'above_sma20': latest['close'] > latest.get('sma_20', latest['close']),
                    'bb_position': self._calculate_bb_position(latest) if 'bb_upper' in latest else None
                }
        
        return analysis
    
    def _calculate_bb_position(self, row) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        if pd.isna(row.get('bb_upper')) or pd.isna(row.get('bb_lower')):
            return 0.5
        
        bb_range = row['bb_upper'] - row['bb_lower']
        if bb_range == 0:
            return 0.5
        
        return (row['close'] - row['bb_lower']) / bb_range
    
    def get_performance_stats(self) -> Dict:
        """Get data client performance statistics"""
        cache_total = self.data_stats['cache_hits'] + self.data_stats['cache_misses']
        cache_hit_rate = (self.data_stats['cache_hits'] / cache_total) if cache_total > 0 else 0
        
        # Phase 3 Memory Optimization - Enhanced memory tracking
        memory_by_timeframe = {}
        total_memory = 0
        total_rows = 0
        
        for symbol, symbol_data in self.market_data.items():
            for timeframe, df in symbol_data.data.items():
                if timeframe not in memory_by_timeframe:
                    memory_by_timeframe[timeframe] = {'memory_mb': 0, 'rows': 0}
                
                df_memory = df.memory_usage(deep=True).sum()
                memory_by_timeframe[timeframe]['memory_mb'] += df_memory / (1024 * 1024)
                memory_by_timeframe[timeframe]['rows'] += len(df)
                total_memory += df_memory
                total_rows += len(df)
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'api_calls': self.data_stats['api_calls'],
            'cached_symbols': len(self.market_data),
            'active_streams': len(self.realtime_quotes),
            'total_memory_mb': total_memory / (1024 * 1024),
            'total_rows': total_rows,
            'memory_by_timeframe': memory_by_timeframe,
            'cache_efficiency': {tf: self.CACHE_SIZES.get(tf, self.MAX_CACHE_SIZE) for tf in self.SUPPORTED_TIMEFRAMES}
        }
    
    def optimize_memory_usage(self) -> Dict:
        """Phase 3 Memory Optimization - Optimize data types and cleanup memory"""
        optimization_stats = {
            'memory_before_mb': 0,
            'memory_after_mb': 0,
            'rows_optimized': 0,
            'symbols_optimized': 0
        }
        
        # Calculate initial memory usage
        for symbol_data in self.market_data.values():
            for df in symbol_data.data.values():
                optimization_stats['memory_before_mb'] += df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Optimize data types for each symbol
        for symbol, symbol_data in self.market_data.items():
            for timeframe, df in symbol_data.data.items():
                if len(df) > 0:
                    # Convert to more efficient data types
                    original_dtypes = df.dtypes.copy()
                    
                    # Optimize numeric columns
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], downcast='float')
                    
                    if 'volume' in df.columns:
                        df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
                    
                    # Update the dataframe
                    symbol_data.data[timeframe] = df
                    optimization_stats['rows_optimized'] += len(df)
            
            optimization_stats['symbols_optimized'] += 1
        
        # Force garbage collection
        gc.collect()
        
        # Calculate final memory usage
        for symbol_data in self.market_data.values():
            for df in symbol_data.data.values():
                optimization_stats['memory_after_mb'] += df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        optimization_stats['memory_saved_mb'] = optimization_stats['memory_before_mb'] - optimization_stats['memory_after_mb']
        optimization_stats['memory_saved_percent'] = (
            (optimization_stats['memory_saved_mb'] / optimization_stats['memory_before_mb']) * 100 
            if optimization_stats['memory_before_mb'] > 0 else 0
        )
        
        self.logger.info(f"🚀 Phase 3 Memory Optimization: Saved {optimization_stats['memory_saved_mb']:.2f}MB "
                        f"({optimization_stats['memory_saved_percent']:.1f}%) across {optimization_stats['symbols_optimized']} symbols")
        
        return optimization_stats
    
    async def warm_cache(self, symbols: List[str], priority_timeframes: List[str] = None) -> Dict:
        """Phase 3 Caching Optimization - Intelligently warm cache for high-priority data"""
        if priority_timeframes is None:
            priority_timeframes = ['1Min', '5Min', '15Min']  # Most frequently used
        
        warming_stats = {
            'symbols_warmed': 0,
            'timeframes_warmed': 0,
            'cache_hits_before': self.data_stats['cache_hits'],
            'total_time_seconds': 0
        }
        
        start_time = time.time()
        self.logger.info(f"🔥 Phase 3: Warming cache for {len(symbols)} symbols, timeframes: {priority_timeframes}")
        
        # Warm cache in parallel for faster loading
        tasks = []
        for symbol in symbols:
            for timeframe in priority_timeframes:
                task = asyncio.create_task(self._fetch_historical_data(symbol, timeframe))
                tasks.append(task)
        
        # Execute all cache warming tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        warming_stats['symbols_warmed'] = len(symbols)
        warming_stats['timeframes_warmed'] = len(priority_timeframes) * len(symbols)
        warming_stats['total_time_seconds'] = time.time() - start_time
        warming_stats['cache_hits_after'] = self.data_stats['cache_hits']
        warming_stats['cache_improvement'] = warming_stats['cache_hits_after'] - warming_stats['cache_hits_before']
        
        self.logger.info(f"🔥 Cache warmed in {warming_stats['total_time_seconds']:.2f}s - "
                        f"{warming_stats['cache_improvement']} new cache entries")
        
        return warming_stats
    
    def get_cache_recommendations(self) -> Dict:
        """Phase 3 Caching Optimization - Analyze cache usage patterns and provide recommendations"""
        recommendations = {
            'memory_optimizations': [],
            'cache_adjustments': [],
            'performance_tips': []
        }
        
        total_memory = 0
        timeframe_usage = {}
        
        for symbol, symbol_data in self.market_data.items():
            for timeframe, df in symbol_data.data.items():
                df_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
                total_memory += df_memory
                
                if timeframe not in timeframe_usage:
                    timeframe_usage[timeframe] = {'memory_mb': 0, 'symbols': 0, 'avg_rows': 0}
                
                timeframe_usage[timeframe]['memory_mb'] += df_memory
                timeframe_usage[timeframe]['symbols'] += 1
                timeframe_usage[timeframe]['avg_rows'] += len(df)
        
        # Calculate averages
        for tf_data in timeframe_usage.values():
            if tf_data['symbols'] > 0:
                tf_data['avg_rows'] /= tf_data['symbols']
        
        # Generate recommendations
        if total_memory > 100:  # Over 100MB
            recommendations['memory_optimizations'].append("Consider running optimize_memory_usage() to reduce memory footprint")
        
        # Find underutilized timeframes
        for timeframe, usage in timeframe_usage.items():
            cache_size = self.CACHE_SIZES.get(timeframe, self.MAX_CACHE_SIZE)
            if usage['avg_rows'] < cache_size * 0.5:
                recommendations['cache_adjustments'].append(
                    f"Reduce cache size for {timeframe} from {cache_size} to {int(usage['avg_rows'] * 1.2)}"
                )
        
        # Performance tips
        cache_total = self.data_stats['cache_hits'] + self.data_stats['cache_misses']
        if cache_total > 0:
            hit_rate = self.data_stats['cache_hits'] / cache_total
            if hit_rate < 0.8:
                recommendations['performance_tips'].append("Consider warming cache for frequently accessed symbols")
        
        return {
            'total_memory_mb': total_memory,
            'timeframe_usage': timeframe_usage,
            'recommendations': recommendations
        }

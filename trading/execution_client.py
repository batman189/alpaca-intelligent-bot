"""
Advanced Execution Client for Professional Trading Bot
High-speed order execution, slippage management, and position tracking
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import queue
from collections import defaultdict, deque
import json
import uuid

from alpaca_trade_api import REST, Stream
from alpaca_trade_api.entity import Order, Position
from alpaca_trade_api.common import URL

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    PENDING_NEW = "pending_new"
    PENDING_CANCEL = "pending_cancel"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

@dataclass
class TradingOrder:
    """Enhanced order representation"""
    id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: int = 0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parent_strategy: Optional[str] = None
    execution_context: Optional[Dict] = None
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]
    
    @property
    def average_fill_price(self) -> float:
        return self.filled_price if self.filled_quantity > 0 else 0.0

@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    total_orders: int = 0
    filled_orders: int = 0
    average_fill_time: float = 0.0
    average_slippage: float = 0.0
    total_commissions: float = 0.0
    rejected_orders: int = 0
    partial_fills: int = 0
    
    @property
    def fill_rate(self) -> float:
        return self.filled_orders / self.total_orders if self.total_orders > 0 else 0.0

@dataclass
class SlippageAnalysis:
    """Slippage analysis for order execution"""
    expected_price: float
    actual_price: float
    slippage_bps: float
    market_impact: float
    timing_cost: float
    
    @property
    def total_slippage(self) -> float:
        return abs(self.actual_price - self.expected_price) / self.expected_price

class AdvancedExecutionClient:
    """
    Professional-grade execution client with advanced order management
    Optimized for speed, minimal slippage, and intelligent execution
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        
        # Initialize Alpaca API
        self.api = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )
        
        # Order management
        self.active_orders: Dict[str, TradingOrder] = {}
        self.completed_orders: List[TradingOrder] = []
        self.execution_queue = asyncio.Queue()
        self.order_lock = threading.RLock()
        
        # Execution tracking
        self.execution_metrics = ExecutionMetrics()
        self.slippage_history: List[SlippageAnalysis] = []
        self.fill_notifications: List[Callable] = []
        
        # Risk management
        self.max_position_size = 10000  # Maximum position size per symbol
        self.max_daily_loss = 5000      # Maximum daily loss limit
        self.current_daily_pnl = 0      # Track daily P&L
        
        # Performance optimization
        self.order_batching = True      # Batch orders for efficiency
        self.batch_size = 10           # Orders per batch
        self.batch_timeout = 0.1       # Seconds to wait for batch
        self.execution_speeds: deque = deque(maxlen=100)
        
        # Smart routing
        self.routing_preferences = {
            'default': 'SMART',
            'options': 'SMART',
            'penny_stocks': 'ARCA'
        }
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.background_tasks = []
        
        self.logger.info("Advanced Execution Client initialized")
    
    async def start(self):
        """Start the execution client"""
        self.running = True
        
        # Start background tasks
        execution_task = asyncio.create_task(self._execution_worker())
        monitoring_task = asyncio.create_task(self._order_monitoring_worker())
        cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        self.background_tasks.extend([execution_task, monitoring_task, cleanup_task])
        
        self.logger.info("Execution client started")
    
    async def stop(self):
        """Stop the execution client"""
        self.running = False
        
        # Cancel all active orders
        await self._cancel_all_orders()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        self.logger.info("Execution client stopped")
    
    async def place_order(self,
                         symbol: str,
                         side: OrderSide,
                         quantity: int,
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: TimeInForce = TimeInForce.DAY,
                         strategy_id: Optional[str] = None,
                         urgency: str = 'normal') -> Optional[TradingOrder]:
        """
        Place a trading order with advanced execution logic
        
        Args:
            symbol: Stock/option symbol
            side: Buy or sell
            quantity: Number of shares/contracts
            order_type: Type of order
            price: Limit price (if applicable)
            stop_price: Stop price (if applicable)
            time_in_force: Order duration
            strategy_id: Associated strategy
            urgency: Execution urgency (normal, high, critical)
        """
        try:
            # Pre-execution validation
            if not await self._validate_order(symbol, side, quantity, price):
                return None
            
            # Create order object
            order = TradingOrder(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                parent_strategy=strategy_id,
                execution_context={'urgency': urgency, 'created_by': 'execution_client'}
            )
            
            # Add to execution queue
            await self.execution_queue.put(order)
            
            with self.order_lock:
                self.active_orders[order.id] = order
            
            self.logger.info(f"Order queued: {order.id} - {side.value} {quantity} {symbol}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def _validate_order(self,
                             symbol: str,
                             side: OrderSide,
                             quantity: int,
                             price: Optional[float]) -> bool:
        """Validate order before execution"""
        try:
            # Check position limits
            current_position = await self._get_current_position(symbol)
            
            if side == OrderSide.BUY:
                new_position_size = current_position + quantity
            else:
                new_position_size = current_position - quantity
            
            if abs(new_position_size) > self.max_position_size:
                self.logger.warning(f"Order would exceed position limit for {symbol}")
                return False
            
            # Check daily loss limit
            if self.current_daily_pnl <= -self.max_daily_loss:
                self.logger.warning("Daily loss limit reached")
                return False
            
            # Check buying power (simplified)
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            if side == OrderSide.BUY and price:
                required_capital = quantity * price
                if required_capital > buying_power:
                    self.logger.warning(f"Insufficient buying power: {buying_power:.2f} < {required_capital:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False
    
    async def _get_current_position(self, symbol: str) -> int:
        """Get current position size for symbol"""
        try:
            positions = self.api.list_positions()
            for position in positions:
                if position.symbol == symbol:
                    return int(position.qty)
            return 0
            
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return 0
    
    async def _execution_worker(self):
        """Background worker for order execution"""
        while self.running:
            try:
                # Get orders from queue with timeout
                orders_to_execute = []
                
                try:
                    # Get first order (blocking)
                    first_order = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                    orders_to_execute.append(first_order)
                    
                    # Collect additional orders for batching (non-blocking)
                    if self.order_batching:
                        batch_start = time.time()
                        while (len(orders_to_execute) < self.batch_size and 
                               time.time() - batch_start < self.batch_timeout):
                            try:
                                additional_order = await asyncio.wait_for(
                                    self.execution_queue.get(), timeout=0.01
                                )
                                orders_to_execute.append(additional_order)
                            except asyncio.TimeoutError:
                                break
                    
                except asyncio.TimeoutError:
                    continue
                
                # Execute orders
                if orders_to_execute:
                    await self._execute_orders_batch(orders_to_execute)
                
            except Exception as e:
                self.logger.error(f"Error in execution worker: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_orders_batch(self, orders: List[TradingOrder]):
        """Execute a batch of orders efficiently"""
        try:
            execution_start = time.time()
            
            for order in orders:
                await self._execute_single_order(order)
            
            execution_time = time.time() - execution_start
            self.execution_speeds.append(execution_time)
            
            avg_time_per_order = execution_time / len(orders)
            self.logger.debug(f"Executed {len(orders)} orders in {execution_time:.3f}s (avg: {avg_time_per_order:.3f}s)")
            
        except Exception as e:
            self.logger.error(f"Error executing order batch: {e}")
    
    async def _execute_single_order(self, order: TradingOrder):
        """Execute a single order with smart routing and slippage management"""
        try:
            execution_start = time.time()
            
            # Pre-execution market analysis
            market_conditions = await self._analyze_execution_conditions(order)
            
            # Optimize execution based on conditions
            optimized_order = await self._optimize_order_execution(order, market_conditions)
            
            # Submit to Alpaca
            alpaca_order = await self._submit_to_alpaca(optimized_order)
            
            if alpaca_order:
                # Update order status
                with self.order_lock:
                    order.status = OrderStatus.PENDING_NEW
                    order.updated_at = datetime.now()
                
                # Track execution metrics
                execution_time = time.time() - execution_start
                self.execution_metrics.total_orders += 1
                
                self.logger.info(f"Order submitted: {order.id} - Alpaca ID: {alpaca_order.id}")
                
                # Start monitoring this order
                asyncio.create_task(self._monitor_order_fills(order, alpaca_order))
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.id}: {e}")
            with self.order_lock:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                self.execution_metrics.rejected_orders += 1
    
    async def _analyze_execution_conditions(self, order: TradingOrder) -> Dict:
        """Analyze market conditions for optimal execution"""
        try:
            symbol = order.symbol
            
            # Get current quote
            latest_quote = self.api.get_latest_quote(symbol)
            
            # Calculate spread
            bid = float(latest_quote.bid_price) if latest_quote.bid_price else 0
            ask = float(latest_quote.ask_price) if latest_quote.ask_price else 0
            spread = ask - bid
            spread_bps = (spread / ((bid + ask) / 2)) * 10000 if bid > 0 and ask > 0 else 0
            
            # Get recent bars for volatility analysis
            bars = self.api.get_bars(symbol, '1Min', limit=20)
            
            volatility = 0
            if len(bars) > 1:
                returns = []
                for i in range(1, len(bars)):
                    prev_close = float(bars[i-1].c)
                    curr_close = float(bars[i].c)
                    returns.append((curr_close - prev_close) / prev_close)
                
                volatility = np.std(returns) * np.sqrt(252 * 390)  # Annualized intraday vol
            
            # Volume analysis
            avg_volume = np.mean([float(bar.v) for bar in bars]) if bars else 0
            current_volume = float(bars[-1].v) if bars else 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            conditions = {
                'bid': bid,
                'ask': ask,
                'spread': spread,
                'spread_bps': spread_bps,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'market_impact_estimate': self._estimate_market_impact(order, spread, avg_volume)
            }
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing execution conditions: {e}")
            return {}
    
    def _estimate_market_impact(self, order: TradingOrder, spread: float, avg_volume: float) -> float:
        """Estimate market impact of order"""
        try:
            if avg_volume == 0:
                return 0.1  # High impact if no volume data
            
            # Simple market impact model
            order_size_ratio = order.quantity / avg_volume
            
            # Impact increases with order size relative to average volume
            if order_size_ratio < 0.01:  # Less than 1% of average volume
                return 0.05  # 5 bps
            elif order_size_ratio < 0.05:  # Less than 5%
                return 0.1   # 10 bps
            elif order_size_ratio < 0.1:   # Less than 10%
                return 0.2   # 20 bps
            else:
                return 0.5   # 50 bps for large orders
                
        except Exception:
            return 0.1
    
    async def _optimize_order_execution(self, order: TradingOrder, market_conditions: Dict) -> TradingOrder:
        """Optimize order execution based on market conditions"""
        try:
            optimized_order = order
            
            # Smart order type selection
            if order.order_type == OrderType.MARKET:
                # Consider using limit orders in volatile conditions
                if market_conditions.get('spread_bps', 0) > 50:  # Wide spread
                    urgency = order.execution_context.get('urgency', 'normal')
                    
                    if urgency != 'critical':
                        # Convert to aggressive limit order
                        if order.side == OrderSide.BUY:
                            optimized_order.order_type = OrderType.LIMIT
                            optimized_order.price = market_conditions.get('ask', 0) * 1.001  # Slightly above ask
                        else:
                            optimized_order.order_type = OrderType.LIMIT
                            optimized_order.price = market_conditions.get('bid', 0) * 0.999  # Slightly below bid
            
            # Smart routing based on symbol characteristics
            if order.symbol.endswith('.OPT') or 'OPT' in order.symbol:
                # Options routing
                optimized_order.execution_context['route'] = self.routing_preferences['options']
            elif market_conditions.get('bid', 0) < 5.0:
                # Penny stock routing
                optimized_order.execution_context['route'] = self.routing_preferences['penny_stocks']
            else:
                optimized_order.execution_context['route'] = self.routing_preferences['default']
            
            # Time in force optimization
            volatility = market_conditions.get('volatility', 0)
            if volatility > 0.5 and order.time_in_force == TimeInForce.DAY:
                # Use IOC for volatile stocks to avoid adverse selection
                optimized_order.time_in_force = TimeInForce.IOC
            
            return optimized_order
            
        except Exception as e:
            self.logger.error(f"Error optimizing order execution: {e}")
            return order
    
    async def _submit_to_alpaca(self, order: TradingOrder) -> Optional[Order]:
        """Submit order to Alpaca with error handling"""
        try:
            # Prepare order parameters
            order_params = {
                'symbol': order.symbol,
                'qty': order.quantity,
                'side': order.side.value,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force.value
            }
            
            # Add price parameters if needed
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order.price:
                    order_params['limit_price'] = order.price
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order.stop_price:
                    order_params['stop_price'] = order.stop_price
            
            # Submit order
            alpaca_order = self.api.submit_order(**order_params)
            
            return alpaca_order
            
        except Exception as e:
            self.logger.error(f"Error submitting order to Alpaca: {e}")
            return None
    
    async def _monitor_order_fills(self, order: TradingOrder, alpaca_order: Order):
        """Monitor order for fills and updates"""
        try:
            while not order.is_complete and self.running:
                try:
                    # Get latest order status
                    updated_alpaca_order = self.api.get_order(alpaca_order.id)
                    
                    # Update our order object
                    with self.order_lock:
                        old_status = order.status
                        order.status = OrderStatus(updated_alpaca_order.status)
                        order.filled_quantity = int(updated_alpaca_order.filled_qty or 0)
                        order.filled_price = float(updated_alpaca_order.filled_avg_price or 0)
                        order.updated_at = datetime.now()
                    
                    # Handle status changes
                    if order.status != old_status:
                        await self._handle_order_status_change(order, old_status)
                    
                    # Check if order is complete
                    if order.is_complete:
                        await self._handle_order_completion(order)
                        break
                    
                    await asyncio.sleep(0.5)  # Check every 500ms
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring order {order.id}: {e}")
                    await asyncio.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Error in order monitoring: {e}")
    
    async def _handle_order_status_change(self, order: TradingOrder, old_status: OrderStatus):
        """Handle order status changes"""
        try:
            self.logger.info(f"Order {order.id} status changed: {old_status.value} -> {order.status.value}")
            
            if order.status == OrderStatus.PARTIALLY_FILLED:
                self.execution_metrics.partial_fills += 1
                
                # Notify fill handlers
                for callback in self.fill_notifications:
                    try:
                        callback(order, 'partial_fill')
                    except Exception as e:
                        self.logger.error(f"Error in fill notification callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling status change: {e}")
    
    async def _handle_order_completion(self, order: TradingOrder):
        """Handle completed orders"""
        try:
            with self.order_lock:
                # Move from active to completed
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                self.completed_orders.append(order)
                
                # Keep only recent completed orders
                if len(self.completed_orders) > 1000:
                    self.completed_orders = self.completed_orders[-1000:]
            
            # Update metrics
            if order.status == OrderStatus.FILLED:
                self.execution_metrics.filled_orders += 1
                
                # Calculate slippage if we have execution context
                if (order.execution_context and 
                    'expected_price' in order.execution_context and 
                    order.filled_quantity > 0):
                    
                    slippage_analysis = SlippageAnalysis(
                        expected_price=order.execution_context['expected_price'],
                        actual_price=order.filled_price,
                        slippage_bps=((order.filled_price - order.execution_context['expected_price']) / 
                                     order.execution_context['expected_price']) * 10000,
                        market_impact=order.execution_context.get('market_impact', 0),
                        timing_cost=0  # Would calculate based on time between decision and execution
                    )
                    
                    self.slippage_history.append(slippage_analysis)
                    
                    # Update average slippage
                    if len(self.slippage_history) > 0:
                        self.execution_metrics.average_slippage = np.mean([
                            abs(s.slippage_bps) for s in self.slippage_history[-100:]
                        ])
            
            elif order.status == OrderStatus.REJECTED:
                self.execution_metrics.rejected_orders += 1
            
            # Notify completion
            for callback in self.fill_notifications:
                try:
                    callback(order, 'completed')
                except Exception as e:
                    self.logger.error(f"Error in completion notification: {e}")
            
            self.logger.info(f"Order completed: {order.id} - Status: {order.status.value}")
            
        except Exception as e:
            self.logger.error(f"Error handling order completion: {e}")
    
    async def _order_monitoring_worker(self):
        """Background worker for order monitoring and management"""
        while self.running:
            try:
                current_time = datetime.now()
                orders_to_cancel = []
                
                with self.order_lock:
                    for order in self.active_orders.values():
                        # Cancel stale orders
                        if (current_time - order.created_at).total_seconds() > 3600:  # 1 hour
                            if order.status in [OrderStatus.NEW, OrderStatus.PENDING_NEW]:
                                orders_to_cancel.append(order.id)
                        
                        # Cancel orders that are not progressing
                        elif (current_time - order.updated_at).total_seconds() > 300:  # 5 minutes
                            if order.status == OrderStatus.PENDING_NEW:
                                orders_to_cancel.append(order.id)
                
                # Cancel identified orders
                for order_id in orders_to_cancel:
                    await self.cancel_order(order_id, reason="stale_order")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order monitoring worker: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        while self.running:
            try:
                # Clean up old slippage data
                if len(self.slippage_history) > 1000:
                    self.slippage_history = self.slippage_history[-1000:]
                
                # Update daily P&L
                try:
                    account = self.api.get_account()
                    self.current_daily_pnl = float(account.unrealized_pl) + float(account.realized_pl)
                except Exception:
                    pass
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(300)
    
    async def cancel_order(self, order_id: str, reason: str = "user_request") -> bool:
        """Cancel an active order"""
        try:
            with self.order_lock:
                if order_id not in self.active_orders:
                    self.logger.warning(f"Order {order_id} not found in active orders")
                    return False
                
                order = self.active_orders[order_id]
                
                if order.is_complete:
                    self.logger.warning(f"Order {order_id} is already complete")
                    return False
            
            # Find Alpaca order ID (simplified - would need better tracking)
            alpaca_orders = self.api.list_orders(status='all', limit=100)
            alpaca_order_id = None
            
            for alpaca_order in alpaca_orders:
                # Match by symbol, quantity, and timestamp (simplified matching)
                if (alpaca_order.symbol == order.symbol and 
                    int(alpaca_order.qty) == order.quantity):
                    alpaca_order_id = alpaca_order.id
                    break
            
            if alpaca_order_id:
                # Cancel with Alpaca
                self.api.cancel_order(alpaca_order_id)
                
                with self.order_lock:
                    order.status = OrderStatus.CANCELED
                    order.updated_at = datetime.now()
                    order.execution_context['cancel_reason'] = reason
                
                self.logger.info(f"Order canceled: {order_id} - Reason: {reason}")
                return True
            else:
                self.logger.error(f"Could not find Alpaca order for {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def _cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            with self.order_lock:
                order_ids = list(self.active_orders.keys())
            
            for order_id in order_ids:
                await self.cancel_order(order_id, reason="shutdown")
            
            # Also cancel any pending Alpaca orders
            self.api.cancel_all_orders()
            
            self.logger.info("All orders canceled")
            
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
    
    def add_fill_notification(self, callback: Callable):
        """Add callback for order fill notifications"""
        self.fill_notifications.append(callback)
    
    def get_active_orders(self) -> List[TradingOrder]:
        """Get list of active orders"""
        with self.order_lock:
            return list(self.active_orders.values())
    
    def get_order_by_id(self, order_id: str) -> Optional[TradingOrder]:
        """Get order by ID"""
        with self.order_lock:
            return self.active_orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[TradingOrder]:
        """Get all orders for a symbol"""
        with self.order_lock:
            return [order for order in self.active_orders.values() if order.symbol == symbol]
    
    def get_execution_metrics(self) -> ExecutionMetrics:
        """Get execution performance metrics"""
        # Update average fill time
        if len(self.execution_speeds) > 0:
            self.execution_metrics.average_fill_time = np.mean(list(self.execution_speeds))
        
        return self.execution_metrics
    
    def get_slippage_analysis(self) -> Dict:
        """Get slippage analysis summary"""
        if not self.slippage_history:
            return {}
        
        recent_slippage = self.slippage_history[-100:]  # Last 100 trades
        
        return {
            'average_slippage_bps': np.mean([abs(s.slippage_bps) for s in recent_slippage]),
            'median_slippage_bps': np.median([abs(s.slippage_bps) for s in recent_slippage]),
            'max_slippage_bps': max([abs(s.slippage_bps) for s in recent_slippage]),
            'slippage_std': np.std([s.slippage_bps for s in recent_slippage]),
            'total_trades': len(recent_slippage)
        }
    
    async def place_options_order(self,
                                 option_symbol: str,
                                 side: OrderSide,
                                 quantity: int,
                                 order_type: OrderType = OrderType.LIMIT,
                                 price: Optional[float] = None,
                                 strategy_id: Optional[str] = None) -> Optional[TradingOrder]:
        """
        Place options order with specialized handling
        """
        try:
            # Options-specific validation
            if not self._validate_options_order(option_symbol, quantity, price):
                return None
            
            # Options typically require limit orders
            if order_type == OrderType.MARKET:
                order_type = OrderType.LIMIT
                # Get current bid/ask for options
                quote = await self._get_options_quote(option_symbol)
                if quote and side == OrderSide.BUY:
                    price = quote.get('ask', price)
                elif quote and side == OrderSide.SELL:
                    price = quote.get('bid', price)
            
            return await self.place_order(
                symbol=option_symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                time_in_force=TimeInForce.DAY,  # Options typically use DAY orders
                strategy_id=strategy_id,
                urgency='normal'
            )
            
        except Exception as e:
            self.logger.error(f"Error placing options order: {e}")
            return None
    
    def _validate_options_order(self, option_symbol: str, quantity: int, price: Optional[float]) -> bool:
        """Validate options-specific order parameters"""
        try:
            # Basic options validation
            if quantity <= 0 or quantity > 100:  # Reasonable contract limits
                self.logger.warning(f"Invalid options quantity: {quantity}")
                return False
            
            if price and (price <= 0 or price > 1000):  # Reasonable price limits
                self.logger.warning(f"Invalid options price: {price}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating options order: {e}")
            return False
    
    async def _get_options_quote(self, option_symbol: str) -> Optional[Dict]:
        """Get real options quote from Alpaca API"""
        try:
            import requests
            
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.secret_key
            }
            
            if not self.api_key or not self.secret_key:
                self.logger.error("❌ API credentials not available for options quote")
                return None
            
            url = f"https://data.alpaca.markets/v1beta1/options/snapshots"
            params = {'symbols': option_symbol}
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'snapshots' in data and option_symbol in data['snapshots']:
                    contract_data = data['snapshots'][option_symbol]
                    
                    latest_quote = contract_data.get('latestQuote', {})
                    latest_trade = contract_data.get('latestTrade', {})
                    
                    self.logger.info(f"✅ Real options quote fetched for {option_symbol}")
                    
                    return {
                        'bid': latest_quote.get('bidPrice', 0.0),
                        'ask': latest_quote.get('askPrice', 0.0),
                        'last_price': latest_trade.get('price', 0.0),
                        'volume': latest_trade.get('size', 0),
                        'timestamp': latest_quote.get('timestamp') or latest_trade.get('timestamp')
                    }
            elif response.status_code == 404:
                self.logger.warning(f"⚠️ Option symbol {option_symbol} not found")
            else:
                self.logger.error(f"❌ Options quote API error: {response.status_code}")
                
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting real options quote for {option_symbol}: {e}")
            return None
    
    async def place_bracket_order(self,
                                 symbol: str,
                                 side: OrderSide,
                                 quantity: int,
                                 entry_price: float,
                                 take_profit: float,
                                 stop_loss: float,
                                 strategy_id: Optional[str] = None) -> List[TradingOrder]:
        """
        Place bracket order (entry + take profit + stop loss)
        """
        try:
            orders = []
            
            # Entry order
            entry_order = await self.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=entry_price,
                strategy_id=strategy_id
            )
            
            if entry_order:
                orders.append(entry_order)
                
                # Exit side (opposite of entry)
                exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
                
                # Take profit order
                tp_order = await self.place_order(
                    symbol=symbol,
                    side=exit_side,
                    quantity=quantity,
                    order_type=OrderType.LIMIT,
                    price=take_profit,
                    strategy_id=strategy_id
                )
                
                if tp_order:
                    orders.append(tp_order)
                
                # Stop loss order
                sl_order = await self.place_order(
                    symbol=symbol,
                    side=exit_side,
                    quantity=quantity,
                    order_type=OrderType.STOP,
                    stop_price=stop_loss,
                    strategy_id=strategy_id
                )
                
                if sl_order:
                    orders.append(sl_order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error placing bracket order: {e}")
            return []
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive execution performance summary"""
        try:
            metrics = self.get_execution_metrics()
            slippage = self.get_slippage_analysis()
            
            with self.order_lock:
                active_count = len(self.active_orders)
                completed_count = len(self.completed_orders)
            
            return {
                'execution_metrics': {
                    'total_orders': metrics.total_orders,
                    'fill_rate': metrics.fill_rate,
                    'average_fill_time': metrics.average_fill_time,
                    'rejected_orders': metrics.rejected_orders
                },
                'slippage_analysis': slippage,
                'order_counts': {
                    'active': active_count,
                    'completed': completed_count
                },
                'daily_pnl': self.current_daily_pnl,
                'system_status': 'running' if self.running else 'stopped'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}

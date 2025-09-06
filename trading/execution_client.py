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
            positions = self.api.list_

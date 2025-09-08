import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class TradeTracker:
    def __init__(self):
        self.open_trades = {}  # key: symbol, value: list of trade IDs or entry data

    def open_trade(self, symbol, trade_id, entry_price, contracts, option_type, strike):
        """Record when a trade is opened"""
        try:
            if symbol not in self.open_trades:
                self.open_trades[symbol] = []
            self.open_trades[symbol].append({
                'trade_id': trade_id,
                'entry_price': entry_price,
                'contracts': contracts,
                'option_type': option_type,
                'strike': strike,
                'entry_time': datetime.now().isoformat()
            })
            logger.info(f"Tracked open trade: {symbol} - ID: {trade_id}")
        except Exception as e:
            logger.error(f"Error tracking open trade: {e}")

    def close_trade(self, symbol, exit_price, trade_id=None):
        """Record when a trade is closed and calculate P&L"""
        try:
            if symbol not in self.open_trades or not self.open_trades[symbol]:
                logger.warning(f"No open trades found for {symbol}")
                return None

            # If no trade_id is provided, close the most recent trade
            if trade_id is None:
                trade = self.open_trades[symbol].pop()
            else:
                trade = next((t for t in self.open_trades[symbol] if t['trade_id'] == trade_id), None)
                if trade:
                    self.open_trades[symbol].remove(trade)

            if trade:
                # Calculate P&L for the option trade
                pnl_per_contract = exit_price - trade['entry_price']
                if trade['option_type'] == 'put':
                    pnl_per_contract = trade['entry_price'] - exit_price
                total_pnl = pnl_per_contract * trade['contracts'] * 100  # Options are 100 shares per contract
                logger.info(f"Closed trade {trade.get('trade_id', 'unknown')} - P&L: ${total_pnl:.2f}")
                return total_pnl
            return None
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return None

    def get_open_trades(self, symbol=None):
        """Get all open trades or for a specific symbol"""
        if symbol:
            return self.open_trades.get(symbol, [])
        return self.open_trades

class ReinforcementLearner:
    def __init__(self):
        self.trade_history_file = 'data/trade_history.json'
        self.performance_metrics = {}
        self.learning_rate = 0.1
        self.tracker = TradeTracker()  # Initialize the trade tracker
        
    def record_trade(self, trade_data: Dict):
        """Record a completed trade for learning"""
        try:
            os.makedirs('data', exist_ok=True)
            
            history = self.load_trade_history()
            
            trade_data['recorded_at'] = datetime.now().isoformat()
            history.append(trade_data)
            
            with open(self.trade_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"Trade recorded: {trade_data['symbol']} - P&L: ${trade_data.get('pnl', 0):.2f}")
            
            self.update_performance_metrics(trade_data)
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def load_trade_history(self) -> List:
        """Load trade history from file"""
        try:
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            return []
    
    def update_performance_metrics(self, trade_data: Dict):
        """Update performance metrics based on trade outcome"""
        symbol = trade_data['symbol']
        pnl = trade_data.get('pnl', 0)
        confidence = trade_data.get('confidence', 0.5)
        
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_pnl': 0,
                'avg_confidence': 0,
                'success_rate': 0.5
            }
        
        metrics = self.performance_metrics[symbol]
        metrics['total_trades'] += 1
        metrics['total_pnl'] += pnl
        metrics['avg_confidence'] = (metrics['avg_confidence'] * (metrics['total_trades'] - 1) + confidence) / metrics['total_trades']
        
        if pnl > 0:
            metrics['profitable_trades'] += 1
        
        metrics['success_rate'] = metrics['profitable_trades'] / metrics['total_trades']
        
        logger.info(f"Updated metrics for {symbol}: Success rate: {metrics['success_rate']:.2f}, Avg P&L: ${metrics['total_pnl']/metrics['total_trades']:.2f}")
    
    def adjust_confidence(self, symbol: str, base_confidence: float) -> float:
        """Adjust confidence based on historical performance"""
        if symbol in self.performance_metrics:
            metrics = self.performance_metrics[symbol]
            if metrics['success_rate'] > 0.6:
                adjusted_confidence = min(0.95, base_confidence * (1 + self.learning_rate))
                logger.info(f"Boosting confidence for {symbol} from {base_confidence:.2f} to {adjusted_confidence:.2f}")
                return adjusted_confidence
            elif metrics['success_rate'] < 0.4:
                adjusted_confidence = max(0.1, base_confidence * (1 - self.learning_rate))
                logger.info(f"Reducing confidence for {symbol} from {base_confidence:.2f} to {adjusted_confidence:.2f}")
                return adjusted_confidence
        
        return base_confidence
    
    def learn_from_trade(self, symbol: str, prediction: int, actual_outcome: int, confidence: float, pnl: float):
        """Main learning function that updates the model"""
        trade_data = {
            'symbol': symbol,
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'confidence': confidence,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat()
        }
        
        self.record_trade(trade_data)
        logger.info(f"Learning from trade: {symbol} - Prediction: {prediction}, Actual: {actual_outcome}, P&L: ${pnl:.2f}")
    
    def get_symbol_performance(self, symbol: str) -> Dict:
        """Get performance metrics for a specific symbol"""
        return self.performance_metrics.get(symbol, {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0,
            'avg_confidence': 0.5,
            'success_rate': 0.5
        })
    
    def should_trade_symbol(self, symbol: str, confidence: float) -> bool:
        """Determine if we should trade a symbol based on historical performance"""
        metrics = self.get_symbol_performance(symbol)
        
        if metrics['total_trades'] > 5 and metrics['success_rate'] < 0.3:
            logger.info(f"Avoiding {symbol} due to poor historical performance (success rate: {metrics['success_rate']:.2f})")
            return False
        
        if metrics['success_rate'] < 0.5 and confidence < 0.7:
            return False
            
        return True

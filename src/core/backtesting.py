import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm

class MarketState:
    def __init__(self, timeframes: List[str] = ['1h', '4h', '1d']):
        self.timeframes = timeframes
        self.data = {tf: None for tf in timeframes}
        self.sentiment = None
        self.regime = 'normal'
        self.volatility = 0.0
        self.risk_adjustment = 1.0

class AdvancedStrategy(bt.Strategy):
    params = (
        ('base_risk_percentage', 0.02),  # Base risk per trade
        ('max_positions', 5),            # Maximum concurrent positions
        ('stop_loss_atr', 2),           # Stop loss in ATR units
        ('take_profit_atr', 4),          # Take profit in ATR units
        ('atr_period', 14),              # ATR calculation period
        ('rsi_period', 14),              # RSI period
        ('ma_period', 20),               # Moving average period
        ('kelly_fraction', 0.5),         # Kelly criterion fraction
        ('timeframes', ['1h', '4h', '1d']),  # Multiple timeframes
    )

    def __init__(self):
        self.market_state = MarketState(self.p.timeframes)
        
        # Initialize indicators for each timeframe
        self.indicators = {tf: {} for tf in self.p.timeframes}
        for tf in self.p.timeframes:
            d = getattr(self.datas[0], tf, self.datas[0])  # Use main timeframe if specific not available
            self.indicators[tf] = {
                'atr': bt.indicators.ATR(d, period=self.p.atr_period),
                'rsi': bt.indicators.RSI(d, period=self.p.rsi_period),
                'sma': bt.indicators.SMA(d, period=self.p.ma_period),
                'macd': bt.indicators.MACD(d),
                'bbands': bt.indicators.BollingerBands(d, period=20),
                'volume_ma': bt.indicators.SMA(d.volume, period=20)
            }
        
        # Position management
        self.orders = {}  # Track open orders
        self.positions_info = defaultdict(dict)  # Track position info
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        self.position_sizes = []
        self.win_rate_history = []
        
        logging.info("Strategy initialized with parameters: %s", self.p.__dict__)

    def calculate_position_size(self, price: float, stop_loss: float, sentiment_risk: float = 1.0) -> int:
        """
        Calculate position size using Kelly Criterion and sentiment adjustment
        """
        account_value = self.broker.getvalue()
        
        # Calculate win rate and average win/loss ratio
        if self.trades:
            wins = [t for t in self.trades if t['pnl'] > 0]
            win_rate = len(wins) / len(self.trades)
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            losses = [t for t in self.trades if t['pnl'] <= 0]
            avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0
            
            if avg_loss > 0:
                # Kelly formula
                kelly_fraction = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                kelly_fraction *= self.p.kelly_fraction  # Use fractional Kelly
                kelly_fraction = max(0, min(kelly_fraction, self.p.base_risk_percentage))
            else:
                kelly_fraction = self.p.base_risk_percentage
        else:
            kelly_fraction = self.p.base_risk_percentage
        
        # Adjust position size based on sentiment and market regime
        risk_amount = account_value * kelly_fraction * sentiment_risk * self.market_state.risk_adjustment
        position_size = risk_amount / (price - stop_loss)
        
        return int(position_size)

    def update_market_state(self) -> None:
        """Update market state with multi-timeframe analysis"""
        # Update volatility
        returns = pd.Series([self.data.close[i] for i in range(-20, 0)]).pct_change()
        self.market_state.volatility = returns.std()
        
        # Calculate trend alignment
        trends = []
        for tf in self.p.timeframes:
            ind = self.indicators[tf]
            trend = 1 if ind['sma'][0] > ind['sma'][-1] else -1
            trends.append(trend)
        
        trend_alignment = sum(1 for t in trends if t == trends[0]) / len(trends)
        
        # Update risk adjustment based on market conditions
        volatility_factor = 0.8 if self.market_state.volatility > 0.02 else 1.2
        trend_factor = 1.0 + (trend_alignment - 0.5)
        self.market_state.risk_adjustment = volatility_factor * trend_factor

    def should_long(self) -> Tuple[bool, float]:
        """Enhanced entry conditions for long positions"""
        confidence = 0.0
        signals = []
        
        for tf in self.p.timeframes:
            ind = self.indicators[tf]
            
            # Technical signals
            trend_signal = ind['sma'][0] > ind['sma'][-1]
            momentum_signal = ind['rsi'][0] < 30
            volatility_signal = self.data.close[0] < ind['bbands'].lines.bot[0]
            volume_signal = self.data.volume[0] > ind['volume_ma'][0]
            
            # Weight signals by timeframe
            weight = 1.0 if tf == '1h' else 0.8 if tf == '4h' else 0.6
            tf_confidence = sum([
                trend_signal,
                momentum_signal,
                volatility_signal,
                volume_signal
            ]) * weight / 4
            
            signals.append(tf_confidence)
        
        # Combine signals across timeframes
        confidence = np.mean(signals)
        return confidence > 0.6, confidence

    def should_short(self) -> Tuple[bool, float]:
        """Enhanced entry conditions for short positions"""
        confidence = 0.0
        signals = []
        
        for tf in self.p.timeframes:
            ind = self.indicators[tf]
            
            # Technical signals
            trend_signal = ind['sma'][0] < ind['sma'][-1]
            momentum_signal = ind['rsi'][0] > 70
            volatility_signal = self.data.close[0] > ind['bbands'].lines.top[0]
            volume_signal = self.data.volume[0] > ind['volume_ma'][0]
            
            # Weight signals by timeframe
            weight = 1.0 if tf == '1h' else 0.8 if tf == '4h' else 0.6
            tf_confidence = sum([
                trend_signal,
                momentum_signal,
                volatility_signal,
                volume_signal
            ]) * weight / 4
            
            signals.append(tf_confidence)
        
        # Combine signals across timeframes
        confidence = np.mean(signals)
        return confidence > 0.6, confidence

    def next(self):
        """Main strategy logic"""
        # Update market state and tracking metrics
        self.update_market_state()
        self.equity_curve.append(self.broker.getvalue())
        current_drawdown = (self.broker.getvalue() - max(self.equity_curve)) / max(self.equity_curve)
        self.drawdowns.append(current_drawdown)
        
        # Check existing positions
        for data in self.datas:
            position = self.getposition(data)
            if position:
                self.manage_position(data, position)
                continue
            
            # Check for new entry if we're under max positions
            if len(self.positions) >= self.p.max_positions:
                continue
                
            self.check_entry_signals(data)

    def manage_position(self, data, position):
        """Enhanced position management"""
        pos_info = self.positions_info[data._name]
        
        # Update trailing stop with dynamic ATR multiplier
        atr_value = self.indicators['1h']['atr'][0]
        if position.size > 0:  # Long position
            dynamic_atr = self.p.stop_loss_atr * (1 + abs(position.pnl) / position.price)
            new_stop = data.close[0] - atr_value * dynamic_atr
            pos_info['trailing_stop'] = max(pos_info['trailing_stop'], new_stop)
        else:  # Short position
            dynamic_atr = self.p.stop_loss_atr * (1 + abs(position.pnl) / position.price)
            new_stop = data.close[0] + atr_value * dynamic_atr
            pos_info['trailing_stop'] = min(pos_info['trailing_stop'], new_stop)
        
        # Check exit conditions
        self.check_exit_signals(data, position, pos_info)

    def check_entry_signals(self, data):
        """Check and execute entry signals with position sizing"""
        should_long, long_confidence = self.should_long()
        should_short, short_confidence = self.should_short()
        
        if should_long:
            atr_value = self.indicators['1h']['atr'][0]
            stop_loss = data.close[0] - atr_value * self.p.stop_loss_atr
            take_profit = data.close[0] + atr_value * self.p.take_profit_atr
            
            # Adjust position size based on signal confidence
            sentiment_risk = 1.0 + (long_confidence - 0.6)
            size = self.calculate_position_size(data.close[0], stop_loss, sentiment_risk)
            
            self.positions_info[data._name] = {
                'entry_price': data.close[0],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': stop_loss,
                'confidence': long_confidence
            }
            
            self.buy(data=data, size=size)
            self.position_sizes.append(size)
            logging.info(f"Long position opened: Price={data.close[0]}, Size={size}, Confidence={long_confidence:.2f}")
            
        elif should_short:
            atr_value = self.indicators['1h']['atr'][0]
            stop_loss = data.close[0] + atr_value * self.p.stop_loss_atr
            take_profit = data.close[0] - atr_value * self.p.take_profit_atr
            
            # Adjust position size based on signal confidence
            sentiment_risk = 1.0 + (short_confidence - 0.6)
            size = self.calculate_position_size(data.close[0], stop_loss, sentiment_risk)
            
            self.positions_info[data._name] = {
                'entry_price': data.close[0],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': stop_loss,
                'confidence': short_confidence
            }
            
            self.sell(data=data, size=size)
            self.position_sizes.append(size)
            logging.info(f"Short position opened: Price={data.close[0]}, Size={size}, Confidence={short_confidence:.2f}")

    def check_exit_signals(self, data, position, pos_info):
        """Enhanced exit signal checking"""
        # Calculate dynamic take profit based on volatility
        atr_value = self.indicators['1h']['atr'][0]
        dynamic_tp_multiplier = self.p.take_profit_atr * (1 + self.market_state.volatility * 10)
        
        if position.size > 0:  # Long position
            dynamic_take_profit = pos_info['entry_price'] + atr_value * dynamic_tp_multiplier
            if (data.close[0] <= pos_info['trailing_stop'] or
                data.close[0] >= dynamic_take_profit):
                self.close(data=data)
                self.record_trade(data, position, pos_info)
                
        else:  # Short position
            dynamic_take_profit = pos_info['entry_price'] - atr_value * dynamic_tp_multiplier
            if (data.close[0] >= pos_info['trailing_stop'] or
                data.close[0] <= dynamic_take_profit):
                self.close(data=data)
                self.record_trade(data, position, pos_info)

    def record_trade(self, data, position, pos_info):
        """Enhanced trade recording"""
        trade_info = {
            'symbol': data._name,
            'entry_date': position.dtopen,
            'exit_date': self.data.datetime.datetime(),
            'entry_price': pos_info['entry_price'],
            'exit_price': data.close[0],
            'size': position.size,
            'pnl': position.pnl,
            'pnl_pct': position.pnlcomm / pos_info['entry_price'] * 100,
            'confidence': pos_info.get('confidence', 0),
            'market_regime': self.market_state.regime,
            'volatility': self.market_state.volatility
        }
        self.trades.append(trade_info)
        
        # Update win rate history
        if len(self.trades) >= 10:
            recent_trades = self.trades[-10:]
            win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)
            self.win_rate_history.append(win_rate)
        
        logging.info(f"Trade closed: {trade_info}")

def run_backtest(data: Dict[str, pd.DataFrame], **kwargs):
    """Run backtest with multi-timeframe data and advanced analytics"""
    # Initialize Cerebro
    cerebro = bt.Cerebro()
    
    # Add strategy
    strategy_params = {
        'base_risk_percentage': kwargs.get('risk_percentage', 0.02),
        'max_positions': kwargs.get('max_positions', 5),
        'stop_loss_atr': kwargs.get('stop_loss_atr', 2),
        'take_profit_atr': kwargs.get('take_profit_atr', 4),
        'kelly_fraction': kwargs.get('kelly_fraction', 0.5),
        'timeframes': list(data.keys())
    }
    cerebro.addstrategy(AdvancedStrategy, **strategy_params)
    
    # Add data for each timeframe
    for tf, df in data.items():
        datafeed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(datafeed, name=tf)
    
    # Set broker parameters
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    # Run backtest
    logging.info("Starting backtest...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    strategy = results[0]
    
    # Calculate comprehensive metrics
    sharpe_ratio = strategy.analyzers.sharpe.get_analysis()['sharperatio']
    max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
    total_return = (final_value - initial_value) / initial_value * 100
    vwr = strategy.analyzers.vwr.get_analysis()['vwr']
    sqn = strategy.analyzers.sqn.get_analysis()['sqn']
    trade_analysis = strategy.analyzers.trades.get_analysis()
    
    # Calculate additional metrics
    if strategy.trades:
        avg_trade_duration = np.mean([(t['exit_date'] - t['entry_date']).total_seconds() / 3600 for t in strategy.trades])
        profit_factor = abs(sum(t['pnl'] for t in strategy.trades if t['pnl'] > 0) / 
                          sum(t['pnl'] for t in strategy.trades if t['pnl'] < 0)) if any(t['pnl'] < 0 for t in strategy.trades) else float('inf')
    else:
        avg_trade_duration = 0
        profit_factor = 0
    
    # Print detailed results
    print("\n=== Backtest Results ===")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Variability-Weighted Return: {vwr:.2f}")
    print(f"System Quality Number: {sqn:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Trade Duration: {avg_trade_duration:.1f} hours")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Equity curve
    ax1.plot(strategy.equity_curve)
    ax1.set_title('Equity Curve')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True)
    
    # Drawdown
    ax2.plot(strategy.drawdowns)
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True)
    
    # Win rate history
    if strategy.win_rate_history:
        ax3.plot(strategy.win_rate_history)
        ax3.set_title('Rolling Win Rate (10 trades)')
        ax3.set_xlabel('Trades')
        ax3.set_ylabel('Win Rate')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    
    # Save detailed trade log
    trades_df = pd.DataFrame(strategy.trades)
    trades_df.to_csv('trade_log.csv', index=False)
    
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'vwr': vwr,
        'sqn': sqn,
        'profit_factor': profit_factor,
        'avg_trade_duration': avg_trade_duration,
        'trade_analysis': trade_analysis,
        'equity_curve': strategy.equity_curve,
        'drawdowns': strategy.drawdowns,
        'win_rate_history': strategy.win_rate_history,
        'trades': strategy.trades,
        'position_sizes': strategy.position_sizes
    }

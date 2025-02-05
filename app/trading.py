import yfinance as yf
import schedule
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

from app.models.price_prediction import PricePredictionModel
from app.models.risk_management import RiskManagementModel
from app.models.tp_sl_management import TpSlManagementModel
from app.models.indicator_management import IndicatorManagementModel
from app.models.rl_trading import train_rl_agent, evaluate_rl_model, create_trading_env
from app.models.sentiment_analysis import SentimentAnalyzer
from app.models.backtesting import run_backtest
from app.telegram_bot import TelegramBot
from app.model_trainer import ModelTrainer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.symbols = symbols
        self.start_date = start_date or "2010-01-01"
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data = {}
        self.initialize_data()
        logging.info(f"DataManager initialized with symbols: {symbols}")

    def initialize_data(self):
        """Initialize data for all symbols"""
        for symbol in self.symbols:
            try:
                self.data[symbol] = self.get_initial_data(symbol)
                logging.info(f"Successfully loaded data for {symbol}")
            except Exception as e:
                logging.error(f"Failed to load data for {symbol}: {e}")

    def get_initial_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data for a symbol"""
        logging.info(f"Fetching initial data for symbol: {symbol}")
        data = yf.download(symbol, start=self.start_date, end=self.end_date)
        
        if data.empty:
            raise ValueError(f"No data received for {symbol}")
            
        # Add technical indicators
        self._add_technical_indicators(data)
        return data

    def _add_technical_indicators(self, data: pd.DataFrame):
        """Add technical indicators to the dataframe"""
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Volatility indicators
        data['ATR'] = self._calculate_atr(data)
        data['Bollinger_Upper'], data['Bollinger_Lower'] = self._calculate_bollinger_bands(data)
        
        # Momentum indicators
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'], data['Signal_Line'] = self._calculate_macd(data['Close'])
        
        data.fillna(method='bfill', inplace=True)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = data['Close'].rolling(window=period).mean()
        std_dev = data['Close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD and Signal Line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def update_data(self):
        """Update data for all symbols"""
        logging.info("Updating data for all symbols.")
        for symbol in self.symbols:
            try:
                logging.info(f"Updating data for symbol: {symbol}")
                new_data = yf.download(symbol, period='1d', interval='1m')
                if not new_data.empty:
                    self._add_technical_indicators(new_data)
                    self.data[symbol] = pd.concat([self.data[symbol], new_data]).drop_duplicates()
                    logging.info(f"Successfully updated data for {symbol}")
                else:
                    logging.warning(f"No new data received for {symbol}")
            except Exception as e:
                logging.error(f"Error updating data for {symbol}: {e}")
        logging.info("Data update completed.")

    def start_data_update(self, interval_minutes: int = 5):
        """Start periodic data updates"""
        logging.info(f"Starting data update every {interval_minutes} minutes.")
        schedule.every(interval_minutes).minutes.do(self.update_data)
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in data update loop: {e}")
                time.sleep(60)  # Wait before retrying

class TradingBot:
    def __init__(self, initial_balance: float = 100000):
        self.price_model = PricePredictionModel()
        self.risk_model = RiskManagementModel()
        self.tp_sl_model = TpSlManagementModel()
        self.indicator_model = IndicatorManagementModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.rl_model = None
        self.telegram_bot = TelegramBot()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.latest_signals = []
        logging.info("TradingBot initialized with models.")

    def get_latest_signals(self) -> List[Dict]:
        """Returns the latest trading signals"""
        return self.latest_signals

    def train_all_models(self, data_manager: DataManager):
        """Train all models with available data"""
        logging.info("Training models for all symbols.")
        try:
            for symbol, data in data_manager.data.items():
                self.train_single_model(data, symbol)
            
            complete_message = "Training for all models has completed."
            self.telegram_bot.send_message(complete_message)
            logging.info(complete_message)
        except Exception as e:
            error_message = f"Error during model training: {e}"
            self.telegram_bot.send_message(error_message)
            logging.error(error_message)

    def train_single_model(self, data: pd.DataFrame, symbol: str):
        """Train models for a single symbol"""
        try:
            logging.info(f"Training model for {symbol}...")
            
            # Train price prediction model
            self.price_model.train(data, symbol)
            logging.info(f"Price prediction model trained for {symbol}")
            
            # Train risk management model
            self.risk_model.train(data, symbol)
            logging.info(f"Risk management model trained for {symbol}")
            
            # Train TP/SL model
            self.tp_sl_model.train(data, symbol)
            logging.info(f"TP/SL model trained for {symbol}")
            
            # Train indicator model
            self.indicator_model.train(data, symbol)
            logging.info(f"Indicator model trained for {symbol}")
            
            complete_message = f"Training for {symbol} completed successfully."
            self.telegram_bot.send_message(complete_message)
            logging.info(complete_message)
            
        except Exception as e:
            error_message = f"Error during training of {symbol}: {e}"
            self.telegram_bot.send_message(error_message)
            logging.error(error_message)
            raise

    def train_rl_model(self, data_path: str):
        """Train the reinforcement learning model"""
        logging.info("Starting reinforcement learning training...")
        try:
            data = pd.read_csv(data_path)
            self.rl_model, eval_results = train_rl_agent(data)
            
            # Log evaluation results
            logging.info("RL Model Training Results:")
            logging.info(f"Mean Reward: {eval_results['mean_reward']}")
            logging.info(f"Sharpe Ratio: {eval_results.get('sharpe_ratio', 'N/A')}")
            
            self.telegram_bot.send_message("RL model training completed successfully.")
            
        except Exception as e:
            error_message = f"Error during RL model training: {e}"
            self.telegram_bot.send_message(error_message)
            logging.error(error_message)
            raise

    def analyze_market_sentiment(self, symbol: str) -> float:
        """Analyze market sentiment for a symbol"""
        try:
            # Get recent news headlines
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if not news:
                logging.warning(f"No news found for {symbol}")
                return 0.0
            
            # Extract headlines
            headlines = [item['title'] for item in news[:10]]  # Analyze last 10 headlines
            
            # Analyze sentiment
            sentiment_results = self.sentiment_analyzer.analyze_market_sentiment(headlines)
            logging.info(f"Sentiment analysis for {symbol}: {sentiment_results}")
            
            return sentiment_results['overall_score']
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis for {symbol}: {e}")
            return 0.0

    def run_backtest(self, data_path: str, **kwargs):
        """Run backtesting with advanced analytics"""
        try:
            results = run_backtest(data_path, **kwargs)
            
            # Log backtest results
            logging.info("\n=== Backtest Results ===")
            logging.info(f"Initial Portfolio Value: ${results['initial_value']:,.2f}")
            logging.info(f"Final Portfolio Value: ${results['final_value']:,.2f}")
            logging.info(f"Total Return: {results['total_return']:.2f}%")
            logging.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            logging.info(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
            
            # Send summary to Telegram
            summary = (f"Backtest Results:\n"
                      f"Total Return: {results['total_return']:.2f}%\n"
                      f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                      f"Max Drawdown: {results['max_drawdown']:.2f}%")
            self.telegram_bot.send_message(summary)
            
            return results
            
        except Exception as e:
            error_message = f"Error during backtesting: {e}"
            self.telegram_bot.send_message(error_message)
            logging.error(error_message)
            raise

    def calculate_position_size(self, symbol: str, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get current price
            current_price = self.data_manager.data[symbol]['Close'].iloc[-1]
            
            # Get risk metrics
            risk_score = self.risk_model.predict(self.data_manager.data[symbol], symbol)
            _, sl = self.tp_sl_model.predict(self.data_manager.data[symbol], symbol)
            
            # Calculate stop loss distance
            stop_loss_distance = abs(current_price - sl)
            
            # Calculate maximum risk amount
            risk_amount = self.balance * risk_per_trade * (1 - risk_score)
            
            # Calculate position size
            position_size = risk_amount / stop_loss_distance
            
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def execute_trades(self, data_manager: DataManager):
        """Execute trades based on all signals"""
        self.data_manager = data_manager  # Store for position sizing
        
        for symbol, data in data_manager.data.items():
            try:
                # Get all signals
                predicted_price = self.price_model.predict(data, symbol)
                indicator_signal = self.indicator_model.predict(data, symbol)
                risk_score = self.risk_model.predict(data, symbol)
                tp, sl = self.tp_sl_model.predict(data, symbol)
                sentiment_score = self.analyze_market_sentiment(symbol)
                
                # Get RL model prediction if available
                rl_signal = None
                if self.rl_model:
                    env = create_trading_env(data)
                    obs = env.reset()[0]
                    rl_action, _ = self.rl_model.predict(obs, deterministic=True)
                    rl_signal = float(rl_action[0])  # Convert to float for signal combination
                
                # Combine signals
                decision = self.combine_signals(
                    predicted_price=predicted_price,
                    indicator_signal=indicator_signal,
                    risk_score=risk_score,
                    tp=tp,
                    sl=sl,
                    rl_signal=rl_signal,
                    sentiment_score=sentiment_score
                )
                
                # Execute the trade
                self.execute_trade(decision, symbol, predicted_price, tp, sl)
                
                # Store signals
                self.latest_signals.append({
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'decision': decision,
                    'predicted_price': predicted_price,
                    'risk_score': risk_score,
                    'tp': tp,
                    'sl': sl,
                    'sentiment_score': sentiment_score,
                    'rl_signal': rl_signal
                })
                
            except Exception as e:
                logging.error(f"Error executing trades for {symbol}: {e}")
                continue

    def combine_signals(self, predicted_price: float, indicator_signal: float, 
                       risk_score: float, tp: float, sl: float, 
                       rl_signal: Optional[float], sentiment_score: float) -> str:
        """Combine all trading signals to make a decision"""
        try:
            # Check risk threshold first
            if risk_score > 0.7:
                return "hold"
            
            # Initialize signal scores
            buy_score = 0
            sell_score = 0
            
            # Price prediction signal (30% weight)
            if predicted_price > indicator_signal:
                buy_score += 0.3
            else:
                sell_score += 0.3
            
            # Sentiment signal (20% weight)
            if sentiment_score > 0.2:
                buy_score += 0.2
            elif sentiment_score < -0.2:
                sell_score += 0.2
            
            # RL signal if available (30% weight)
            if rl_signal is not None:
                if rl_signal > 0.2:
                    buy_score += 0.3
                elif rl_signal < -0.2:
                    sell_score += 0.3
            
            # Technical indicators (20% weight)
            if tp > sl:
                buy_score += 0.2
            else:
                sell_score += 0.2
            
            # Make decision
            if buy_score > sell_score and buy_score > 0.5:
                return "buy"
            elif sell_score > buy_score and sell_score > 0.5:
                return "sell"
            else:
                return "hold"
                
        except Exception as e:
            logging.error(f"Error in combine_signals: {e}")
            return "hold"

    def execute_trade(self, decision: str, symbol: str, predicted_price: float, tp: float, sl: float):
        """Execute a trade decision"""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(symbol)
            
            # Prepare trade message
            message = (
                f"Trading Decision for {symbol}:\n"
                f"Action: {decision}\n"
                f"Entry Price: ${predicted_price:.2f}\n"
                f"Take Profit: ${tp:.2f}\n"
                f"Stop Loss: ${sl:.2f}\n"
                f"Position Size: {position_size:.2f} units\n"
            )
            
            # Execute trade based on decision
            if decision == "buy" and position_size > 0:
                if symbol in self.positions:
                    logging.info(f"Already holding position in {symbol}")
                else:
                    self.positions[symbol] = {
                        'type': 'long',
                        'entry_price': predicted_price,
                        'size': position_size,
                        'tp': tp,
                        'sl': sl,
                        'entry_time': datetime.now()
                    }
                    self.balance -= predicted_price * position_size
                    message += "\nAction taken: Long position opened"
                    
            elif decision == "sell" and position_size > 0:
                if symbol in self.positions:
                    logging.info(f"Already holding position in {symbol}")
                else:
                    self.positions[symbol] = {
                        'type': 'short',
                        'entry_price': predicted_price,
                        'size': position_size,
                        'tp': tp,
                        'sl': sl,
                        'entry_time': datetime.now()
                    }
                    message += "\nAction taken: Short position opened"
            
            else:
                message += "\nAction taken: Holding"
            
            # Log trade
            logging.info(message)
            self.telegram_bot.send_message(message)
            
            # Record trade in history
            self.trade_history.append({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'action': decision,
                'price': predicted_price,
                'size': position_size if decision in ['buy', 'sell'] else 0,
                'tp': tp,
                'sl': sl
            })
            
        except Exception as e:
            error_message = f"Error executing trade for {symbol}: {e}"
            logging.error(error_message)
            self.telegram_bot.send_message(error_message)

    def manage_open_positions(self, data_manager: DataManager):
        """Manage and monitor open positions"""
        for symbol, position in list(self.positions.items()):
            try:
                current_price = data_manager.data[symbol]['Close'].iloc[-1]
                entry_price = position['entry_price']
                position_type = position['type']
                
                # Calculate profit/loss
                if position_type == 'long':
                    pnl = (current_price - entry_price) * position['size']
                else:  # short
                    pnl = (entry_price - current_price) * position['size']
                
                # Check stop loss
                if (position_type == 'long' and current_price <= position['sl']) or \
                   (position_type == 'short' and current_price >= position['sl']):
                    self._close_position(symbol, current_price, 'Stop Loss')
                    
                # Check take profit
                elif (position_type == 'long' and current_price >= position['tp']) or \
                     (position_type == 'short' and current_price <= position['tp']):
                    self._close_position(symbol, current_price, 'Take Profit')
                    
                # Update trailing stop if applicable
                else:
                    self._update_trailing_stop(symbol, current_price, position)
                    
            except Exception as e:
                logging.error(f"Error managing position for {symbol}: {e}")

    def _close_position(self, symbol: str, current_price: float, reason: str):
        """Close a position and record the trade"""
        try:
            position = self.positions[symbol]
            pnl = (current_price - position['entry_price']) * position['size']
            if position['type'] == 'short':
                pnl = -pnl
            
            self.balance += current_price * position['size'] + pnl
            
            # Record trade
            self.trade_history.append({
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'type': position['type'],
                'size': position['size'],
                'pnl': pnl,
                'reason': reason
            })
            
            # Notify
            message = (
                f"Position Closed - {symbol}\n"
                f"Reason: {reason}\n"
                f"Entry Price: ${position['entry_price']:.2f}\n"
                f"Exit Price: ${current_price:.2f}\n"
                f"P/L: ${pnl:.2f}\n"
                f"Return: {(pnl / (position['entry_price'] * position['size'])) * 100:.2f}%"
            )
            logging.info(message)
            self.telegram_bot.send_message(message)
            
            # Remove position
            del self.positions[symbol]
            
        except Exception as e:
            logging.error(f"Error closing position for {symbol}: {e}")

    def _update_trailing_stop(self, symbol: str, current_price: float, position: Dict):
        """Update trailing stop loss if applicable"""
        try:
            atr = self.data_manager.data[symbol]['ATR'].iloc[-1]
            
            if position['type'] == 'long':
                new_stop = current_price - (2 * atr)  # 2 ATR units below current price
                if new_stop > position['sl']:  # Only move stop up for long positions
                    position['sl'] = new_stop
                    logging.info(f"Updated trailing stop for {symbol} long position to: ${new_stop:.2f}")
            else:  # short position
                new_stop = current_price + (2 * atr)  # 2 ATR units above current price
                if new_stop < position['sl']:  # Only move stop down for short positions
                    position['sl'] = new_stop
                    logging.info(f"Updated trailing stop for {symbol} short position to: ${new_stop:.2f}")
                    
        except Exception as e:
            logging.error(f"Error updating trailing stop for {symbol}: {e}")

    def start_real_time_scanning(self, data_manager: DataManager, interval_seconds: int = 60):
        """Start real-time market scanning and trading"""
        logging.info(f"Starting real-time market scanning every {interval_seconds} seconds.")
        
        def scan_and_trade():
            try:
                logging.info("Scanning market for opportunities...")
                self.execute_trades(data_manager)
                self.manage_open_positions(data_manager)
            except Exception as e:
                logging.error(f"Error in scan_and_trade: {e}")
        
        schedule.every(interval_seconds).seconds.do(scan_and_trade)
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in scanning loop: {e}")
                time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    try:
        # Initialize with multiple symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        data_manager = DataManager(symbols)
        
        # Create and initialize trading bot
        trading_bot = TradingBot(initial_balance=100000)
        
        # Train all models
        trading_bot.train_all_models(data_manager)
        
        # Train RL model
        trading_bot.train_rl_model('market_data_cleaned_auto.csv')
        
        # Run backtest
        backtest_results = trading_bot.run_backtest(
            'market_data_cleaned_auto.csv',
            risk_percentage=0.02,
            max_positions=5
        )
        
        # Start real-time trading
        trading_bot.start_real_time_scanning(data_manager)
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")

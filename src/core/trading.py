import yfinance as yf
import schedule
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import os
import json
import uuid
import redis
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import asyncio
import secrets

from app.models.price_prediction import PricePredictionModel
from app.models.risk_management import RiskManagementModel
from app.models.tp_sl_management import TpSlManagementModel
from app.models.indicator_management import IndicatorManagementModel
from app.models.rl_trading import train_rl_agent, evaluate_rl_model, create_trading_env
from app.models.sentiment_analysis import SentimentAnalyzer
from app.models.backtesting import run_backtest
from app.telegram_bot import TelegramBot
from app.model_trainer import ModelTrainer
from app.models.news_retrieval import NewsRetriever
from app.plugins import plugin_manager
from app.plugins.events import EventType, NewsEventType

# Import des utilitaires créés
from src.utils.log_config import setup_logging
from src.utils.env_config import get_db_params, get_redis_params
from src.utils.redis_manager import redis_manager, retry_redis_operation, RedisConnectionError

# Load environment variables
load_dotenv()

# Configure logging
logger = setup_logging('trading', 'trading_bot.log')

# Initialize Redis (for AI validation communication)
# Utilise le gestionnaire Redis amélioré au lieu d'une connexion directe
redis_client = redis_manager.client

# Initialize database connection
db_params = get_db_params()

def get_db_connection(max_retries=3, retry_delay=2):
    """Get a connection to the PostgreSQL database"""
    retry_count = 0
    last_exception = None
    
    while retry_count < max_retries:
        try:
            conn = psycopg2.connect(**db_params)
            logger.info(f"Connected to database at {db_params['host']}:{db_params['port']}")
            return conn
        except psycopg2.OperationalError as e:
            retry_count += 1
            last_exception = e
            wait_time = retry_delay * (2 ** (retry_count - 1))  # Backoff exponentiel
            logger.warning(f"Database connection attempt {retry_count}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
            
    logger.error(f"All {max_retries} database connection attempts failed. Last error: {last_exception}")
    return None

# Fonction décorée pour publier un message sur Redis avec gestion des erreurs
@retry_redis_operation(max_retries=3)
def publish_trade_request(request_data):
    """
    Publie une demande de validation de trading sur Redis avec gestion des erreurs.
    
    Args:
        request_data: Données de la requête de validation
        
    Returns:
        True si la publication a réussi, False sinon
    """
    try:
        channel = 'trade_requests'
        message = json.dumps(request_data)
        return redis_manager.safe_publish(channel, message)
    except Exception as e:
        logger.error(f"Error publishing trade request: {e}")
        raise

# Fonction décorée pour s'abonner aux réponses de validation
@retry_redis_operation(max_retries=3)
def subscribe_to_validation_responses(timeout=5):
    """
    S'abonne aux réponses de validation avec un timeout.
    
    Args:
        timeout: Délai d'attente en secondes
        
    Returns:
        Message de validation ou None si timeout
    """
    try:
        pubsub = redis_manager.get_pubsub()
        channel = 'trade_responses'
        pubsub.subscribe(channel)
        
        # Attendre la réponse avec un timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            message = pubsub.get_message(timeout=0.1)
            if message and message['type'] == 'message':
                return json.loads(message['data'])
            time.sleep(0.1)
            
        logger.warning(f"Timeout waiting for validation response after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"Error subscribing to validation responses: {e}")
        raise

class DataManager:
    def __init__(self, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.symbols = symbols
        self.start_date = start_date or (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
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
        logging.info(f"Downloading data for {symbol} from {self.start_date} to {self.end_date}")
        try:
            # Use auto_adjust=False to maintain backward compatibility
            data = yf.download(symbol, start=self.start_date, end=self.end_date, auto_adjust=False)
            return data
        except Exception as e:
            logging.error(f"Error downloading data: {e}")
            return None

    def get_real_time_data(self, symbol):
        """Get real-time (1-minute interval) data for the given symbol"""
        try:
            # Use auto_adjust=False to maintain backward compatibility
            new_data = yf.download(symbol, period='1d', interval='1m', auto_adjust=False)
            if new_data.empty:
                logging.warning(f"No real-time data available for {symbol}")
                return None
            return new_data
        except Exception as e:
            logging.error(f"Error getting real-time data: {e}")
            return None

    def update_data(self):
        """Update data for all symbols"""
        logging.info("Updating data for all symbols.")
        for symbol in self.symbols:
            try:
                logging.info(f"Updating data for symbol: {symbol}")
                new_data = self.get_real_time_data(symbol)
                if new_data is not None:
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

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol"""
        try:
            if symbol in self.data and not self.data[symbol].empty:
                return self.data[symbol]['Close'].iloc[-1]
            else:
                # Try to fetch the latest price from yfinance directly
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    return data['Close'].iloc[-1]
                else:
                    raise ValueError(f"No data available for {symbol}")
        except Exception as e:
            logging.error(f"Error getting latest price for {symbol}: {e}")
            return 0.0

class TradingBot:
    """Advanced AI-powered trading bot with multiple decision models."""
    
    def __init__(self, initial_balance=100000.0, position_manager=None, model_trainer=None):
        self.initial_balance = initial_balance
        self.position_manager = position_manager
        self.order_history = []
        self.performance_history = []
        self.model_predictions = {}
        self.market_data = {}
        self.latest_signals = []
        self.trade_id_counter = 0
        
        # Model instances
        self.price_model = None
        self.risk_model = None
        self.tpsl_model = None
        self.indicator_model = None
        self.sentiment_analyzer = None
        self.transformer_model = None  # New Transformer model
        
        # Model trainer instance
        self.model_trainer = model_trainer
        
        # Trading settings
        self.max_positions = 5
        self.risk_percentage = 0.02  # 2% risk per trade
        self.use_ai_validation = True
        self.enable_transformer = os.environ.get('USE_TRANSFORMER_MODEL', 'true').lower() == 'true'
        
        # Status
        self.status = 'initialized'
        
        # Initialize configuration
        self.config = {
            'initial_balance': initial_balance,
            'max_positions': self.max_positions,
            'risk_percentage': self.risk_percentage,
            'use_ai_validation': self.use_ai_validation,
            'enable_transformer': self.enable_transformer
        }
        
        # Initialize position manager if not provided
        if self.position_manager is None:
            from app.models.position_manager import PositionManager
            self.position_manager = PositionManager(initial_balance=initial_balance)
            logging.info("Created new position manager")
        else:
            logging.info("Using provided position manager")
        
        # Create model trainer if not provided
        if self.model_trainer is None:
            from app.model_trainer import ModelTrainer
            self.model_trainer = ModelTrainer(trading_bot=self)
            logging.info("Created new model trainer")
        else:
            logging.info("Using provided model trainer")
        
        # Initialize plugin system
        self.plugin_manager = plugin_manager
        
        # Load enabled plugins
        self.plugin_manager.load_all_enabled_plugins()
        
        # Trigger system startup event
        self.plugin_manager.trigger_event(
            str(EventType.SYSTEM_STARTUP),
            config=self.config,
            timestamp=datetime.now()
        )
        
        self._load_models()
        logger.info(f"TradingBot initialized with ${initial_balance} balance")
        logger.info(f"Transformer model is {'enabled' if self.enable_transformer else 'disabled'}")

    def _load_models(self):
        """Load all trained models."""
        try:
            model_dir = os.environ.get('MODELS_DIR', 'saved_models')
            
            # Try to load global models first
            try:
                self.price_model = PricePredictionModel()
                self.price_model.load(os.path.join(model_dir, 'global_price_model.h5'))
                
                self.risk_model = RiskManagementModel()
                self.risk_model.load(os.path.join(model_dir, 'global_risk_model.h5'))
                
                self.tpsl_model = TpSlManagementModel()
                self.tpsl_model.load(os.path.join(model_dir, 'global_tpsl_model.h5'))
                
                self.indicator_model = IndicatorManagementModel()
                self.indicator_model.load(os.path.join(model_dir, 'global_indicator_model.h5'))
                
                # Try to load transformer model if enabled
                if self.enable_transformer:
                    try:
                        from app.models.transformer_model import FinancialTransformer
                        self.transformer_model = FinancialTransformer()
                        self.transformer_model.load(
                            os.path.join(model_dir, 'global_transformer_model'),
                            os.path.join(model_dir, 'global_transformer_scalers.pkl')
                        )
                        logger.info("Loaded global transformer model")
                    except Exception as e:
                        logger.warning(f"Failed to load global transformer model: {e}")
                
                logger.info("Loaded global models")
            except Exception as e:
                logger.warning(f"Failed to load global models: {e}. Will try symbol-specific models later.")
                
            # Initialize sentiment analyzer
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("Initialized sentiment analyzer")
            except Exception as e:
                logger.warning(f"Failed to initialize sentiment analyzer: {e}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def get_status(self) -> str:
        """Returns the current status of the trading bot"""
        return self.status
        
    def set_status(self, status: str) -> None:
        """Sets the status of the trading bot"""
        if status in ["running", "paused", "stopped"]:
            self.status = status
            logging.info(f"Trading bot status changed to: {status}")
        else:
            logging.error(f"Invalid status: {status}")

    def get_latest_signals(self) -> List[Dict]:
        """Returns the latest trading signals from database"""
        try:
            # Récupérer les signaux réels depuis la base de données
            conn = get_db_connection()
            if conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("""
                        SELECT s.symbol, s.decision, s.timestamp, s.validation_confidence
                        FROM trading_signals s
                        WHERE s.timestamp >= NOW() - INTERVAL '24 hours'
                        ORDER BY s.timestamp DESC
                        LIMIT 10
                    """)
                    
                    results = cur.fetchall()
                    if results:
                        self.latest_signals = [dict(row) for row in results]
                conn.close()
            
            return self.latest_signals
        except Exception as e:
            logging.error(f"Error retrieving latest signals: {e}")
            return self.latest_signals  # Return last known signals in case of error

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
            self.tpsl_model.train(data, symbol)
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
            # Initialiser le récupérateur de news
            news_retriever = NewsRetriever()
            
            # Trigger before news fetch event
            self.plugin_manager.trigger_event(
                str(NewsEventType.BEFORE_NEWS_FETCH),
                symbol=symbol,
                sources=["openrouter", "perplexity", "apis"],
                max_results=15
            )
            
            # Récupérer les news complètes avec des données de sentiment pré-analysées
            news_data = news_retriever.get_news_with_sentiment_data(symbol, max_results=15)
            
            # Trigger after news fetch event (let plugins modify the news data)
            news_event_results = self.plugin_manager.trigger_event(
                str(NewsEventType.AFTER_NEWS_FETCH),
                symbol=symbol,
                news=news_data,
                sources=["openrouter", "perplexity", "apis"],
                timestamp=datetime.now()
            )
            
            # If any plugin returned modified news data, use it
            if news_event_results:
                for result in news_event_results:
                    if result is not None and isinstance(result, list) and len(result) > 0:
                        news_data = result
                        break
            
            if not news_data:
                # Fallback à yfinance si aucune headline n'est trouvée
                logging.warning(f"Aucune news trouvée via APIs pour {symbol}, essai avec yfinance")
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    if news:
                        headlines = [item['title'] for item in news[:10]]
                        # Analyze sentiment from headlines only
                        sentiment_results = self.sentiment_analyzer.analyze_market_sentiment(headlines)
                        logging.info(f"Sentiment analysis for {symbol} (yfinance fallback): {sentiment_results}")
                        return sentiment_results['score']
                except Exception as e:
                    logging.error(f"Erreur lors de la récupération des news via yfinance: {e}")
            
            if not news_data:
                logging.warning(f"No news found for {symbol}")
                return 0.0
            
            # Log des données récupérées pour debug
            logging.info(f"News récupérées pour {symbol}: {len(news_data)} articles")
            
            # Extraire les titres et textes complets pour une analyse plus riche
            headlines = [item['title'] for item in news_data]
            full_texts = [item.get('full_text', item['title']) for item in news_data]
            
            # Utiliser les scores pré-calculés pour enrichir l'analyse
            pre_sentiment_scores = [item.get('sentiment_score', 0.0) for item in news_data]
            relevance_scores = [item.get('relevance_score', 0.5) for item in news_data]
            
            # Trigger before sentiment analysis event
            sentiment_params_results = self.plugin_manager.trigger_event(
                str(NewsEventType.BEFORE_SENTIMENT_ANALYSIS),
                symbol=symbol,
                news=news_data,
                analysis_method="ensemble"
            )
            
            # Apply any modifications to the analysis parameters
            analysis_method = "ensemble"
            if sentiment_params_results:
                for result in sentiment_params_results:
                    if result is not None and isinstance(result, dict):
                        if "news" in result:
                            news_data = result["news"]
                            # Update extracted data if news has been modified
                            headlines = [item['title'] for item in news_data]
                            full_texts = [item.get('full_text', item['title']) for item in news_data]
                            pre_sentiment_scores = [item.get('sentiment_score', 0.0) for item in news_data]
                            relevance_scores = [item.get('relevance_score', 0.5) for item in news_data]
                        
                        if "analysis_method" in result:
                            analysis_method = result["analysis_method"]
            
            # Analyze sentiment with comprehensive data
            sentiment_results = self.sentiment_analyzer.analyze_market_sentiment(
                headlines=headlines,
                full_texts=full_texts,
                pre_calculated_scores=pre_sentiment_scores,
                relevance_weights=relevance_scores
            )
            
            # Trigger after sentiment analysis event
            enhanced_results = self.plugin_manager.trigger_event(
                str(NewsEventType.AFTER_SENTIMENT_ANALYSIS),
                symbol=symbol,
                news=news_data,
                sentiment_results=sentiment_results,
                overall_score=sentiment_results['score']
            )
            
            # Apply any enhancements to the sentiment results
            if enhanced_results:
                for result in enhanced_results:
                    if result is not None and isinstance(result, dict) and "adjusted_score" in result:
                        sentiment_results = result
                        logging.info(f"Sentiment analysis enhanced by plugin: {result.get('enhanced_by', 'unknown')}")
            
            logging.info(f"Sentiment analysis for {symbol}: {sentiment_results}")
            
            # Use adjusted score if available
            final_score = sentiment_results.get('adjusted_score', sentiment_results['score'])
            
            # Trigger sentiment analyzed event
            self.plugin_manager.trigger_event(
                str(EventType.SENTIMENT_ANALYZED),
                symbol=symbol,
                sentiment_score=final_score,
                news=news_data,
                details=sentiment_results
            )
            
            return final_score
            
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
            _, sl = self.tpsl_model.predict(self.data_manager.data[symbol], symbol)
            
            # Calculate stop loss distance
            stop_loss_distance = abs(current_price - sl)
            
            # Calculate maximum risk amount
            risk_amount = self.current_balance * risk_per_trade * (1 - risk_score)
            
            # Calculate position size
            position_size = risk_amount / stop_loss_distance
            
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def execute_trades(self, data_manager: DataManager):
        """Execute trades based on all signals with AI validation"""
        self.data_manager = data_manager  # Store for position sizing
        
        for symbol, data in data_manager.data.items():
            try:
                # Get all signals
                predicted_price = self.price_model.predict(data, symbol)
                indicator_signal = self.indicator_model.predict(data, symbol)
                risk_score = self.risk_model.predict(data, symbol)
                tp, sl = self.tpsl_model.predict(data, symbol)
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
                
                # Prepare trade data
                current_price = data['Close'].iloc[-1]
                trade_data = {
                    'request_id': str(uuid.uuid4()),
                    'symbol': symbol,
                    'action': decision,
                    'price': float(current_price),
                    'take_profit': float(tp),
                    'stop_loss': float(sl),
                    'risk_score': float(risk_score),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store signals in database
                signal_id = self._store_signal(
                    symbol=symbol,
                    decision=decision,
                    predicted_price=predicted_price,
                    risk_score=risk_score,
                    tp=tp,
                    sl=sl,
                    sentiment_score=sentiment_score,
                    rl_signal=rl_signal
                )
                
                # Skip if decision is to hold
                if decision == "hold":
                    logging.info(f"Decision for {symbol} is to hold. Skipping trade execution.")
                    continue
                
                # Get AI validation if Redis is available
                validated = False
                validation_confidence = 0.0
                
                if redis_client:
                    validation_result = self._get_ai_validation(trade_data)
                    validated = validation_result.get('validated', False)
                    validation_confidence = validation_result.get('confidence', 0.0)
                    
                    # Update signal with validation results
                    self._update_signal_validation(signal_id, validated, validation_confidence)
                    
                    if not validated:
                        reason = validation_result.get('reason', 'Failed AI validation')
                        logging.info(f"Trade for {symbol} rejected: {reason}")
                        self.telegram_bot.send_message(f"Trade rejected for {symbol}: {reason}")
                        continue
                
                # If validated (or no validation required), execute the trade
                self.execute_trade(decision, symbol, current_price, tp, sl, validated, validation_confidence)
                
                # Store in latest signals
                self.latest_signals.append({
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'decision': decision,
                    'predicted_price': predicted_price,
                    'risk_score': risk_score,
                    'tp': tp,
                    'sl': sl,
                    'sentiment_score': sentiment_score,
                    'rl_signal': rl_signal,
                    'validated': validated,
                    'validation_confidence': validation_confidence
                })
                
            except Exception as e:
                logging.error(f"Error executing trades for {symbol}: {e}")
                continue
    
    def _get_ai_validation(self, trade_data: Dict) -> Dict:
        """Get validation from the AI validation service"""
        try:
            # Set channels
            request_channel = 'trade_requests'
            response_channel = 'trade_responses'
            request_id = trade_data['request_id']
            
            # Create pubsub to listen for response
            pubsub = redis_client.pubsub()
            pubsub.subscribe(response_channel)
            
            # Send trade request
            redis_client.publish(request_channel, json.dumps(trade_data))
            logging.info(f"Sent trade request for validation: {request_id}")
            
            # Wait for response with timeout
            start_time = time.time()
            timeout = 30  # 30 seconds timeout
            
            while time.time() - start_time < timeout:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    response = json.loads(message['data'])
                    if response.get('request_id') == request_id:
                        logging.info(f"Received validation response: {response}")
                        pubsub.unsubscribe()
                        return response
                        
            pubsub.unsubscribe()
            logging.warning(f"Timeout waiting for validation response for request: {request_id}")
            return {
                'validated': False,
                'confidence': 0.0,
                'reason': 'Validation timeout'
            }
            
        except Exception as e:
            logging.error(f"Error getting AI validation: {e}")
            return {
                'validated': False,
                'confidence': 0.0,
                'reason': f'Validation error: {str(e)}'
            }
            
    def _store_signal(self, symbol: str, decision: str, predicted_price: float,
                     risk_score: float, tp: float, sl: float, 
                     sentiment_score: Optional[float] = None,
                     rl_signal: Optional[float] = None) -> int:
        """Store trading signal in the database"""
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO trading_signals 
                        (symbol, timestamp, decision, predicted_price, risk_score, tp, sl, sentiment_score, rl_signal)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            symbol,
                            datetime.now(),
                            decision,
                            predicted_price,
                            risk_score,
                            tp,
                            sl,
                            sentiment_score,
                            rl_signal
                        )
                    )
                    signal_id = cursor.fetchone()[0]
                    conn.commit()
                conn.close()
                return signal_id
        except Exception as e:
            logging.error(f"Error storing signal: {e}")
        return -1
        
    def _update_signal_validation(self, signal_id: int, validated: bool, confidence: float) -> None:
        """Update a signal with AI validation results."""
        for signals_group in self.latest_signals:
            for signal in signals_group.get('signals', []):
                if signal.get('id') == signal_id:
                    signal['validated'] = validated
                    signal['validation_confidence'] = confidence
                    logging.info(f"Updated signal {signal_id} with validation: {validated}, confidence: {confidence}")
                    return
        logging.warning(f"Signal ID {signal_id} not found for validation update")
    
    def get_model_predictions(self, symbol):
        """
        Récupère les prédictions réelles des modèles pour un symbole donné.
        
        Args:
            symbol (str): Le symbole pour lequel récupérer les prédictions
            
        Returns:
            dict: Un dictionnaire contenant les prédictions des différents modèles
        """
        try:
            # Vérifier si nous avons des données de marché pour ce symbole
            if symbol not in self.market_data or self.market_data[symbol].empty:
                logging.warning(f"No market data available for {symbol}")
                return None
            
            # Récupérer les données récentes pour faire des prédictions
            data = self.market_data[symbol].tail(30).copy()
            
            # Initialiser le dictionnaire de prédictions
            predictions = {}
            
            # Prédictions du modèle de prix
            if self.price_model is not None:
                try:
                    # Préparer les données pour le modèle
                    prepared_data = data.copy()
                    
                    # Faire la prédiction pour les 5 prochains jours
                    price_predictions = []
                    current_data = prepared_data.copy()
                    
                    for i in range(5):
                        prediction = self.price_model.predict(current_data)
                        price_predictions.append(prediction)
                        
                        # Mettre à jour les données avec la prédiction pour la prochaine itération
                        new_row = current_data.iloc[-1].copy()
                        new_row['Close'] = prediction
                        current_data = current_data.append(new_row)
                    
                    # Ajouter les derniers prix réels suivis des prédictions
                    predictions['price_predictions'] = data['Close'].tolist() + price_predictions
                except Exception as e:
                    logging.error(f"Error getting price predictions for {symbol}: {e}")
            
            # Prédictions du modèle Transformer
            if self.transformer_model is not None and self.enable_transformer:
                try:
                    # Faire la prédiction pour les 5 prochains jours avec le modèle Transformer
                    transformer_preds = self.transformer_model.predict_sequence(data, steps=5)
                    
                    # Ajouter les derniers prix réels suivis des prédictions
                    predictions['transformer_predictions'] = data['Close'].tolist() + transformer_preds
                except Exception as e:
                    logging.error(f"Error getting transformer predictions for {symbol}: {e}")
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error in get_model_predictions for {symbol}: {e}")
            return None

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

    def execute_trade(self, decision: str, symbol: str, current_price: float, 
                     tp: float, sl: float, validated: bool = True, 
                     validation_confidence: float = 0.0):
        """Execute a trade decision"""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(symbol)
            
            # Prepare trade message
            message = (
                f"Trading Decision for {symbol}:\n"
                f"Action: {decision}\n"
                f"Entry Price: ${current_price:.2f}\n"
                f"Take Profit: ${tp:.2f}\n"
                f"Stop Loss: ${sl:.2f}\n"
                f"Position Size: {position_size:.2f} units\n"
                f"AI Validated: {validated}\n"
                f"Confidence: {validation_confidence:.2f}\n"
            )
            
            # Check if trading is enabled
            trading_enabled = self._get_bot_setting('trading_enabled') == 'true'
            
            # Execute trade based on decision
            if trading_enabled:
                # Create metadata for additional trade information
                metadata = {
                    'validated': validated,
                    'validation_confidence': validation_confidence,
                    'decision_source': 'combined_signals',
                    'signal_timestamp': datetime.now().isoformat()
                }
                
                # Open a position based on the decision
                if decision == "buy" and position_size > 0:
                    # Check if we already have a position for this symbol
                    if self.position_manager.get_position_count(symbol) > 0:
                        logging.info(f"Already holding position in {symbol}")
                    else:
                        # Open a long position
                        position = self.position_manager.open_position(
                            symbol=symbol,
                            direction='long',
                            entry_price=current_price,
                            size=position_size,
                            stop_loss=sl,
                            take_profit=tp,
                            metadata=metadata
                        )
                        message += f"\nAction taken: Long position opened (ID: {position.id})"
                        
                        # Store in database
                        self._store_trade_entry(symbol, 'long', current_price, position_size, tp, sl, validated, validation_confidence)
                        
                elif decision == "sell" and position_size > 0:
                    # Check if we already have a position for this symbol
                    if self.position_manager.get_position_count(symbol) > 0:
                        logging.info(f"Already holding position in {symbol}")
                    else:
                        # Open a short position
                        position = self.position_manager.open_position(
                            symbol=symbol,
                            direction='short',
                            entry_price=current_price,
                            size=position_size,
                            stop_loss=sl,
                            take_profit=tp,
                            metadata=metadata
                        )
                        message += f"\nAction taken: Short position opened (ID: {position.id})"
                        
                        # Store in database
                        self._store_trade_entry(symbol, 'short', current_price, position_size, tp, sl, validated, validation_confidence)
                
                else:
                    message += "\nAction taken: Holding"
            else:
                message += "\nTrading is disabled. This is a simulated trade."
            
            # Log trade
            logging.info(message)
            self.telegram_bot.send_message(message)
            
            # Record trade in history
            self.order_history.append({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'action': decision,
                'price': current_price,
                'size': position_size if decision in ['buy', 'sell'] else 0,
                'tp': tp,
                'sl': sl,
                'validated': validated,
                'validation_confidence': validation_confidence
            })
            
        except Exception as e:
            error_message = f"Error executing trade for {symbol}: {e}"
            logging.error(error_message)
            self.telegram_bot.send_message(error_message)
            
    def _store_trade_entry(self, symbol: str, trade_type: str, entry_price: float, 
                          size: float, tp: float, sl: float, 
                          validated: bool, validation_confidence: float) -> None:
        """Store trade entry in the database"""
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO trade_history 
                        (symbol, entry_time, entry_price, type, size, validated, validation_confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            symbol,
                            datetime.now(),
                            entry_price,
                            trade_type,
                            size,
                            validated,
                            validation_confidence
                        )
                    )
                    conn.commit()
                conn.close()
        except Exception as e:
            logging.error(f"Error storing trade entry: {e}")
            
    def _get_bot_setting(self, setting_key: str, default_value: str = '') -> str:
        """Get a bot setting from the database"""
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT setting_value FROM bot_settings 
                        WHERE setting_key = %s
                        """,
                        (setting_key,)
                    )
                    result = cursor.fetchone()
                conn.close()
                
                if result:
                    return result['setting_value']
                    
        except Exception as e:
            logging.error(f"Error getting bot setting: {e}")
            
        return default_value

    def manage_open_positions(self, data_manager: DataManager):
        """Check and manage open positions"""
        try:
            # Get current market data for all symbols with open positions
            open_position_symbols = set()
            for position in self.position_manager.positions.values():
                open_position_symbols.add(position.symbol)
            
            # Check each symbol with an open position
            for symbol in open_position_symbols:
                try:
                    # Get current price
                    current_price = data_manager.get_latest_price(symbol)
                    
                    # Check positions for this symbol
                    closed_position_ids = self.position_manager.check_stops(symbol, current_price)
                    
                    # Process closed positions
                    for position_id in closed_position_ids:
                        # Find the closed position in closed_positions
                        closed_position = next(
                            (p for p in self.position_manager.closed_positions if p.id == position_id), 
                            None
                        )
                        
                        if closed_position:
                            # Get close reason (should be stored in the position's metadata from check_stops)
                            reason = closed_position.metadata.get('close_reason', 'Unknown')
                            
                            # Store trade exit in database
                            self._store_trade_exit(
                                symbol=closed_position.symbol,
                                entry_time=closed_position.entry_time,
                                exit_time=closed_position.exit_time,
                                exit_price=closed_position.exit_price,
                                pnl=closed_position.realized_pnl,
                                reason=reason
                            )
                            
                            # Notify
                            message = (
                                f"Position Closed - {closed_position.symbol}\n"
                                f"Reason: {reason}\n"
                                f"Entry Price: ${closed_position.entry_price:.2f}\n"
                                f"Exit Price: ${closed_position.exit_price:.2f}\n"
                                f"P/L: ${closed_position.realized_pnl:.2f}\n"
                                f"Return: {closed_position.realized_pnl_percentage:.2f}%"
                            )
                            logging.info(message)
                            self.telegram_bot.send_message(message)
                    
                    # Update trailing stops
                    for position in self.position_manager.positions.values():
                        if position.symbol == symbol:
                            self._update_trailing_stop(position.id, current_price)
                    
                except Exception as e:
                    logging.error(f"Error managing position for {symbol}: {e}")
            
            # Calculate and store metrics
            self._calculate_and_store_metrics()
            
        except Exception as e:
            logging.error(f"Error in manage_open_positions: {e}")

    def _close_position(self, position_id: str, current_price: float, reason: str):
        """Close a position and record the trade"""
        try:
            # Find the position
            position = self.position_manager.positions.get(position_id)
            if not position:
                logging.error(f"Position {position_id} not found")
                return
            
            # Update position metadata with close reason
            if not position.metadata:
                position.metadata = {}
            position.metadata['close_reason'] = reason
            
            # Close the position
            pnl = self.position_manager.close_position(position_id, current_price)
            
            # Log the closure
            message = (
                f"Position Closed - {position.symbol}\n"
                f"Reason: {reason}\n"
                f"Entry Price: ${position.entry_price:.2f}\n"
                f"Exit Price: ${current_price:.2f}\n"
                f"P/L: ${pnl:.2f}"
            )
            logging.info(message)
            
        except Exception as e:
            logging.error(f"Error in _close_position: {e}")

    def _update_trailing_stop(self, position_id: str, current_price: float):
        """Update trailing stop loss for a position if applicable"""
        try:
            # Get the position
            position = self.position_manager.positions.get(position_id)
            if not position:
                return
            
            # Trailing stop configuration
            trailing_stop_enabled = self._get_bot_setting('trailing_stop_enabled') == 'true'
            activation_threshold_pct = float(self._get_bot_setting('trailing_activation_pct', '1.0')) / 100
            trailing_distance_pct = float(self._get_bot_setting('trailing_distance_pct', '0.5')) / 100
            
            if not trailing_stop_enabled or not position.stop_loss:
                return
                
            # Calculate profit threshold and new stop level
            if position.direction == 'long':
                profit_threshold = position.entry_price * (1 + activation_threshold_pct)
                if current_price > profit_threshold:
                    new_stop = current_price * (1 - trailing_distance_pct)
                    if new_stop > position.stop_loss:
                        self.position_manager.update_position_stops(position_id, new_stop_loss=new_stop)
                        logging.info(f"Updated trailing stop for {position.symbol} to {new_stop:.2f}")
            else:  # Short position
                profit_threshold = position.entry_price * (1 - activation_threshold_pct)
                if current_price < profit_threshold:
                    new_stop = current_price * (1 + trailing_distance_pct)
                    if new_stop < position.stop_loss:
                        self.position_manager.update_position_stops(position_id, new_stop_loss=new_stop)
                        logging.info(f"Updated trailing stop for {position.symbol} to {new_stop:.2f}")
                        
        except Exception as e:
            logging.error(f"Error updating trailing stop: {e}")

    def _store_trade_exit(self, symbol: str, entry_time: datetime, exit_time: datetime, 
                         exit_price: float, pnl: float, reason: str) -> None:
        """Store trade exit in the database"""
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE trade_history
                        SET exit_time = %s, exit_price = %s, pnl = %s, reason = %s
                        WHERE symbol = %s AND entry_time = %s
                        """,
                        (
                            exit_time,
                            exit_price,
                            pnl,
                            reason,
                            symbol,
                            entry_time
                        )
                    )
                    conn.commit()
                conn.close()
        except Exception as e:
            logging.error(f"Error storing trade exit: {e}")

    def start_real_time_scanning(self, data_manager: DataManager, interval_seconds: int = 60, enable_incremental_learning: bool = True):
        """Start real-time market scanning and trading
        
        Args:
            data_manager: The data manager instance
            interval_seconds: Seconds between each scan
            enable_incremental_learning: Whether to update models with new data
        """
        logging.info(f"Starting real-time market scanning every {interval_seconds} seconds.")
        logging.info(f"Incremental learning is {'enabled' if enable_incremental_learning else 'disabled'}")
        
        # Create data directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Store settings
        self.enable_incremental_learning = enable_incremental_learning
        
        def scan_and_trade():
            try:
                logging.info("Scanning market for opportunities...")
                
                # Update market data
                prev_data = {}
                if self.enable_incremental_learning:
                    # Store previous data snapshots to determine what's new
                    for symbol, data in data_manager.data.items():
                        prev_data[symbol] = data.copy()
                
                # Update market data
                data_manager.update_data()
                
                # Perform incremental learning if enabled
                if self.enable_incremental_learning and hasattr(self.model_trainer, 'incremental_update_models'):
                    for symbol, current_data in data_manager.data.items():
                        if symbol in prev_data:
                            # Get new data since last update (may be empty if no new data)
                            latest_timestamp = prev_data[symbol].index[-1] if not prev_data[symbol].empty else None
                            if latest_timestamp is not None:
                                new_data = current_data[current_data.index > latest_timestamp]
                                
                                if not new_data.empty:
                                    logging.info(f"Found {len(new_data)} new data points for {symbol}")
                                    
                                    # Update models with new data (non-blocking)
                                    asyncio.create_task(
                                        self.model_trainer.incremental_update_models(symbol, new_data)
                                    )
                
                # Store market snapshots
                self._store_market_snapshots(data_manager)
                
                # Execute trades based on signals
                self.execute_trades(data_manager)
                
                # Manage open positions
                self.manage_open_positions(data_manager)
                
                # Calculate and store performance metrics
                self._calculate_and_store_metrics()
                
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
                
    def _store_market_snapshots(self, data_manager: DataManager) -> None:
        """Store market data snapshots in the database"""
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    for symbol, data in data_manager.data.items():
                        last_row = data.iloc[-1]
                        cursor.execute(
                            """
                            INSERT INTO market_data 
                            (symbol, timestamp, open, high, low, close, volume, sma_20, sma_50, rsi, macd, bb_upper, bb_lower)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                symbol,
                                datetime.now(),
                                float(last_row['Open']),
                                float(last_row['High']),
                                float(last_row['Low']),
                                float(last_row['Close']),
                                int(last_row['Volume']),
                                float(last_row.get('SMA_20', 0)),
                                float(last_row.get('SMA_50', 0)),
                                float(last_row.get('RSI', 0)),
                                float(last_row.get('MACD', 0)),
                                float(last_row.get('Bollinger_Upper', 0)),
                                float(last_row.get('Bollinger_Lower', 0))
                            )
                        )
                    conn.commit()
                conn.close()
        except Exception as e:
            logging.error(f"Error storing market snapshots: {e}")
            
    def _calculate_and_store_metrics(self) -> None:
        """Calculate and store performance metrics"""
        try:
            # Get current date and market data for open positions
            current_date = datetime.now().date()
            market_prices = {}
            
            # Gather current prices for all positions
            for position in self.position_manager.positions.values():
                try:
                    if position.symbol not in market_prices:
                        market_prices[position.symbol] = self.data_manager.get_latest_price(position.symbol)
                except Exception as e:
                    logging.error(f"Error getting price for {position.symbol}: {e}")
            
            # Get portfolio stats from position manager
            portfolio_stats = self.position_manager.get_portfolio_stats(market_prices)
            
            # Calculate trading statistics
            closed_positions = self.position_manager.closed_positions
            
            total_trades = len(closed_positions)
            if total_trades == 0:
                winning_trades = 0
                losing_trades = 0
                win_rate = 0
                average_win = 0
                average_loss = 0
                profit_factor = 0
            else:
                # Calculate win/loss statistics
                winning_trades = sum(1 for pos in closed_positions if pos.realized_pnl > 0)
                losing_trades = sum(1 for pos in closed_positions if pos.realized_pnl <= 0)
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                winning_amounts = [pos.realized_pnl for pos in closed_positions if pos.realized_pnl > 0]
                losing_amounts = [abs(pos.realized_pnl) for pos in closed_positions if pos.realized_pnl <= 0]
                
                average_win = sum(winning_amounts) / len(winning_amounts) if winning_amounts else 0
                average_loss = sum(losing_amounts) / len(losing_amounts) if losing_amounts else 0
                
                gross_profit = sum(winning_amounts)
                gross_loss = sum(losing_amounts)
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1 if gross_profit > 0 else 0)
            
            # Store metrics in database
            self._store_performance_metrics(
                date=current_date,
                balance=portfolio_stats['current_balance'],
                equity=portfolio_stats['total_equity'],
                daily_pnl=portfolio_stats['unrealized_pnl'],  # This is an approximation
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor
            )
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            
    def _store_performance_metrics(self, date, balance, equity, daily_pnl, total_trades,
                                 winning_trades, losing_trades, win_rate, average_win,
                                 average_loss, profit_factor):
        """Store performance metrics in the database"""
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    # Check if we already have a record for today
                    cursor.execute(
                        "SELECT id FROM performance_metrics WHERE date = %s",
                        (date,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        # Update existing record
                        cursor.execute(
                            """
                            UPDATE performance_metrics
                            SET balance = %s, equity = %s, daily_pnl = %s,
                                total_trades = %s, winning_trades = %s, losing_trades = %s,
                                win_rate = %s, average_win = %s, average_loss = %s,
                                profit_factor = %s, updated_at = %s
                            WHERE date = %s
                            """,
                            (
                                balance, equity, daily_pnl,
                                total_trades, winning_trades, losing_trades,
                                win_rate, average_win, average_loss,
                                profit_factor, datetime.now(), date
                            )
                        )
                    else:
                        # Insert new record
                        cursor.execute(
                            """
                            INSERT INTO performance_metrics
                            (date, balance, equity, daily_pnl, total_trades, winning_trades,
                            losing_trades, win_rate, average_win, average_loss, profit_factor)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                date, balance, equity, daily_pnl,
                                total_trades, winning_trades, losing_trades,
                                win_rate, average_win, average_loss, profit_factor
                            )
                        )
                    conn.commit()
                conn.close()
        except Exception as e:
            logging.error(f"Error storing performance metrics: {e}")

    def generate_trading_signals(self, symbol, data):
        """Generate trading signals using multiple models."""
        logger.info(f"Generating trading signals for {symbol}")
        
        signals = []
        
        try:
            # Get the latest data
            latest_data = data.copy().iloc[-100:]  # Use last 100 bars for signal generation
            
            # Price prediction signals
            price_predictions = None
            transformer_predictions = None
            
            if self.price_model:
                try:
                    price_predictions = self.price_model.predict(latest_data)
                    self.model_predictions[f"{symbol}_price"] = price_predictions
                except Exception as e:
                    logger.error(f"Error getting price predictions for {symbol}: {e}")
            
            # Transformer model predictions (if available)
            if self.transformer_model and self.enable_transformer:
                try:
                    transformer_predictions = self.transformer_model.predict(latest_data)
                    self.model_predictions[f"{symbol}_transformer"] = transformer_predictions
                    logger.info(f"Transformer predictions for {symbol}: {transformer_predictions}")
                except Exception as e:
                    logger.error(f"Error getting transformer predictions for {symbol}: {e}")
            
            # Combine price predictions (using transformer if available)
            final_predictions = None
            if transformer_predictions is not None:
                final_predictions = transformer_predictions
                logger.info(f"Using transformer predictions for {symbol}")
            elif price_predictions is not None:
                final_predictions = price_predictions
                logger.info(f"Using traditional model predictions for {symbol}")
            
            if final_predictions is not None:
                # Calculate prediction direction and strength
                current_price = latest_data['Close'].iloc[-1]
                future_price = final_predictions[0]  # First prediction
                
                price_change_pct = (future_price - current_price) / current_price * 100
                
                # Generate signal based on prediction
                if price_change_pct > 1.0:  # More than 1% increase expected
                    signals.append({
                        'symbol': symbol,
                        'direction': 'buy',
                        'confidence': min(abs(price_change_pct) / 5, 1.0),  # Scale confidence
                        'timestamp': datetime.now().isoformat(),
                        'source': 'transformer' if transformer_predictions is not None else 'price_model',
                        'predicted_change': price_change_pct
                    })
                elif price_change_pct < -1.0:  # More than 1% decrease expected
                    signals.append({
                        'symbol': symbol,
                        'direction': 'sell',
                        'confidence': min(abs(price_change_pct) / 5, 1.0),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'transformer' if transformer_predictions is not None else 'price_model',
                        'predicted_change': price_change_pct
                    })
            
            # Add risk assessment to signals
            if self.risk_model and signals:
                for signal in signals:
                    try:
                        risk_score = self.risk_model.predict_risk(latest_data)
                        signal['risk_score'] = risk_score
                        
                        # Adjust confidence based on risk
                        signal['confidence'] *= (1 - risk_score)
                    except Exception as e:
                        logger.error(f"Error calculating risk for {symbol}: {e}")
            
            # Add sentiment analysis if available
            if self.sentiment_analyzer:
                try:
                    sentiment = self.sentiment_analyzer.analyze_sentiment(symbol)
                    
                    if sentiment:
                        sentiment_signal = {
                            'symbol': symbol,
                            'direction': 'buy' if sentiment['sentiment_score'] > 0.2 else 'sell' if sentiment['sentiment_score'] < -0.2 else 'neutral',
                            'confidence': abs(sentiment['sentiment_score']),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'sentiment',
                            'sentiment': sentiment
                        }
                        signals.append(sentiment_signal)
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            
            # Validate signals with AI validation if enabled
            if self.use_ai_validation and redis_client:
                try:
                    validated_signals = self._validate_signals_with_ai(symbol, signals, latest_data)
                    signals = validated_signals
                except Exception as e:
                    logger.error(f"Error validating signals with AI: {e}")
                    
            # Store the signals
            if symbol not in self.latest_signals:
                self.latest_signals.append({
                    'symbol': symbol,
                    'signals': signals,
                    'timestamp': datetime.now().isoformat()
                })
                
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals for {symbol}: {e}")
            return []

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

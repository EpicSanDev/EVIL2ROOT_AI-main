#!/usr/bin/env python3
import os
import redis
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import pickle
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
import traceback
import requests
import re
import threading
import uuid
try:
    import talib
except ImportError:
    try:
        # Si le module principal n'est pas disponible, essayer talib.abstract
        import talib.abstract as talib
    except ImportError:
        # Si toujours pas disponible, essayer de trouver le module dans d'autres emplacements
        try:
            # Parfois installé sous ce nom
            from ta_lib import talib
        except ImportError:
            raise ImportError("""
Le module 'talib' n'est pas installé correctement. 
Pour l'installer :
1. Compiler et installer TA-Lib depuis les sources
2. Installer le package Python TA-Lib
Référez-vous au script fix-talib-install.sh pour plus d'informations.
""")
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
# Nouveaux imports pour modèles avancés
from langchain import LLMChain
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Import des utilitaires créés
from src.utils.log_config import setup_logging
from src.utils.env_config import get_redis_params, get_env_var
from src.utils.redis_manager import redis_manager, retry_redis_operation

# Setup logging
logger = setup_logging('ai_validator', 'ai_validator.log')

class AITradeValidator:
    """
    AI service to validate trading decisions before execution.
    Acts as an additional layer of confirmation to improve trading accuracy.
    """
    
    def __init__(self):
        # Utilisation du gestionnaire Redis amélioré au lieu d'une connexion directe
        self.redis_manager = redis_manager
        
        # Subscription channels
        self.trade_request_channel = 'trade_requests'
        self.trade_response_channel = 'trade_responses'
        
        # Model paths
        self.model_dir = os.environ.get('MODEL_DIR', 'saved_models')
        
        # Load common models if available
        self.common_models = self._load_common_models()
        
        # Load advanced embeddings model for semantic analysis
        try:
            # Chargement d'un modèle d'embedding avancé pour l'analyse sémantique
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Loaded advanced embedding model for semantic analysis")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None
            
        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            logger.info("Loaded sentiment analysis pipeline")
        except Exception as e:
            logger.error(f"Error loading sentiment analysis pipeline: {e}")
            self.sentiment_analyzer = None
        
        # Decision thresholds
        self.confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.65))
        
        # OpenRouter API configuration
        self.openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', '')
        self.use_openrouter = bool(self.openrouter_api_key)
        self.claude_model = os.environ.get('CLAUDE_MODEL', 'anthropic/claude-3.7-sonnet')
        
        # Check for OpenRouter API key
        if not self.openrouter_api_key:
            logger.warning("OpenRouter API key not found. AI validation will be limited to ML models only.")
            # Notifier l'administrateur si les notifications sont configurées
            try:
                from src.services.notification_service import NotificationService
                notification_service = NotificationService()
                notification_service.send_admin_notification(
                    "Configuration incomplète",
                    "La clé API OpenRouter n'est pas configurée. La validation AI sera limitée aux modèles ML uniquement."
                )
            except ImportError:
                logger.warning("Impossible de notifier l'administrateur : service de notification non disponible.")
            except Exception as e:
                logger.warning(f"Erreur lors de la notification : {e}")
        
        # Start listener thread
        self.running = True
        self.listener_thread = threading.Thread(target=self.listen_for_trade_requests)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
        logger.info("AI Trade Validator initialized successfully")
        logger.info(f"Using advanced AI model: {self.claude_model}")

    def _load_common_models(self) -> Dict:
        """Load common models that aren't symbol-specific"""
        try:
            common_models = {}
            
            # Example: Load ensemble model if available
            ensemble_path = os.path.join(self.model_dir, 'ensemble_model.pkl')
            if os.path.exists(ensemble_path):
                common_models['ensemble'] = joblib.load(ensemble_path)
                logger.info("Loaded ensemble model")
            
            # Add other common models here
            
            return common_models
        except Exception as e:
            logger.error(f"Error loading common models: {e}")
            return {}

    def _load_symbol_models(self, symbol: str) -> Dict:
        """Load models for a specific symbol"""
        try:
            models = {}
            
            # Define file paths for each model
            price_model_path = os.path.join(self.model_dir, f"{symbol}_price_model.pkl")
            risk_model_path = os.path.join(self.model_dir, f"{symbol}_risk_model.pkl")
            tp_sl_model_path = os.path.join(self.model_dir, f"{symbol}_tp_sl_model.pkl")
            indicator_model_path = os.path.join(self.model_dir, f"{symbol}_indicator_model.pkl")
            
            # Load models if they exist
            if os.path.exists(price_model_path):
                models['price'] = joblib.load(price_model_path)
            
            if os.path.exists(risk_model_path):
                models['risk'] = joblib.load(risk_model_path)
                
            if os.path.exists(tp_sl_model_path):
                models['tp_sl'] = joblib.load(tp_sl_model_path)
                
            if os.path.exists(indicator_model_path):
                models['indicator'] = joblib.load(indicator_model_path)
            
            return models
        
        except Exception as e:
            logger.error(f"Error loading models for {symbol}: {e}")
            return {}

    def fetch_market_data(self, symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
        """
        Récupère les données de marché pour un symbole donné
        
        Args:
            symbol: Symbole boursier
            period: Période de temps ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Intervalle entre les points de données ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame contenant les données de marché
        """
        try:
            logging.info(f"Fetching market data for {symbol}")
            
            # Use auto_adjust=False to maintain backward compatibility
            data = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
            
            if data.empty:
                logging.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            logging.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, data: pd.DataFrame) -> None:
        """Add technical indicators to the dataframe"""
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        std_dev = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (std_dev * 2)
        data['BB_Lower'] = data['BB_Middle'] - (std_dev * 2)
        
        # ATR
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()
        
        # Fill NaN values
        data.fillna(method='bfill', inplace=True)

    def validate_trade(self, trade_request: Dict) -> Dict:
        """Validate a trading decision using multiple AI models and LLMs"""
        logger.info(f"Validating trade: {trade_request}")
        
        symbol = trade_request.get('symbol', '')
        proposed_action = trade_request.get('decision', 'hold')
        proposed_price = float(trade_request.get('price', 0))
        proposed_tp = float(trade_request.get('tp', 0)) 
        proposed_sl = float(trade_request.get('sl', 0))
        risk_score = float(trade_request.get('risk_score', 0.5))
        signal_id = int(trade_request.get('signal_id', -1))
        
        try:
            # Fetch recent market data for analysis
            market_data = self.fetch_market_data(symbol, period="7d", interval="1h")
            
            # Load symbol-specific models
            models = self._load_symbol_models(symbol)
            
            # Combine with common models
            models.update(self.common_models)
            
            # Run multi-layered validation
            validation_result = self._run_validation(
                symbol, proposed_action, proposed_price, proposed_tp, proposed_sl,
                risk_score, market_data, models
            )
            
            # Log the detailed validation results
            logger.info(f"Validation result for {symbol} {proposed_action}: {validation_result}")
            
            # Add signal_id to the result
            validation_result['signal_id'] = signal_id
            
            # Send the result back to Redis using a different channel
            if self.redis_manager:
                self.redis_manager.safe_publish(
                    self.trade_response_channel,
                    json.dumps(validation_result)
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}", exc_info=True)
            error_result = {
                'validated': False,
                'confidence': 0.0,
                'signal_id': signal_id,
                'reason': f"Validation error: {str(e)}"
            }
            if self.redis_manager:
                self.redis_manager.safe_publish(
                    self.trade_response_channel,
                    json.dumps(error_result)
                )
            return error_result

    def _run_validation(self, symbol: str, proposed_action: str, proposed_price: float, 
                     proposed_tp: float, proposed_sl: float, risk_score: float,
                     market_data: pd.DataFrame, models: Dict) -> Dict:
        """Run multi-layered validation of the trade decision using advanced AI models"""
        validation_results = {
            'validated': False,
            'confidence': 0.0,
            'reason': "",
            'validations': [],
        }
        
        validations = []
        
        # Skip validation for 'hold' decisions
        if proposed_action.lower() == 'hold':
            validation_results['validated'] = True
            validation_results['confidence'] = 0.75
            validation_results['reason'] = "Hold decisions are automatically validated."
            return validation_results
        
        try:
            # 1. Base model validation - only if models are available
            base_validation = None
            base_confidence = 0.0
            
            if models and 'price' in models:
                # Prepare features for model validation
                last_data = market_data.iloc[-1:].copy()
                # Here we would prepare features for model input
                
                try:
                    # Run model validation
                    if proposed_action == 'buy':
                        prediction = models['price'].predict_proba(last_data)[0][1]  # probability of positive class
                        base_validation = prediction > 0.55  # slightly higher than random
                        base_confidence = prediction
                    else:  # sell
                        prediction = models['price'].predict_proba(last_data)[0][0]  # probability of negative class
                        base_validation = prediction > 0.55  # slightly higher than random
                        base_confidence = prediction
                        
                    validations.append(('base_model', base_validation, base_confidence))
                    logger.info(f"Base model validation: {base_validation} (confidence: {base_confidence:.2f})")
                except Exception as e:
                    logger.error(f"Error in base model validation: {e}")
            
            # 2. Validate market trend alignment using technical indicators
            trend_confidence, trend_valid = self._validate_market_trend(market_data, proposed_action)
            validations.append(('market_trend', trend_valid, trend_confidence))
            logger.info(f"Market trend validation: {trend_valid} (confidence: {trend_confidence:.2f})")
            
            # 3. Sentiment analysis on recent data if available
            sentiment_valid = True  # default to neutral
            sentiment_confidence = 0.5  # neutral confidence
            
            if self.sentiment_analyzer:
                try:
                    # Calculate price momentum score
                    price_changes = market_data['Close'].pct_change().dropna().tail(10).values
                    momentum_score = np.mean(price_changes) / np.std(price_changes) if np.std(price_changes) > 0 else 0
                    
                    # Normalize to 0-1 range
                    normalized_momentum = 1 / (1 + np.exp(-5 * momentum_score))  # sigmoid function
                    
                    # Check if momentum aligns with proposed action
                    if (proposed_action == 'buy' and normalized_momentum > 0.5) or \
                       (proposed_action == 'sell' and normalized_momentum < 0.5):
                        sentiment_valid = True
                        sentiment_confidence = abs(normalized_momentum - 0.5) * 2  # Scale to 0-1
                    else:
                        sentiment_valid = False
                        sentiment_confidence = abs(normalized_momentum - 0.5) * 2  # Scale to 0-1
                        
                    validations.append(('sentiment', sentiment_valid, sentiment_confidence))
                    logger.info(f"Sentiment validation: {sentiment_valid} (confidence: {sentiment_confidence:.2f})")
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")
            
            # 4. Risk assessment based on stop loss and take profit levels
            risk_reward_ratio = float('inf')  # Default to infinity
            risk_valid = True  # Default to valid
            risk_confidence = 0.5  # Default to neutral
            
            if proposed_sl > 0 and proposed_tp > 0:
                # Calculate risk-reward ratio
                if proposed_action == 'buy':
                    risk = proposed_price - proposed_sl
                    reward = proposed_tp - proposed_price
                else:  # sell
                    risk = proposed_sl - proposed_price
                    reward = proposed_price - proposed_tp
                
                if risk > 0:
                    risk_reward_ratio = reward / risk
                    # Validate if risk-reward ratio is at least 1.5
                    risk_valid = risk_reward_ratio >= 1.5
                    # Confidence based on how good the R/R ratio is (1.5 -> 0.5, 3.0 -> 1.0)
                    risk_confidence = min(risk_reward_ratio / 3.0, 1.0)
                    
            validations.append(('risk_reward', risk_valid, risk_confidence))
            logger.info(f"Risk-reward validation: {risk_valid} (ratio: {risk_reward_ratio:.2f}, confidence: {risk_confidence:.2f})")
            
            # 5. Validate with Claude via OpenRouter (highest weight)
            claude_valid = False
            claude_confidence = 0.0
            claude_reason = ""
            
            if self.use_openrouter:
                claude_valid, claude_confidence, claude_reason = self._validate_with_claude(
                    symbol, proposed_action, proposed_price, proposed_tp, proposed_sl, market_data
                )
                validations.append(('claude', claude_valid, claude_confidence))
                logger.info(f"Claude validation: {claude_valid} (confidence: {claude_confidence:.2f})")
            
            # Calculate weighted confidence score and final decision
            # Give higher weight to Claude and risk assessment
            weights = {
                'base_model': 0.15,
                'market_trend': 0.2,
                'sentiment': 0.15,
                'risk_reward': 0.2,
                'claude': 0.3,  # Claude has the highest weight
            }
            
            total_weight = 0
            weighted_confidence = 0
            positive_weight = 0
            
            # Process all validations with their weights
            for name, valid, confidence in validations:
                weight = weights.get(name, 0.1)  # Default weight for unknown validations
                total_weight += weight
                weighted_confidence += confidence * weight
                
                if valid:
                    positive_weight += weight
            
            # Normalize confidence
            if total_weight > 0:
                normalized_confidence = weighted_confidence / total_weight
            else:
                normalized_confidence = 0.5  # Default
                
            # Determine final validation based on weighted positive validations
            final_validation = (positive_weight / total_weight > 0.6) if total_weight > 0 else False
            
            # If Claude was used and has high confidence, give its opinion more weight
            if self.use_openrouter and claude_confidence > 0.7:
                # Override with Claude's decision if it has high confidence
                final_validation = claude_valid
                normalized_confidence = (normalized_confidence + claude_confidence) / 2
            
            # Final validation also needs to meet confidence threshold
            validation_results['validated'] = final_validation and normalized_confidence >= self.confidence_threshold
            validation_results['confidence'] = normalized_confidence
            
            # Use Claude's reason if available, otherwise generate one
            if claude_reason:
                validation_results['reason'] = claude_reason
            else:
                validation_results['reason'] = self._generate_reason(
                    validation_results['validated'], 
                    normalized_confidence,
                    [v[1] for v in validations]
                )
                
            validation_results['validations'] = [
                {'name': name, 'valid': valid, 'confidence': conf} 
                for name, valid, conf in validations
            ]
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}", exc_info=True)
            return {
                'validated': False,
                'confidence': 0.0,
                'reason': f"Validation error: {str(e)}",
                'validations': []
            }

    def _validate_market_trend(self, data: pd.DataFrame, proposed_action: str) -> Tuple[float, bool]:
        """Validate if the proposed action aligns with the market trend"""
        try:
            # Simple trend analysis based on moving averages
            last_row = data.iloc[-1]
            
            # Short-term trend
            short_trend_up = last_row['SMA_20'] > last_row['SMA_50']
            
            # Check if action aligns with trend
            if proposed_action == 'buy':
                valid = short_trend_up
            else:  # sell
                valid = not short_trend_up
            
            # Calculate confidence based on the strength of the trend
            trend_strength = abs(last_row['SMA_20'] - last_row['SMA_50']) / last_row['SMA_50']
            confidence = min(trend_strength * 5, 1.0)  # Scale to 0-1
            
            return confidence, valid
            
        except Exception as e:
            logger.error(f"Error in trend validation: {e}")
            return 0.5, True  # Neutral position if error

    def _validate_with_claude(self, symbol: str, proposed_action: str, 
                            proposed_price: float, proposed_tp: float, 
                            proposed_sl: float, market_data: pd.DataFrame) -> Tuple[bool, float, str]:
        """Validate trade using Claude via OpenRouter API"""
        if not self.openrouter_api_key:
            logger.error("OpenRouter API key not available. AI validation cannot be performed.")
            return False, 0.0, "La validation avancée par IA n'est pas disponible : clé API OpenRouter manquante. Veuillez configurer cette clé ou désactiver la validation IA avancée."
        
        try:
            # Amélioration: utilisation d'une fenêtre de données plus large pour un meilleur contexte
            recent_data = market_data.tail(48)  # Étendu à 48 heures pour plus de contexte
            
            # Calcul des métriques de marché sur différentes périodes pour une meilleure analyse
            price_change_48h = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]) * 100 if len(recent_data) >= 48 else 0
            price_change_24h = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-24]) / recent_data['Close'].iloc[-24]) * 100 if len(recent_data) >= 24 else 0
            price_change_12h = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-12]) / recent_data['Close'].iloc[-12]) * 100 if len(recent_data) >= 12 else 0
            price_change_6h = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-6]) / recent_data['Close'].iloc[-6]) * 100 if len(recent_data) >= 6 else 0
            price_change_1h = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-2]) / recent_data['Close'].iloc[-2]) * 100 if len(recent_data) >= 2 else 0
            
            # Amélioration: calculs de volatilité plus détaillés
            price_volatility_48h = recent_data['Close'].pct_change().std() * 100
            price_volatility_24h = recent_data['Close'].iloc[-24:].pct_change().std() * 100 if len(recent_data) >= 24 else 0
            price_volatility_12h = recent_data['Close'].iloc[-12:].pct_change().std() * 100 if len(recent_data) >= 12 else 0
            price_volatility_6h = recent_data['Close'].iloc[-6:].pct_change().std() * 100 if len(recent_data) >= 6 else 0
            
            # Amélioration: analyse de volume avancée
            avg_volume = recent_data['Volume'].mean()
            recent_avg_volume = recent_data['Volume'].iloc[-6:].mean()  # Volume moyen des 6 dernières heures
            volume_ratio = recent_avg_volume / avg_volume if avg_volume > 0 else 1.0
            volume_change = ((recent_data['Volume'].iloc[-1] - recent_data['Volume'].iloc[0]) / recent_data['Volume'].iloc[0]) * 100 if recent_data['Volume'].iloc[0] > 0 else 0
            
            # Amélioration: détection de divergence de volume/prix
            price_direction = "up" if price_change_6h > 0 else "down"
            volume_direction = "up" if recent_data['Volume'].iloc[-1] > recent_data['Volume'].iloc[-6:].mean() else "down"
            volume_price_divergence = price_direction != volume_direction
            
            # Données de marché actuelles
            current_price = recent_data['Close'].iloc[-1]
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Amélioration: calcul des supports et résistances basés sur les extrema locaux
            supports = []
            resistances = []
            
            # Identifier les points hauts et bas significatifs sur les 48 dernières heures
            for i in range(5, len(recent_data) - 5):
                # Identifier les supports potentiels (points bas locaux)
                if (recent_data['Low'].iloc[i] <= recent_data['Low'].iloc[i-1:i+2].min() and 
                    recent_data['Low'].iloc[i] <= recent_data['Low'].iloc[i-5:i+5].min()):
                    supports.append(recent_data['Low'].iloc[i])
                
                # Identifier les résistances potentielles (points hauts locaux)
                if (recent_data['High'].iloc[i] >= recent_data['High'].iloc[i-1:i+2].max() and 
                    recent_data['High'].iloc[i] >= recent_data['High'].iloc[i-5:i+5].max()):
                    resistances.append(recent_data['High'].iloc[i])
            
            # Conserver uniquement les 3 supports et résistances les plus proches
            supports = sorted([s for s in supports if s < current_price], reverse=True)[:3]
            resistances = sorted([r for r in resistances if r > current_price])[:3]
            
            # Préparer le résumé des indicateurs techniques
            last_row = recent_data.iloc[-1]
            
            # S'assurer que les indicateurs techniques existent dans les données
            rsi_value = last_row.get('RSI', 'N/A')
            macd_value = last_row.get('MACD', 'N/A')
            macd_signal = last_row.get('Signal_Line', 'N/A')
            macd_hist = last_row.get('MACD_Histogram', 'N/A') if 'MACD_Histogram' in last_row else (float(macd_value) - float(macd_signal) if macd_value != 'N/A' and macd_signal != 'N/A' else 'N/A')
            bb_upper = last_row.get('BB_Upper', 'N/A')
            bb_lower = last_row.get('BB_Lower', 'N/A')
            bb_middle = last_row.get('BB_Middle', 'N/A')
            
            # Amélioration: calcul d'indicateurs supplémentaires
            ema9 = last_row.get('EMA9', 'N/A')
            ema21 = last_row.get('EMA21', 'N/A')
            ema50 = last_row.get('EMA50', 'N/A')
            ema200 = last_row.get('EMA200', 'N/A')
            
            # Déterminer la direction des EMAs s'ils sont disponibles
            ema_trend = "N/A"
            if ema9 != 'N/A' and ema21 != 'N/A' and ema50 != 'N/A':
                if ema9 > ema21 > ema50:
                    ema_trend = "Strong Bullish"
                elif ema9 > ema21 and ema21 < ema50:
                    ema_trend = "Moderately Bullish"
                elif ema9 < ema21 < ema50:
                    ema_trend = "Strong Bearish"
                elif ema9 < ema21 and ema21 > ema50:
                    ema_trend = "Moderately Bearish"
                else:
                    ema_trend = "Mixed"
            
            # Amélioration: calcul du squeeze momentum indicator
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 'N/A' and bb_upper != 'N/A' and bb_lower != 'N/A' and bb_middle > 0 else 'N/A'
            if bb_width != 'N/A':
                squeeze_momentum = "High Volatility" if bb_width > 0.05 else "Low Volatility (Potential Squeeze)"
            else:
                squeeze_momentum = "N/A"
            
            # Calculer la position du prix dans les bandes de Bollinger en pourcentage
            if bb_upper != 'N/A' and bb_lower != 'N/A':
                bb_width_value = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
                price_position_in_bb = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            else:
                bb_width_value = 'N/A'
                price_position_in_bb = 'N/A'
            
            # Amélioration: calculer des statistiques supplémentaires sur le risk-reward ratio
            if proposed_action.lower() == 'buy':
                risk = proposed_price - proposed_sl if proposed_sl > 0 else 0
                reward = proposed_tp - proposed_price if proposed_tp > 0 else 0
                distance_to_nearest_resistance = min([(r - current_price) for r in resistances]) if resistances else float('inf')
                distance_to_nearest_support = min([(current_price - s) for s in supports]) if supports else float('inf')
            else:  # sell
                risk = proposed_sl - proposed_price if proposed_sl > 0 else 0
                reward = proposed_price - proposed_tp if proposed_tp > 0 else 0
                distance_to_nearest_resistance = min([(r - current_price) for r in resistances]) if resistances else float('inf')
                distance_to_nearest_support = min([(current_price - s) for s in supports]) if supports else float('inf')
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Amélioration: déterminer si le trade est aligné avec les niveaux support/résistance
            if proposed_action.lower() == 'buy':
                sr_alignment = "Good" if distance_to_nearest_support < distance_to_nearest_resistance * 0.5 else "Poor"
            else:
                sr_alignment = "Good" if distance_to_nearest_resistance < distance_to_nearest_support * 0.5 else "Poor"
            
            # Amélioration: créer un prompt plus détaillé et structuré pour Claude
            prompt = f"""Vous êtes un analyste trading expert avec une vaste expertise en analyse quantitative et technique. Analysez cette proposition de trade sur {symbol} de façon exhaustive et déterminez si elle est valide selon les données de marché fournies.

DÉTAILS DU TRADE:
- Symbole: {symbol}
- Action: {proposed_action.upper()}
- Prix de marché actuel: ${current_price:.4f}
- Prix d'entrée proposé: ${proposed_price:.4f}
- Objectif de prise de profit: ${proposed_tp:.4f} ({((proposed_tp - proposed_price) / proposed_price * 100):.2f}% depuis l'entrée)
- Niveau de stop loss: ${proposed_sl:.4f} ({((proposed_sl - proposed_price) / proposed_price * 100):.2f}% depuis l'entrée)
- Ratio risque-récompense: {risk_reward_ratio:.2f}
- Heure d'analyse: {current_time}

DONNÉES DE MARCHÉ RÉCENTES:
- Variation de prix 48h: {price_change_48h:.2f}%
- Variation de prix 24h: {price_change_24h:.2f}%
- Variation de prix 12h: {price_change_12h:.2f}%
- Variation de prix 6h: {price_change_6h:.2f}%
- Variation de prix 1h: {price_change_1h:.2f}%
- Volatilité 48h: {price_volatility_48h:.2f}%
- Volatilité 24h: {price_volatility_24h:.2f}%
- Volatilité 12h: {price_volatility_12h:.2f}%
- Volatilité 6h: {price_volatility_6h:.2f}%

ANALYSE DE VOLUME:
- Volume moyen: {avg_volume:.2f}
- Volume récent (6h): {recent_avg_volume:.2f}
- Ratio volume récent/moyen: {volume_ratio:.2f}
- Variation de volume: {volume_change:.2f}%
- Divergence volume/prix: {"Oui" if volume_price_divergence else "Non"}

NIVEAUX SUPPORTS ET RÉSISTANCES:
- Supports récents: {", ".join([f"${s:.4f}" for s in supports]) if supports else "Aucun identifié"}
- Résistances récentes: {", ".join([f"${r:.4f}" for r in resistances]) if resistances else "Aucun identifié"}
- Distance au support le plus proche: ${distance_to_nearest_support:.4f} ({(distance_to_nearest_support/current_price*100):.2f}%)
- Distance à la résistance la plus proche: ${distance_to_nearest_resistance:.4f} ({(distance_to_nearest_resistance/current_price*100):.2f}%)
- Alignement avec support/résistance: {sr_alignment}

INDICATEURS TECHNIQUES:
- RSI (14): {rsi_value}
- MACD: {macd_value}
- Signal MACD: {macd_signal}
- Histogramme MACD: {macd_hist}
- EMA9: {ema9}
- EMA21: {ema21}
- EMA50: {ema50}
- EMA200: {ema200}
- Tendance EMA: {ema_trend}
- Bande de Bollinger Supérieure: ${bb_upper if bb_upper != 'N/A' else 'N/A'}
- Bande de Bollinger Moyenne: ${bb_middle if bb_middle != 'N/A' else 'N/A'}
- Bande de Bollinger Inférieure: ${bb_lower if bb_lower != 'N/A' else 'N/A'}
- Largeur des bandes de Bollinger: {bb_width_value if bb_width_value != 'N/A' else 'N/A'}
- Position du prix dans BB (0-1): {price_position_in_bb if price_position_in_bb != 'N/A' else 'N/A'}
- Indicateur de Squeeze: {squeeze_momentum}

Effectuez une analyse complète et critique de cette proposition de trade en suivant ces étapes:

1. ANALYSE DES INDICATEURS TECHNIQUES:
   - Évaluez la configuration des indicateurs (RSI, MACD, Bandes de Bollinger, EMAs)
   - Vérifiez si les indicateurs confirment ou contredisent la direction du trade proposé
   - Identifiez toute divergence ou signal de retournement de tendance

2. ÉVALUATION DU RISQUE ET DE LA RÉCOMPENSE:
   - Analysez le ratio risque-récompense par rapport aux conditions actuelles du marché
   - Estimez la probabilité de succès basée sur la volatilité et les niveaux support/résistance
   - Évaluez si le stop loss et le take profit sont placés à des niveaux optimaux

3. ANALYSE DE LA STRUCTURE DU MARCHÉ:
   - Examinez comment le prix se comporte par rapport aux niveaux support/résistance
   - Déterminez si le volume confirme l'action du prix
   - Identifiez si le niveau d'entrée proposé correspond à un point stratégique

4. ÉVALUATION DU CONTEXTE GLOBAL:
   - Déterminez si le trade est aligné avec la tendance générale
   - Vérifiez s'il y a des signaux contradictoires entre différentes périodes de temps
   - Évaluez si le moment est opportun pour ce type de trade

Après votre analyse, fournissez votre évaluation finale sous forme d'objet JSON avec exactement ces champs:
1. "validated": true/false (si ce trade devrait être exécuté)
2. "confidence": 0.0-1.0 (votre confiance dans cette évaluation, avec des valeurs plus élevées indiquant une conviction plus forte)
3. "reason": explication concise de votre décision, mentionnant les facteurs déterminants (max 3 phrases)
4. "key_factors": liste des 3 facteurs les plus importants qui ont influencé votre décision

Répondez UNIQUEMENT avec l'objet JSON.
"""
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://evil2root.ai/",  # Required by OpenRouter
                "X-Title": "Evil2Root Trading AI"  # Optional - helps OpenRouter identify your app
            }
            
            # Amélioration: utilisation d'une température plus basse pour des réponses plus cohérentes
            payload = {
                "model": self.claude_model,
                "messages": [
                    {"role": "system", "content": "Vous êtes un analyste quantitatif de trading hautement qualifié avec une vaste expérience en analyse technique, gestion des risques et comportement du marché. Vous basez vos décisions sur des données et des analyses objectives plutôt que sur l'émotion. Vous êtes rigoureux, précis et conservateur - vous ne validez un trade que lorsque les preuves sont solides."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.05,  # Température beaucoup plus basse pour des réponses plus déterministes et précises
                "max_tokens": 750,
                "stream": False
            }
            
            logger.info(f"Envoi de la demande de validation de trade à OpenRouter pour {symbol} {proposed_action}")
            
            # Amélioration: gestion de retry en cas d'échec
            max_retries = 2
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30  # Timeout augmenté pour permettre une analyse plus approfondie
                    )
                    
                    if response.status_code == 200:
                        break
                    else:
                        retry_count += 1
                        logger.warning(f"OpenRouter API error (retry {retry_count}/{max_retries}): {response.status_code} - {response.text}")
                        time.sleep(2)  # Pause avant de réessayer
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Request error (retry {retry_count}/{max_retries}): {e}")
                    time.sleep(2)
            
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return False, 0.0, f"OpenRouter API error: {response.status_code}"
            
            result = response.json()
            
            if 'choices' not in result or len(result['choices']) == 0:
                logger.error(f"Invalid response from OpenRouter: {result}")
                return False, 0.0, "Invalid response from OpenRouter"
            
            answer = result['choices'][0]['message']['content'].strip()
            
            # Amélioration: parsing JSON plus robuste
            try:
                # Tenter d'extraire uniquement la partie JSON
                json_pattern = r'({[\s\S]*})'
                json_matches = re.search(json_pattern, answer)
                
                if json_matches:
                    json_str = json_matches.group(1)
                    # Nettoyer toute syntaxe incorrecte potentielle
                    json_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', json_str)
                    json_str = re.sub(r',\s*}', r'}', json_str)
                    
                    response_data = json.loads(json_str)
                    
                    validated = response_data.get('validated', False)
                    confidence = float(response_data.get('confidence', 0.0))
                    reason = response_data.get('reason', "No reason provided")
                    key_factors = response_data.get('key_factors', [])
                    
                    # Enrichir le message du log avec des facteurs clés
                    factors_str = ", ".join(key_factors) if isinstance(key_factors, list) else "No key factors provided"
                    logger.info(f"Claude validation for {symbol} {proposed_action}: {validated} (confidence: {confidence:.2f})")
                    logger.info(f"Key factors: {factors_str}")
                    logger.debug(f"Claude reasoning: {reason}")
                    
                    return validated, confidence, reason
                else:
                    logger.warning(f"Failed to extract JSON from Claude response: {answer}")
                    
                    # Fallback: try to determine validation from text analysis if JSON extraction fails
                    positive_indicators = ['valid', 'validated', 'recommend', 'execute', 'take the trade', 'good opportunity']
                    negative_indicators = ['invalid', 'not recommended', 'avoid', 'poor', 'high risk', 'reject']
                    
                    validated = any(indicator in answer.lower() for indicator in positive_indicators)
                    confidence = 0.6 if validated else 0.4  # Default moderate confidence
                    
                    logger.info(f"Used fallback text analysis for validation: {validated} with confidence 0.6")
                    return validated, confidence, "Parsed from non-JSON response (fallback mode)"
            except Exception as e:
                logger.error(f"Error parsing Claude response: {e} - {answer}")
                return False, 0.0, f"Error parsing response: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in Claude validation: {e}")
            traceback.print_exc()
            return False, 0.0, f"Validation error: {str(e)}"

    def _generate_reason(self, validated: bool, confidence: float, validations: List[bool]) -> str:
        """Generate a human-readable reason for the validation result"""
        if validated:
            return f"Trade validated with {confidence:.2f} confidence."
        else:
            if not any(validations):
                return "All validation checks failed."
            elif confidence < self.confidence_threshold:
                return f"Confidence score {confidence:.2f} below threshold {self.confidence_threshold}."
            else:
                return "Some validation checks failed."

    @retry_redis_operation(max_retries=5)
    def listen_for_trade_requests(self):
        """Listen for trade requests from the trading bot"""
        logger.info("Starting to listen for trade requests...")
        pubsub = self.redis_manager.get_pubsub()
        pubsub.subscribe(self.trade_request_channel)
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    trade_request = json.loads(message['data'])
                    logger.info(f"Received trade request: {trade_request}")
                    
                    # Validate trade
                    validation_result = self.validate_trade(trade_request)
                    
                    # Send response back with error handling
                    success = self.redis_manager.safe_publish(
                        self.trade_response_channel,
                        json.dumps(validation_result)
                    )
                    
                    if success:
                        logger.info(f"Sent validation response: {validation_result}")
                    else:
                        logger.error("Failed to send validation response")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create and start the validator
    validator = AITradeValidator()
    
    try:
        logger.info("AI Trade Validator service starting...")
        validator.listen_for_trade_requests()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        logger.error(traceback.format_exc())

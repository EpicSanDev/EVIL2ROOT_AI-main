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
import ollama
import re
import threading
import uuid
import talib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ai_validator')

class AITradeValidator:
    """
    AI service to validate trading decisions before execution.
    Acts as an additional layer of confirmation to improve trading accuracy.
    """
    
    def __init__(self):
        # Connect to Redis
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Subscription channels
        self.trade_request_channel = 'trade_requests'
        self.trade_response_channel = 'trade_responses'
        
        # Model paths
        self.model_dir = os.environ.get('MODEL_DIR', 'saved_models')
        
        # Load common models if available
        self.common_models = self._load_common_models()
        
        # Decision thresholds
        self.confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.65))
        
        # OpenRouter API configuration
        self.openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', '')
        self.use_openrouter = bool(self.openrouter_api_key)
        self.claude_model = os.environ.get('CLAUDE_MODEL', 'anthropic/claude-3-opus-20240229')
        
        # Check for OpenRouter API key
        if not self.openrouter_api_key:
            logger.warning("OpenRouter API key not found. AI validation will be limited to ML models only.")
            self.use_openrouter = False
        
        # Start listener thread
        self.running = True
        self.listener_thread = threading.Thread(target=self.listen_for_trade_requests)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
        logger.info("AI Trade Validator initialized successfully")

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

    def fetch_market_data(self, symbol: str, period: str = "7d", interval: str = "1h") -> pd.DataFrame:
        """Fetch the latest market data for validation"""
        try:
            logger.info(f"Fetching market data for {symbol}")
            data = yf.download(symbol, period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Add technical indicators
            self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

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
            if self.redis_client:
                self.redis_client.publish('trade_validation_results', json.dumps(validation_result))
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}", exc_info=True)
            error_result = {
                'validated': False,
                'confidence': 0.0,
                'signal_id': signal_id,
                'reason': f"Validation error: {str(e)}"
            }
            if self.redis_client:
                self.redis_client.publish('trade_validation_results', json.dumps(error_result))
            return error_result

    def _run_validation(self, symbol: str, proposed_action: str, proposed_price: float, 
                     proposed_tp: float, proposed_sl: float, risk_score: float,
                     market_data: pd.DataFrame, models: Dict) -> Dict:
        """Run multi-layered validation of the trade decision"""
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
            validation_results['reason'] = "Hold position validated automatically"
            return validation_results
            
        # Validate risk/reward ratio
        if proposed_sl > 0 and proposed_tp > 0:
            if proposed_action.lower() == 'buy':
                risk = proposed_price - proposed_sl
                reward = proposed_tp - proposed_price
            else:  # sell
                risk = proposed_sl - proposed_price
                reward = proposed_price - proposed_tp
                
            if risk > 0:  # Valid stop loss
                rr_ratio = reward / risk
                validated_rr = rr_ratio >= 1.5
                validations.append(validated_rr)
                validation_results['validations'].append({
                    'name': 'risk_reward_ratio',
                    'validated': validated_rr,
                    'value': float(rr_ratio),
                    'threshold': 1.5
                })
        
        # Validate market trend
        trend_confidence, trend_aligned = self._validate_market_trend(
            market_data, proposed_action
        )
        validations.append(trend_aligned)
        validation_results['validations'].append({
            'name': 'market_trend',
            'validated': trend_aligned,
            'value': float(trend_confidence),
            'threshold': 0.6
        })
        
        # ML model validation
        if 'xgb_classifier' in models:
            try:
                # Prepare features (recent market data)
                features = market_data.tail(1)[['open', 'high', 'low', 'close', 'volume', 
                                            'rsi', 'macd', 'macd_signal', 'upper_band', 
                                            'middle_band', 'lower_band']].values
                
                # Predict using XGBoost
                proba = models['xgb_classifier'].predict_proba(features)[0]
                predicted_idx = models['xgb_classifier'].predict(features)[0]
                
                # Map index to action ('buy', 'sell', 'hold')
                predicted_action = ['sell', 'hold', 'buy'][predicted_idx]
                
                # Check alignment with proposal
                aligned = (
                    (predicted_action == 'buy' and proposed_action.lower() == 'buy') or
                    (predicted_action == 'sell' and proposed_action.lower() == 'sell')
                )
                
                ml_confidence = float(max(proba))
                validations.append(aligned)
                validation_results['validations'].append({
                    'name': 'ml_model',
                    'validated': aligned,
                    'value': ml_confidence,
                    'threshold': 0.55
                })
            except Exception as e:
                logger.error(f"Error in ML validation: {e}")
        
        # LLM validation with Claude (if API key is available)
        if self.use_openrouter:
            try:
                is_valid, confidence, reason = self._validate_with_claude(
                    symbol, proposed_action, proposed_price, proposed_tp, proposed_sl, market_data
                )
                validations.append(is_valid)
                validation_results['validations'].append({
                    'name': 'claude_validation',
                    'validated': is_valid,
                    'value': float(confidence),
                    'threshold': 0.7
                })
                
                # Add Claude's reasoning to the result
                validation_results['llm_reason'] = reason
            except Exception as e:
                logger.error(f"Error in Claude validation: {e}")
        
        # Calculate overall validation result
        if validations:
            # Calculate weighted confidence score
            validation_count = len(validations)
            if validation_count > 0:
                confidence = sum(v['value'] for v in validation_results['validations']) / validation_count
                # Consider validated if majority of checks passed and confidence exceeds threshold
                passes = sum(1 for v in validations if v)
                validation_results['validated'] = (passes / validation_count >= 0.5 and 
                                              confidence >= self.confidence_threshold)
                validation_results['confidence'] = float(confidence)
                validation_results['reason'] = self._generate_reason(
                    validation_results['validated'], confidence, validations
                )
        
        return validation_results

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
            return False, 0.0, "OpenRouter API key not available"
        
        try:
            # Prepare market data summary for Claude
            recent_data = market_data.tail(24)  # Last 24 hours for hourly data
            price_change_24h = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100
            
            current_price = recent_data['close'].iloc[-1]
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate additional metrics
            avg_volume = recent_data['volume'].mean()
            price_volatility = recent_data['close'].pct_change().std() * 100
            
            # Prepare technical indicator summary
            last_row = recent_data.iloc[-1]
            rsi_value = last_row['rsi']
            macd_value = last_row['macd']
            macd_signal = last_row['macd_signal']
            bb_upper = last_row['upper_band']
            bb_lower = last_row['lower_band']
            
            # Create prompt for Claude
            prompt = f"""You are an expert trading assistant evaluating a potential trade. Analyze this {symbol} trade proposal carefully and determine if it's valid based on the provided market data.

Trade Details:
- Symbol: {symbol}
- Action: {proposed_action}
- Current Price: ${current_price:.2f}
- Proposed Entry: ${proposed_price:.2f}
- Take Profit: ${proposed_tp:.2f}
- Stop Loss: ${proposed_sl:.2f}
- Time: {current_time}

Recent Market Data:
- 24h Price Change: {price_change_24h:.2f}%
- Average Volume: {avg_volume:.2f}
- Price Volatility: {price_volatility:.2f}%

Technical Indicators:
- RSI (14): {rsi_value:.2f}
- MACD: {macd_value:.2f}
- MACD Signal: {macd_signal:.2f}
- Bollinger Upper: ${bb_upper:.2f}
- Bollinger Lower: ${bb_lower:.2f}

First, analyze the trade proposal in detail, evaluating its risk/reward ratio, alignment with technical indicators, and current market conditions.

Then, provide your final assessment as a JSON object with these exact fields:
1. "validated": true/false (whether this trade should be executed)
2. "confidence": 0.0-1.0 (your confidence in this assessment)
3. "reason": brief explanation of your decision

Respond ONLY with the JSON object.
"""
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.claude_model,
                "messages": [
                    {"role": "system", "content": "You are a highly skilled trading analyst who evaluates trades based on technical and fundamental factors."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            # Make API call to OpenRouter
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    parsed = json.loads(content)
                    validated = parsed.get('validated', False)
                    confidence = float(parsed.get('confidence', 0.0))
                    reason = parsed.get('reason', "No reason provided")
                    
                    logger.info(f"Claude validation for {symbol} {proposed_action}: {validated} (conf: {confidence})")
                    return validated, confidence, reason
                    
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Error parsing Claude response: {e} - {content}")
                    # Try to extract boolean and confidence with regex as fallback
                    validation_match = re.search(r'(validated|valid).*?(?::|is)\s*(true|false)', content, re.IGNORECASE)
                    confidence_match = re.search(r'confidence.*?(?::|is)\s*(0\.\d+)', content, re.IGNORECASE)
                    
                    validated = validation_match.group(2).lower() == 'true' if validation_match else False
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                    
                    return validated, confidence, "Error parsing full response, extracted partial results"
            else:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                return False, 0.0, f"API error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error in Claude validation: {e}", exc_info=True)
            return False, 0.0, f"Error: {str(e)}"

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

    def listen_for_trade_requests(self):
        """Listen for trade requests from the trading bot"""
        logger.info("Starting to listen for trade requests...")
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.trade_request_channel)
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    trade_request = json.loads(message['data'])
                    logger.info(f"Received trade request: {trade_request}")
                    
                    # Validate trade
                    validation_result = self.validate_trade(trade_request)
                    
                    # Send response back
                    self.redis_client.publish(
                        self.trade_response_channel,
                        json.dumps(validation_result)
                    )
                    
                    logger.info(f"Sent validation response: {validation_result}")
                    
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

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Union, Optional, Tuple
import re
import logging
import os
from collections import deque

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class MarketRegimeDetector:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.sentiment_history = deque(maxlen=window_size)
        self.volatility_history = deque(maxlen=window_size)
        
    def update(self, sentiment: float, volatility: float) -> str:
        """Update histories and detect market regime"""
        self.sentiment_history.append(sentiment)
        self.volatility_history.append(volatility)
        
        if len(self.sentiment_history) < self.window_size:
            return 'normal'
            
        avg_sentiment = np.mean(list(self.sentiment_history))
        sentiment_std = np.std(list(self.sentiment_history))
        avg_volatility = np.mean(list(self.volatility_history))
        
        if avg_volatility > 0.02:  # High volatility
            if avg_sentiment > 0.2:
                return 'volatile_bullish'
            elif avg_sentiment < -0.2:
                return 'volatile_bearish'
            return 'volatile_neutral'
        else:  # Normal volatility
            if avg_sentiment > 0.2:
                return 'stable_bullish'
            elif avg_sentiment < -0.2:
                return 'stable_bearish'
            return 'stable_neutral'

class SentimentAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        self.tokenizer = None
        self.model = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.max_length = 100
        self.vocab_size = 10000
        self.regime_detector = MarketRegimeDetector()
        
        # Initialize transformer model for financial sentiment
        self.transformer_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        
        # Sentiment history for trend analysis
        self.sentiment_history = deque(maxlen=100)
        self.trend_weights = np.exp(np.linspace(-1, 0, 100))  # Exponential weights for trend
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            logging.info(f"Loaded sentiment model from {model_path}")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)

    def prepare_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert text to sequences for model input"""
        if not self.tokenizer:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded_sequences

    def build_model(self) -> Sequential:
        """Build and compile the sentiment analysis model"""
        model = Sequential([
            Embedding(self.vocab_size, 128, input_length=self.max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, texts: List[str], labels: List[int], validation_split: float = 0.2, epochs: int = 10) -> Dict:
        """Train the sentiment analysis model"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Prepare sequences
        X = self.prepare_sequences(processed_texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(
                'best_sentiment_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks
        )
        
        return history.history

    def predict_sentiment_deep(self, text: str) -> float:
        """Predict sentiment using deep learning model"""
        if not self.model:
            raise ValueError("Model not trained or loaded")
        
        processed_text = self.preprocess_text(text)
        sequence = self.prepare_sequences([processed_text])
        prediction = self.model.predict(sequence)[0][0]
        return float(prediction)

    def predict_sentiment_vader(self, text: str) -> float:
        """Predict sentiment using VADER"""
        sentiment = self.vader_analyzer.polarity_scores(text)
        return sentiment['compound']

    def predict_sentiment_transformer(self, text: str) -> float:
        """Predict sentiment using FinBERT"""
        inputs = self.transformer_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.transformer_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        return float(probabilities[0][1])  # Probability of positive sentiment

    def analyze_headlines(self, headlines: List[str], method: str = 'ensemble') -> List[Dict]:
        """
        Analyze sentiment of headlines using specified method
        method: 'deep', 'vader', 'transformer', or 'ensemble'
        """
        results = []
        
        for headline in headlines:
            if method == 'deep':
                sentiment = self.predict_sentiment_deep(headline)
            elif method == 'vader':
                sentiment = self.predict_sentiment_vader(headline)
            elif method == 'transformer':
                sentiment = self.predict_sentiment_transformer(headline)
            else:  # ensemble
                deep_sentiment = self.predict_sentiment_deep(headline)
                vader_sentiment = self.predict_sentiment_vader(headline)
                transformer_sentiment = self.predict_sentiment_transformer(headline)
                # Weighted ensemble
                sentiment = (
                    0.4 * transformer_sentiment +  # FinBERT specialized in financial text
                    0.3 * deep_sentiment +        # Custom trained model
                    0.3 * vader_sentiment         # Rule-based system
                )
            
            self.sentiment_history.append(sentiment)
            
            results.append({
                'headline': headline,
                'sentiment': sentiment,
                'classification': 'positive' if sentiment > 0 else 'negative'
            })
        
        return results

    def calculate_sentiment_trend(self) -> Dict[str, float]:
        """Calculate sentiment trend metrics"""
        if len(self.sentiment_history) < 2:
            return {'trend': 0, 'momentum': 0, 'volatility': 0}
            
        sentiments = np.array(list(self.sentiment_history))
        weights = self.trend_weights[-len(sentiments):]
        
        weighted_sentiment = np.average(sentiments, weights=weights)
        sentiment_momentum = np.mean(np.diff(sentiments[-20:]))  # Last 20 points
        sentiment_volatility = np.std(sentiments[-50:])  # Last 50 points
        
        return {
            'trend': weighted_sentiment,
            'momentum': sentiment_momentum,
            'volatility': sentiment_volatility
        }

    def analyze_market_sentiment(self, headlines: List[str], market_volatility: Optional[float] = None) -> Dict:
        """
        Analyze overall market sentiment with enhanced metrics
        
        Args:
            headlines: List of news headlines
            market_volatility: Current market volatility (if available)
        """
        sentiments = self.analyze_headlines(headlines)
        sentiment_scores = [s['sentiment'] for s in sentiments]
        
        # Calculate trend metrics
        trend_metrics = self.calculate_sentiment_trend()
        
        # Detect market regime
        if market_volatility is not None:
            market_regime = self.regime_detector.update(
                np.mean(sentiment_scores),
                market_volatility
            )
        else:
            market_regime = 'unknown'
        
        # Calculate confidence metrics
        sentiment_std = np.std(sentiment_scores)
        agreement_ratio = sum(1 for s in sentiment_scores if np.sign(s) == np.sign(np.mean(sentiment_scores))) / len(sentiment_scores)
        
        # Risk adjustment factor based on sentiment
        risk_adjustment = 1.0
        if sentiment_std > 0.5:  # High disagreement
            risk_adjustment *= 0.8  # Reduce risk
        if abs(trend_metrics['momentum']) > 0.1:  # Strong momentum
            risk_adjustment *= 1.2  # Increase risk
        
        market_sentiment = {
            'overall_score': np.mean(sentiment_scores),
            'sentiment_std': sentiment_std,
            'positive_ratio': sum(1 for s in sentiment_scores if s > 0) / len(sentiment_scores),
            'strong_signals': sum(1 for s in sentiment_scores if abs(s) > 0.5),
            'market_regime': market_regime,
            'trend_metrics': trend_metrics,
            'agreement_ratio': agreement_ratio,
            'risk_adjustment': risk_adjustment,
            'market_mood': 'bullish' if np.mean(sentiment_scores) > 0.1 else 
                          'bearish' if np.mean(sentiment_scores) < -0.1 else 'neutral'
        }
        
        return market_sentiment

def analyze_headlines(headlines: List[str]) -> List[float]:
    """
    Main function to analyze headlines (maintains compatibility with existing code)
    """
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_headlines(headlines)
    return [r['sentiment'] for r in results]

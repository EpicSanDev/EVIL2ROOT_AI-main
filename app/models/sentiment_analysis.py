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
        # Ajout d'une fenêtre plus courte pour détecter les changements récents
        self.short_sentiment_history = deque(maxlen=10)
        self.short_volatility_history = deque(maxlen=10)
        
    def update(self, sentiment: float, volatility: float) -> Dict[str, Union[str, float]]:
        """Update histories and detect market regime with more detailed analysis"""
        self.sentiment_history.append(sentiment)
        self.volatility_history.append(volatility)
        self.short_sentiment_history.append(sentiment)
        self.short_volatility_history.append(volatility)
        
        if len(self.sentiment_history) < 10:
            return {'regime': 'insufficient_data', 'confidence': 0.5, 'trend': 'neutral'}
        
        # Calcul des métriques sur différentes fenêtres temporelles
        avg_sentiment_long = np.mean(list(self.sentiment_history))
        avg_sentiment_short = np.mean(list(self.short_sentiment_history))
        sentiment_std_long = np.std(list(self.sentiment_history))
        sentiment_std_short = np.std(list(self.short_sentiment_history))
        
        avg_volatility_long = np.mean(list(self.volatility_history))
        avg_volatility_short = np.mean(list(self.short_volatility_history))
        
        # Détection de changement de régime (comparaison court terme vs long terme)
        sentiment_delta = avg_sentiment_short - avg_sentiment_long
        volatility_delta = avg_volatility_short - avg_volatility_long
        
        # Détection de tendance
        sentiment_trend = 'neutral'
        if len(self.sentiment_history) >= 20:
            recent_sentiment = list(self.sentiment_history)[-20:]
            # Utiliser une régression linéaire simple pour déterminer la tendance
            x = np.arange(len(recent_sentiment))
            slope = np.polyfit(x, recent_sentiment, 1)[0]
            
            if slope > 0.005:
                sentiment_trend = 'improving'
            elif slope < -0.005:
                sentiment_trend = 'deteriorating'
        
        # Détermination du régime avec confiance
        regime = 'normal'
        confidence = 0.5
        
        # Seuils de volatilité ajustés et basés sur des percentiles
        if avg_volatility_long > 0.02:  # Haute volatilité
            if avg_sentiment_long > 0.2:
                regime = 'volatile_bullish'
                confidence = min(0.5 + avg_sentiment_long + avg_volatility_long, 0.95)
            elif avg_sentiment_long < -0.2:
                regime = 'volatile_bearish'
                confidence = min(0.5 - avg_sentiment_long + avg_volatility_long, 0.95)
            else:
                regime = 'volatile_neutral'
                confidence = min(0.5 + avg_volatility_long, 0.9)
                
            # Détection de transition de régime
            if abs(sentiment_delta) > 0.15:
                if sentiment_delta > 0:
                    regime = 'volatile_transition_bullish'
                else:
                    regime = 'volatile_transition_bearish'
                confidence = min(0.5 + abs(sentiment_delta) + avg_volatility_short, 0.95)
        else:  # Volatilité normale
            if avg_sentiment_long > 0.2:
                regime = 'stable_bullish'
                confidence = min(0.5 + avg_sentiment_long, 0.9)
            elif avg_sentiment_long < -0.2:
                regime = 'stable_bearish'
                confidence = min(0.5 - avg_sentiment_long, 0.9)
            else:
                regime = 'stable_neutral'
                confidence = 0.6
                
            # Signes de transition imminente?
            if avg_volatility_short > 1.5 * avg_volatility_long:
                regime = f"{regime}_volatility_increasing"
                confidence = min(confidence + 0.1, 0.95)
        
        return {
            'regime': regime, 
            'confidence': confidence, 
            'trend': sentiment_trend,
            'sentiment_long': avg_sentiment_long,
            'sentiment_short': avg_sentiment_short,
            'volatility_long': avg_volatility_long,
            'volatility_short': avg_volatility_short,
            'delta': sentiment_delta
        }

class SentimentAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        self.tokenizer = None
        self.model = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.max_length = 100
        self.vocab_size = 10000
        self.regime_detector = MarketRegimeDetector()
        
        # Initialiser le modèle transformer pour le sentiment financier
        try:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            # Ajouter un modèle spécialisé pour les nouvelles économiques
            self.economic_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            self.economic_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
            logging.info("Loaded all transformer models successfully")
        except Exception as e:
            logging.error(f"Error loading transformer models: {e}")
            self.transformer_tokenizer = None
            self.transformer_model = None
            self.economic_tokenizer = None
            self.economic_model = None
        
        # Historique de sentiment pour l'analyse de tendance
        self.sentiment_history = deque(maxlen=100)
        # Fenêtres multiples pour l'analyse de tendance
        self.sentiment_windows = {
            'very_short': deque(maxlen=5),
            'short': deque(maxlen=20),
            'medium': deque(maxlen=50),
            'long': deque(maxlen=100)
        }
        # Poids exponentiels pour la tendance
        self.trend_weights = np.exp(np.linspace(-1, 0, 100))
        
        # Dictionnaires spécifiques au domaine financier
        self.bullish_terms = set(['bullish', 'uptrend', 'growth', 'rally', 'recovery', 'boom', 'uptick', 'outperform', 'beat', 'upgrade', 'raise', 'buy', 'strong', 'positive', 'optimistic', 'support'])
        self.bearish_terms = set(['bearish', 'downtrend', 'decline', 'fall', 'correction', 'recession', 'crash', 'downturn', 'underperform', 'miss', 'downgrade', 'cut', 'sell', 'weak', 'negative', 'pessimistic', 'resistance'])
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            logging.info(f"Loaded sentiment model from {model_path}")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data with improved handling"""
        if not text or not isinstance(text, str):
            return ""
            
        # Convertir en minuscules
        text = text.lower()
        
        # Conserver certains symboles financiers et caractères spéciaux pertinents
        text = re.sub(r'[^a-zA-Z\s\$\%\+\-]', '', text)
        
        # Tokeniser
        tokens = word_tokenize(text)
        
        # Supprimer les stopwords et lemmatiser
        stop_words = set(stopwords.words('english'))
        # Ne pas supprimer certains mots négatifs importants pour le sentiment
        important_negations = {'no', 'not', 'none', 'never', 'neither', 'nor', 'against'}
        stop_words = stop_words - important_negations
        
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
        """Predict sentiment using financial transformer model with improved handling"""
        if not self.transformer_tokenizer or not self.transformer_model:
            return self.predict_sentiment_vader(text)  # Fallback to VADER
        
        try:
            # Prétraiter le texte pour de meilleurs résultats
            processed_text = self.preprocess_text(text)
            
            # Si le texte est vide après prétraitement
            if not processed_text:
                return 0.0
                
            # Tokeniser le texte
            inputs = self.transformer_tokenizer(processed_text, return_tensors="pt", 
                                                truncation=True, max_length=512,
                                                padding=True)
            
            # Obtenir la prédiction
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
            # Convertir en score sur l'échelle [-1, 1]
            # FinBERT a 3 classes: négatif (0), neutre (1), positif (2)
            neg_prob = probabilities[0][0].item()
            neu_prob = probabilities[0][1].item()
            pos_prob = probabilities[0][2].item()
            
            # Calculer un score basé sur les probabilités
            sentiment_score = pos_prob - neg_prob
            
            # Détecter les termes spécifiques au secteur financier pour ajuster le score
            tokens = set(processed_text.split())
            bullish_count = len(tokens.intersection(self.bullish_terms))
            bearish_count = len(tokens.intersection(self.bearish_terms))
            
            # Ajuster le score en fonction des termes financiers spécifiques
            domain_adjustment = (bullish_count - bearish_count) * 0.05
            adjusted_score = sentiment_score + domain_adjustment
            
            # Limiter le score à [-1, 1]
            final_score = max(min(adjusted_score, 1.0), -1.0)
            
            return final_score
            
        except Exception as e:
            logging.error(f"Error in transformer sentiment prediction: {e}")
            return self.predict_sentiment_vader(text)  # Fallback à VADER en cas d'erreur
    
    def predict_sentiment_economic(self, text: str) -> float:
        """Predict sentiment specifically for economic news"""
        if not self.economic_tokenizer or not self.economic_model:
            return self.predict_sentiment_transformer(text)  # Fallback
        
        try:
            inputs = self.economic_tokenizer(text, return_tensors="pt", 
                                           truncation=True, max_length=512,
                                           padding=True)
            
            with torch.no_grad():
                outputs = self.economic_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
            # Finbert-tone a 3 classes: négatif (0), neutre (1), positif (2)
            neg_prob = probabilities[0][0].item()
            neu_prob = probabilities[0][1].item()
            pos_prob = probabilities[0][2].item()
            
            sentiment_score = pos_prob - neg_prob
            return max(min(sentiment_score, 1.0), -1.0)
            
        except Exception as e:
            logging.error(f"Error in economic sentiment prediction: {e}")
            return self.predict_sentiment_transformer(text)

    def analyze_headlines(self, headlines: List[str], method: str = 'ensemble') -> List[Dict]:
        """
        Analyze a list of headlines with improved ensemble method and detailed results
        """
        results = []
        
        if not headlines:
            return results
            
        # Détecter les thèmes des gros titres pour un meilleur contexte
        themes = self._detect_themes(headlines)
        
        for headline in headlines:
            if not headline.strip():
                continue
                
            # Tenter de classifier le type de nouvelle
            news_type = self._classify_news_type(headline)
            
            # Choisir la méthode appropriée en fonction du type de nouvelle
            if news_type == 'economic' and method == 'ensemble':
                vader_score = self.predict_sentiment_vader(headline)
                transformer_score = self.predict_sentiment_transformer(headline)
                economic_score = self.predict_sentiment_economic(headline)
                
                # Pondérer davantage le score du modèle économique
                ensemble_score = (vader_score * 0.2 + transformer_score * 0.3 + economic_score * 0.5)
                
                # Détecter la magnitude du sentiment (force)
                magnitude = (abs(vader_score) + abs(transformer_score) + abs(economic_score)) / 3
                if magnitude < 0.3:
                    strength = "weak"
                elif magnitude < 0.6:
                    strength = "moderate"
                else:
                    strength = "strong"
                
                result = {
                    'headline': headline,
                    'sentiment': ensemble_score,
                    'magnitude': magnitude,
                    'strength': strength,
                    'type': news_type,
                    'details': {
                        'vader': vader_score,
                        'transformer': transformer_score,
                        'economic': economic_score
                    }
                }
                
            elif method.lower() == 'vader':
                score = self.predict_sentiment_vader(headline)
                result = {'headline': headline, 'sentiment': score, 'type': news_type}
                
            elif method.lower() == 'transformer':
                score = self.predict_sentiment_transformer(headline)
                result = {'headline': headline, 'sentiment': score, 'type': news_type}
                
            else:  # ensemble par défaut
                vader_score = self.predict_sentiment_vader(headline)
                transformer_score = self.predict_sentiment_transformer(headline)
                
                # Moyenne pondérée
                ensemble_score = (vader_score * 0.35 + transformer_score * 0.65)
                
                # Détecter la magnitude du sentiment (force)
                magnitude = (abs(vader_score) + abs(transformer_score)) / 2
                if magnitude < 0.3:
                    strength = "weak"
                elif magnitude < 0.6:
                    strength = "moderate"
                else:
                    strength = "strong"
                
                result = {
                    'headline': headline,
                    'sentiment': ensemble_score,
                    'magnitude': magnitude,
                    'strength': strength,
                    'type': news_type,
                    'details': {
                        'vader': vader_score,
                        'transformer': transformer_score
                    }
                }
            
            results.append(result)
            
            # Mettre à jour l'historique
            self.sentiment_history.append(result['sentiment'])
            for window_name, window in self.sentiment_windows.items():
                window.append(result['sentiment'])
        
        return results
    
    def _classify_news_type(self, headline: str) -> str:
        """Classify the type of financial news"""
        headline_lower = headline.lower()
        
        economic_terms = ['gdp', 'inflation', 'economy', 'economic', 'fed', 'central bank', 
                          'interest rate', 'unemployment', 'jobs', 'labor', 'recession',
                          'growth', 'treasury', 'deficit', 'debt']
        
        earnings_terms = ['earnings', 'profit', 'revenue', 'eps', 'quarter', 'guidance',
                         'outlook', 'forecast', 'beat', 'miss', 'estimate', 'expectation']
        
        merger_terms = ['merger', 'acquisition', 'takeover', 'buy out', 'purchase',
                       'deal', 'acquire', 'bid', 'offer', 'sell', 'spin off', 'divest']
        
        tech_terms = ['tech', 'technology', 'ai', 'artificial intelligence', 'software',
                     'app', 'digital', 'cloud', 'cyber', 'data', 'internet', 'platform']
        
        # Comptage simple pour la classification
        econ_count = sum(1 for term in economic_terms if term in headline_lower)
        earnings_count = sum(1 for term in earnings_terms if term in headline_lower)
        merger_count = sum(1 for term in merger_terms if term in headline_lower)
        tech_count = sum(1 for term in tech_terms if term in headline_lower)
        
        # Déterminer le type en fonction du score le plus élevé
        counts = {
            'economic': econ_count,
            'earnings': earnings_count,
            'merger': merger_count,
            'tech': tech_count
        }
        
        max_type = max(counts, key=counts.get)
        
        # Vérifier s'il y a suffisamment de preuves pour classifier
        if counts[max_type] > 0:
            return max_type
        else:
            return 'general'
    
    def _detect_themes(self, headlines: List[str]) -> Dict[str, int]:
        """Detect common themes across multiple headlines"""
        # Liste simple de thèmes financiers à détecter
        financial_themes = {
            'interest_rates': ['interest rate', 'rate hike', 'rate cut', 'fed', 'central bank', 'fomc'],
            'inflation': ['inflation', 'cpi', 'price index', 'price increase', 'cost of living'],
            'recession': ['recession', 'economic downturn', 'economic slowdown', 'contraction'],
            'growth': ['growth', 'expansion', 'gdp', 'economic growth', 'recovery'],
            'earnings': ['earnings', 'profit', 'revenue', 'quarterly report', 'financial results'],
            'crypto': ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'token', 'coin'],
            'stocks': ['stock', 'share', 'equity', 'market', 'index', 'etf', 'fund'],
            'commodities': ['oil', 'gold', 'silver', 'commodity', 'crude', 'natural gas'],
        }
        
        # Compter les occurrences de chaque thème
        theme_counts = {theme: 0 for theme in financial_themes}
        
        for headline in headlines:
            headline_lower = headline.lower()
            for theme, keywords in financial_themes.items():
                if any(keyword in headline_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Filtrer pour n'inclure que les thèmes présents
        return {theme: count for theme, count in theme_counts.items() if count > 0}

    def calculate_sentiment_trend(self) -> Dict[str, float]:
        """Calculate sentiment trends across multiple time windows with improved metrics"""
        trends = {}
        
        # Si nous n'avons pas assez de données, renvoyer des valeurs neutres
        if len(self.sentiment_history) < 5:
            return {
                'current': 0,
                'trend_direction': 'neutral',
                'trend_strength': 0,
                'volatility': 0,
                'momentum': 0
            }
        
        # Calculer les moyennes pour différentes fenêtres temporelles
        for window_name, window in self.sentiment_windows.items():
            if len(window) > 0:
                trends[f'avg_{window_name}'] = np.mean(list(window))
        
        # Calculer le sentiment actuel (moyenne des 5 derniers)
        current_sentiment = np.mean(list(self.sentiment_windows['very_short']))
        trends['current'] = current_sentiment
        
        # Calculer le sentiment récent vs sentiment à moyen terme
        if 'avg_short' in trends and 'avg_medium' in trends:
            short_med_delta = trends['avg_short'] - trends['avg_medium']
            trends['short_med_delta'] = short_med_delta
            
            # Direction et force de la tendance
            if abs(short_med_delta) < 0.05:
                trends['trend_direction'] = 'neutral'
                trends['trend_strength'] = abs(short_med_delta) * 10  # Scale to 0-1
            elif short_med_delta > 0:
                trends['trend_direction'] = 'improving'
                trends['trend_strength'] = min(short_med_delta * 5, 1.0)  # Scale to 0-1
            else:
                trends['trend_direction'] = 'deteriorating'
                trends['trend_strength'] = min(abs(short_med_delta) * 5, 1.0)  # Scale to 0-1
        
        # Calculer la volatilité du sentiment (écart-type)
        if len(self.sentiment_history) >= 10:
            trends['volatility'] = np.std(list(self.sentiment_history)[-20:])
        else:
            trends['volatility'] = 0
        
        # Calculer le momentum (changement absolu récent)
        if len(self.sentiment_history) >= 10:
            oldest = np.mean(list(self.sentiment_history)[-20:-10])
            newest = np.mean(list(self.sentiment_history)[-10:])
            trends['momentum'] = newest - oldest
        else:
            trends['momentum'] = 0
            
        # Analyse exponentielle pondérée pour lisser la tendance
        if len(self.sentiment_history) >= 50:
            sentiment_array = np.array(list(self.sentiment_history)[-50:])
            weights = np.exp(np.linspace(-2, 0, len(sentiment_array)))
            weights = weights / weights.sum()  # Normalisation
            trends['exp_weighted_avg'] = np.sum(sentiment_array * weights)
        
        return trends

    def analyze_market_sentiment(self, headlines: List[str], market_volatility: Optional[float] = None) -> Dict:
        """
        Analyze overall market sentiment with improved contextual analysis
        """
        if not headlines:
            return {'score': 0, 'regime': 'neutral', 'confidence': 0.5}
            
        # Analyse détaillée des titres
        headline_analysis = self.analyze_headlines(headlines, method='ensemble')
        
        if not headline_analysis:
            return {'score': 0, 'regime': 'neutral', 'confidence': 0.5}
        
        # Extraire les scores de sentiment
        sentiment_scores = [item['sentiment'] for item in headline_analysis]
        
        # Pondérer davantage les nouvelles économiques et financières
        weighted_scores = []
        total_weight = 0
        
        for item in headline_analysis:
            weight = 1.0  # Poids par défaut
            
            # Augmentation de la pondération pour des types de nouvelles spécifiques
            if item['type'] == 'economic':
                weight = 1.5
            elif item['type'] == 'earnings':
                weight = 1.3
            elif item['type'] == 'merger':
                weight = 1.2
            
            # Augmentation de la pondération pour des sentiments forts
            if 'strength' in item and item['strength'] == 'strong':
                weight *= 1.3
            elif 'strength' in item and item['strength'] == 'moderate':
                weight *= 1.1
            
            weighted_scores.append(item['sentiment'] * weight)
            total_weight += weight
        
        # Calculer le score de sentiment global pondéré
        if total_weight > 0:
            overall_sentiment = sum(weighted_scores) / total_weight
        else:
            overall_sentiment = 0
        
        # Mettre à jour l'historique et le détecteur de régime
        self.sentiment_history.append(overall_sentiment)
        
        # Définir la volatilité du marché ou l'estimer à partir du sentiment
        if market_volatility is None:
            if len(sentiment_scores) > 1:
                market_volatility = np.std(sentiment_scores)
            else:
                market_volatility = 0.01  # Valeur par défaut faible
        
        # Obtenir le régime de marché actuel avec information enrichie
        regime_info = self.regime_detector.update(overall_sentiment, market_volatility)
        
        # Calculer les tendances du sentiment
        trend_info = self.calculate_sentiment_trend()
        
        # Calculer la dispersion du sentiment (accord/désaccord des sources)
        sentiment_dispersion = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
        
        # Préparer les informations sur la distribution du sentiment
        sentiment_distribution = {
            'positive': len([s for s in sentiment_scores if s > 0.2]),
            'neutral': len([s for s in sentiment_scores if -0.2 <= s <= 0.2]),
            'negative': len([s for s in sentiment_scores if s < -0.2])
        }
        
        # Classifier les titres par type pour une analyse par secteur
        headlines_by_type = {}
        for item in headline_analysis:
            news_type = item.get('type', 'general')
            if news_type not in headlines_by_type:
                headlines_by_type[news_type] = []
            headlines_by_type[news_type].append(item)
        
        # Calculer le sentiment par type de nouvelle
        sentiment_by_type = {}
        for news_type, items in headlines_by_type.items():
            if items:
                sentiment_by_type[news_type] = np.mean([item['sentiment'] for item in items])
        
        # Analyse thématique
        themes = self._detect_themes(headlines)
        
        # Construire la réponse complète
        result = {
            'score': overall_sentiment,
            'magnitude': np.mean([abs(s) for s in sentiment_scores]),
            'regime': regime_info['regime'],
            'confidence': regime_info['confidence'],
            'trend': trend_info,
            'dispersion': sentiment_dispersion,
            'distribution': sentiment_distribution,
            'by_type': sentiment_by_type,
            'themes': themes,
            'regime_details': regime_info,
            'headline_count': len(headline_analysis)
        }
        
        return result

def analyze_headlines(headlines: List[str]) -> List[float]:
    """
    Main function to analyze headlines (maintains compatibility with existing code)
    """
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_headlines(headlines)
    return [r['sentiment'] for r in results]

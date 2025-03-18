import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, Embedding, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import logging
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import BertTokenizer, TFBertForSequenceClassification
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import deque, Counter
import requests
from bs4 import BeautifulSoup
import tweepy
import time
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Télécharger les ressources NLTK si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FinancialNewsProcessor:
    """
    Classe pour collecter et prétraiter les actualités financières de diverses sources.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialise le processeur d'actualités financières.
        
        Args:
            api_keys (Dict[str, str]): Clés API pour différentes sources d'actualités.
        """
        self.api_keys = api_keys or {}
        self.logger = logging.getLogger(__name__)
        self.sources = {
            'twitter': self._setup_twitter_api() if 'twitter' in self.api_keys else None,
            'newsapi': self.api_keys.get('newsapi'),
            'alpha_vantage': self.api_keys.get('alpha_vantage'),
            'finnhub': self.api_keys.get('finnhub')
        }
        
        # Pour le stockage des actualités récentes
        self.recent_news = {}
        self.latest_update = {}
        
        # Pour le prétraitement du texte
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _setup_twitter_api(self) -> Optional[tweepy.API]:
        """
        Configure l'API Twitter.
        
        Returns:
            tweepy.API ou None en cas d'échec.
        """
        try:
            auth = tweepy.OAuthHandler(
                self.api_keys['twitter_consumer_key'],
                self.api_keys['twitter_consumer_secret']
            )
            auth.set_access_token(
                self.api_keys['twitter_access_token'],
                self.api_keys['twitter_access_token_secret']
            )
            api = tweepy.API(auth, wait_on_rate_limit=True)
            return api
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de l'API Twitter: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Nettoie et prétraite le texte.
        
        Args:
            text (str): Texte à nettoyer.
            
        Returns:
            str: Texte nettoyé.
        """
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Supprimer les mentions et hashtags Twitter
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Supprimer les caractères non alphanumériques
        text = re.sub(r'[^\w\s]', '', text)
        
        # Supprimer les chiffres
        text = re.sub(r'\d+', '', text)
        
        # Tokenisation
        tokens = word_tokenize(text)
        
        # Supprimer les stopwords et lemmatiser
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
        
        # Rejoindre les tokens
        return ' '.join(tokens)
    
    def get_news_from_twitter(self, query: str, count: int = 100, days_back: int = 1) -> List[Dict]:
        """
        Récupère les tweets pertinents pour une requête donnée.
        
        Args:
            query (str): Terme de recherche.
            count (int): Nombre de tweets à récupérer.
            days_back (int): Nombre de jours dans le passé pour la recherche.
            
        Returns:
            List[Dict]: Liste de tweets avec leur score de sentiment.
        """
        if not self.sources['twitter']:
            self.logger.warning("API Twitter non configurée.")
            return []
        
        try:
            # Construire la requête de recherche
            search_query = f"{query} -filter:retweets"
            
            # Définir la date limite
            since_date = datetime.now() - timedelta(days=days_back)
            
            # Récupérer les tweets
            tweets = []
            for tweet in tweepy.Cursor(
                self.sources['twitter'].search_tweets,
                q=search_query,
                lang="en",
                since_id=since_date.strftime('%Y-%m-%d'),
                tweet_mode='extended'
            ).items(count):
                # Extraire le texte
                if hasattr(tweet, 'full_text'):
                    text = tweet.full_text
                else:
                    text = tweet.text
                
                # Nettoyer le texte
                clean_text = self._clean_text(text)
                
                tweets.append({
                    'text': clean_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'source': 'twitter'
                })
            
            return tweets
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des tweets: {str(e)}")
            return []
    
    def get_news_from_newsapi(self, query: str, days_back: int = 1) -> List[Dict]:
        """
        Récupère les actualités à partir de l'API News.
        
        Args:
            query (str): Terme de recherche.
            days_back (int): Nombre de jours dans le passé pour la recherche.
            
        Returns:
            List[Dict]: Liste d'articles d'actualité.
        """
        if not self.sources['newsapi']:
            self.logger.warning("Clé API NewsAPI non configurée.")
            return []
        
        try:
            # Construire l'URL
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&"
                f"from={from_date}&"
                f"sortBy=publishedAt&"
                f"language=en&"
                f"apiKey={self.sources['newsapi']}"
            )
            
            # Faire la requête
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Traiter les résultats
            articles = []
            for article in data.get('articles', []):
                # Nettoyer le texte
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                full_text = f"{title} {description} {content}"
                clean_text = self._clean_text(full_text)
                
                articles.append({
                    'text': clean_text,
                    'title': title,
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'published_at': article.get('publishedAt')
                })
            
            return articles
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des actualités de NewsAPI: {str(e)}")
            return []
    
    def get_news_from_finnhub(self, symbol: str, days_back: int = 1) -> List[Dict]:
        """
        Récupère les actualités à partir de l'API Finnhub.
        
        Args:
            symbol (str): Symbole de l'entreprise.
            days_back (int): Nombre de jours dans le passé pour la recherche.
            
        Returns:
            List[Dict]: Liste d'articles d'actualité.
        """
        if not self.sources['finnhub']:
            self.logger.warning("Clé API Finnhub non configurée.")
            return []
        
        try:
            # Construire l'URL
            from_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
            to_timestamp = int(datetime.now().timestamp())
            
            url = (
                f"https://finnhub.io/api/v1/company-news?"
                f"symbol={symbol}&"
                f"from={datetime.fromtimestamp(from_timestamp).strftime('%Y-%m-%d')}&"
                f"to={datetime.fromtimestamp(to_timestamp).strftime('%Y-%m-%d')}&"
                f"token={self.sources['finnhub']}"
            )
            
            # Faire la requête
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Traiter les résultats
            articles = []
            for article in data:
                # Nettoyer le texte
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                full_text = f"{headline} {summary}"
                clean_text = self._clean_text(full_text)
                
                articles.append({
                    'text': clean_text,
                    'title': headline,
                    'url': article.get('url', ''),
                    'source': article.get('source', 'Finnhub'),
                    'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat()
                })
            
            return articles
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des actualités de Finnhub: {str(e)}")
            return []
    
    def get_all_news(self, query: str, symbol: str = None, days_back: int = 1) -> List[Dict]:
        """
        Récupère les actualités de toutes les sources disponibles.
        
        Args:
            query (str): Terme de recherche général.
            symbol (str): Symbole de l'entreprise pour les sources spécifiques.
            days_back (int): Nombre de jours dans le passé pour la recherche.
            
        Returns:
            List[Dict]: Liste combinée d'articles d'actualité de toutes les sources.
        """
        all_news = []
        
        # Twitter
        twitter_news = self.get_news_from_twitter(query, days_back=days_back)
        all_news.extend(twitter_news)
        
        # NewsAPI
        newsapi_news = self.get_news_from_newsapi(query, days_back=days_back)
        all_news.extend(newsapi_news)
        
        # Finnhub (nécessite un symbole)
        if symbol:
            finnhub_news = self.get_news_from_finnhub(symbol, days_back=days_back)
            all_news.extend(finnhub_news)
        
        # Mettre à jour le cache
        cache_key = f"{query}_{symbol or ''}"
        self.recent_news[cache_key] = all_news
        self.latest_update[cache_key] = datetime.now()
        
        return all_news
    
    def get_cached_news(self, query: str, symbol: str = None, max_age_hours: int = 1) -> List[Dict]:
        """
        Récupère les actualités mises en cache ou actualise si nécessaire.
        
        Args:
            query (str): Terme de recherche général.
            symbol (str): Symbole de l'entreprise pour les sources spécifiques.
            max_age_hours (int): Âge maximum du cache en heures.
            
        Returns:
            List[Dict]: Liste d'articles d'actualité.
        """
        cache_key = f"{query}_{symbol or ''}"
        
        # Vérifier si nous avons des actualités en cache et si elles sont récentes
        if (
            cache_key in self.recent_news and 
            cache_key in self.latest_update and 
            (datetime.now() - self.latest_update[cache_key]).total_seconds() < max_age_hours * 3600
        ):
            return self.recent_news[cache_key]
        
        # Sinon, récupérer de nouvelles actualités
        return self.get_all_news(query, symbol)

class AdvancedSentimentModel:
    """
    Modèle d'analyse de sentiment avancé pour les textes financiers.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le modèle d'analyse de sentiment.
        
        Args:
            model_path (str, optional): Chemin vers un modèle préentraîné.
        """
        self.logger = logging.getLogger(__name__)
        
        # Pour le prétraitement du texte
        self.max_sequence_length = 128
        self.tokenizer = None
        
        # Chargement des modèles Transformers pré-entraînés
        try:
            # Modèle FinBERT pour le sentiment financier
            self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            
            # Pipeline Hugging Face pour les prédictions
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # VADER pour l'analyse de sentiment basée sur les règles
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Si un modèle personnalisé est fourni, le charger
            if model_path and os.path.exists(model_path):
                self.custom_model = tf.keras.models.load_model(model_path)
                
                # Charger le tokenizer associé
                tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer.pickle')
                if os.path.exists(tokenizer_path):
                    with open(tokenizer_path, 'rb') as handle:
                        self.tokenizer = pickle.load(handle)
            else:
                self.custom_model = None
            
            self.logger.info("Modèles d'analyse de sentiment chargés avec succès")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des modèles: {str(e)}")
            self.finbert_model = None
            self.finbert_tokenizer = None
            self.sentiment_pipeline = None
            self.custom_model = None
    
    def _preprocess_text(self, text: str) -> np.ndarray:
        """
        Prétraite le texte pour l'analyse par le modèle personnalisé.
        
        Args:
            text (str): Texte à prétraiter.
            
        Returns:
            np.ndarray: Texte tokenisé et paddé.
        """
        if not self.tokenizer:
            self.logger.warning("Tokenizer non disponible, utilisation du modèle par défaut.")
            return None
        
        # Tokeniser et padder le texte
        sequences = self.tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        return padded_sequences
    
    def predict_sentiment_with_custom_model(self, text: str) -> Dict[str, float]:
        """
        Prédit le sentiment avec le modèle personnalisé.
        
        Args:
            text (str): Texte à analyser.
            
        Returns:
            Dict[str, float]: Scores de sentiment.
        """
        if not self.custom_model or not self.tokenizer:
            self.logger.warning("Modèle personnalisé non disponible.")
            return None
        
        # Prétraiter le texte
        preprocessed_text = self._preprocess_text(text)
        
        if preprocessed_text is None:
            return None
        
        # Prédire le sentiment
        predictions = self.custom_model.predict(preprocessed_text)
        
        # Pour un modèle à 3 classes (négatif, neutre, positif)
        sentiment_scores = {
            'negative': float(predictions[0][0]),
            'neutral': float(predictions[0][1]),
            'positive': float(predictions[0][2])
        }
        
        return sentiment_scores
    
    def predict_sentiment_with_finbert(self, text: str) -> Dict[str, float]:
        """
        Prédit le sentiment avec le modèle FinBERT.
        
        Args:
            text (str): Texte à analyser.
            
        Returns:
            Dict[str, float]: Scores de sentiment.
        """
        if not self.sentiment_pipeline:
            self.logger.warning("Pipeline FinBERT non disponible.")
            return None
        
        try:
            # Limiter la longueur du texte si nécessaire
            if len(text) > 512:
                text = text[:512]
            
            # Prédire le sentiment
            result = self.sentiment_pipeline(text)[0]
            
            # Extraire et normaliser la classe et le score
            label = result['label'].lower()
            score = result['score']
            
            sentiment_scores = {
                'negative': score if label == 'negative' else 0.0,
                'neutral': score if label == 'neutral' else 0.0,
                'positive': score if label == 'positive' else 0.0
            }
            
            return sentiment_scores
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction avec FinBERT: {str(e)}")
            return None
    
    def predict_sentiment_with_vader(self, text: str) -> Dict[str, float]:
        """
        Prédit le sentiment avec VADER.
        
        Args:
            text (str): Texte à analyser.
            
        Returns:
            Dict[str, float]: Scores de sentiment.
        """
        if not self.vader_analyzer:
            self.logger.warning("Analyseur VADER non disponible.")
            return None
        
        try:
            # Obtenir les scores de sentiment
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Normaliser les scores
            sentiment_scores = {
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'positive': scores['pos'],
                'compound': scores['compound']
            }
            
            return sentiment_scores
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction avec VADER: {str(e)}")
            return None
    
    def predict_ensemble_sentiment(self, text: str) -> Dict[str, float]:
        """
        Prédit le sentiment en combinant plusieurs modèles.
        
        Args:
            text (str): Texte à analyser.
            
        Returns:
            Dict[str, float]: Scores de sentiment combinés.
        """
        # Obtenir les prédictions de chaque modèle
        vader_scores = self.predict_sentiment_with_vader(text)
        finbert_scores = self.predict_sentiment_with_finbert(text)
        custom_scores = self.predict_sentiment_with_custom_model(text)
        
        # Initialiser les scores combinés
        combined_scores = {
            'negative': 0.0,
            'neutral': 0.0,
            'positive': 0.0,
            'compound': 0.0
        }
        
        # Nombre de modèles utilisés
        num_models = 0
        
        # Ajouter les scores de VADER
        if vader_scores:
            for key in ['negative', 'neutral', 'positive']:
                combined_scores[key] += vader_scores[key]
            combined_scores['compound'] += vader_scores['compound']
            num_models += 1
        
        # Ajouter les scores de FinBERT
        if finbert_scores:
            for key in ['negative', 'neutral', 'positive']:
                combined_scores[key] += finbert_scores[key]
            num_models += 1
        
        # Ajouter les scores du modèle personnalisé
        if custom_scores:
            for key in ['negative', 'neutral', 'positive']:
                combined_scores[key] += custom_scores[key]
            num_models += 1
        
        # Calculer la moyenne
        if num_models > 0:
            for key in combined_scores:
                combined_scores[key] /= num_models
        
        return combined_scores
    
    def predict_batch_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Prédit le sentiment pour un lot de textes.
        
        Args:
            texts (List[str]): Liste de textes à analyser.
            
        Returns:
            List[Dict[str, float]]: Liste de scores de sentiment.
        """
        results = []
        
        for text in texts:
            sentiment = self.predict_ensemble_sentiment(text)
            results.append(sentiment)
        
        return results
    
    def train_custom_model(self, texts: List[str], labels: List[int], 
                          validation_split: float = 0.2, epochs: int = 10, 
                          batch_size: int = 32, save_path: str = None) -> Dict:
        """
        Entraîne un modèle personnalisé sur des données étiquetées.
        
        Args:
            texts (List[str]): Textes d'entraînement.
            labels (List[int]): Étiquettes (0: négatif, 1: neutre, 2: positif).
            validation_split (float): Fraction des données pour la validation.
            epochs (int): Nombre d'époques d'entraînement.
            batch_size (int): Taille des lots pour l'entraînement.
            save_path (str): Chemin pour sauvegarder le modèle.
            
        Returns:
            Dict: Historique d'entraînement.
        """
        # Vérifier les données
        if len(texts) != len(labels):
            raise ValueError("Le nombre de textes et d'étiquettes doit être identique.")
        
        # Convertir les étiquettes en catégories
        num_classes = len(set(labels))
        y = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        # Créer et entraîner le tokenizer
        self.tokenizer = Tokenizer(num_words=10000)
        self.tokenizer.fit_on_texts(texts)
        
        # Tokeniser et padder les textes
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Diviser les données en entraînement et validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Construire le modèle
        embedding_dim = 128
        
        model = tf.keras.Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index) + 1, 
                      output_dim=embedding_dim, 
                      input_length=self.max_sequence_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_loss',
                    save_best_only=True
                )
            )
        
        # Entraîner le modèle
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Sauvegarder le modèle et le tokenizer
        if save_path:
            model.save(save_path)
            tokenizer_path = os.path.join(os.path.dirname(save_path), 'tokenizer.pickle')
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.custom_model = model
        
        return history.history
    
    def evaluate_custom_model(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Évalue le modèle personnalisé sur des données de test.
        
        Args:
            texts (List[str]): Textes de test.
            labels (List[int]): Étiquettes (0: négatif, 1: neutre, 2: positif).
            
        Returns:
            Dict[str, float]: Métriques d'évaluation.
        """
        if not self.custom_model or not self.tokenizer:
            self.logger.warning("Modèle personnalisé non disponible.")
            return None
        
        # Tokeniser et padder les textes
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Convertir les étiquettes en catégories
        num_classes = len(set(labels))
        y = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        # Évaluer le modèle
        loss, accuracy = self.custom_model.evaluate(X, y)
        
        # Prédire les probabilités
        y_pred_proba = self.custom_model.predict(X)
        
        # Prédire les classes
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y, axis=1)
        
        # Calculer les métriques
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        } 
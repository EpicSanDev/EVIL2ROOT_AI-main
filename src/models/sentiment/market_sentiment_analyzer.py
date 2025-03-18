import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
import os
import json
from collections import deque, Counter
import pickle
import time
from .advanced_sentiment_analyzer import AdvancedSentimentModel, FinancialNewsProcessor

class MarketSentimentAnalyzer:
    """
    Analyse le sentiment des marchés financiers en combinant l'analyse technique et l'analyse de sentiment.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None, model_path: Optional[str] = None,
                 sentiment_window: int = 14, volatility_window: int = 20,
                 sentiment_threshold: float = 0.2, volatility_threshold: float = 0.015):
        """
        Initialise l'analyseur de sentiment de marché.
        
        Args:
            api_keys (Dict[str, str]): Clés API pour les sources de données.
            model_path (str): Chemin vers un modèle personnalisé.
            sentiment_window (int): Taille de la fenêtre pour le sentiment.
            volatility_window (int): Taille de la fenêtre pour la volatilité.
            sentiment_threshold (float): Seuil pour les changements de sentiment.
            volatility_threshold (float): Seuil pour les changements de volatilité.
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les modèles et processeurs
        self.news_processor = FinancialNewsProcessor(api_keys)
        self.sentiment_model = AdvancedSentimentModel(model_path)
        
        # Paramètres de configuration
        self.sentiment_window = sentiment_window
        self.volatility_window = volatility_window
        self.sentiment_threshold = sentiment_threshold
        self.volatility_threshold = volatility_threshold
        
        # Historiques pour le suivi
        self.sentiment_history = deque(maxlen=100)  # Historique plus long pour l'analyse
        self.rolling_sentiment = deque(maxlen=sentiment_window)
        self.volatility_history = deque(maxlen=volatility_window)
        
        # Cache des dernières analyses par symbole
        self.last_analysis = {}
        self.last_update_time = {}
    
    def _calculate_volatility(self, prices: pd.Series) -> float:
        """
        Calcule la volatilité à partir d'une série de prix.
        
        Args:
            prices (pd.Series): Série de prix.
            
        Returns:
            float: Volatilité (écart-type des rendements).
        """
        # Calculer les rendements logarithmiques
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculer l'écart-type des rendements (volatilité)
        volatility = log_returns.std()
        
        return volatility
    
    def _normalize_sentiment_score(self, sentiment: Dict[str, float]) -> float:
        """
        Normalise un score de sentiment en une valeur entre -1 et 1.
        
        Args:
            sentiment (Dict[str, float]): Scores de sentiment.
            
        Returns:
            float: Score de sentiment normalisé entre -1 et 1.
        """
        if 'compound' in sentiment:
            # Si nous avons un score composé (VADER), l'utiliser directement
            return sentiment['compound']
        
        # Sinon, calculer un score composite à partir des composantes
        if 'positive' in sentiment and 'negative' in sentiment:
            positive = sentiment['positive']
            negative = sentiment['negative']
            
            # Score = positive - negative, normalisé entre -1 et 1
            return positive - negative
        
        return 0.0
    
    def _get_weighted_sentiment(self, sentiment_scores: List[Dict[str, float]], 
                               metadata: List[Dict] = None) -> Dict[str, float]:
        """
        Calcule un sentiment pondéré pour un ensemble de scores.
        
        Args:
            sentiment_scores (List[Dict[str, float]]): Liste de scores de sentiment.
            metadata (List[Dict]): Métadonnées pour la pondération (popularité, etc.).
            
        Returns:
            Dict[str, float]: Score de sentiment pondéré.
        """
        if not sentiment_scores:
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        # Initialiser les scores combinés
        combined = {
            'compound': 0.0,
            'positive': 0.0,
            'neutral': 0.0,
            'negative': 0.0
        }
        
        # Initialiser les poids pour chaque source
        weights = []
        
        # Calculer les poids en fonction des métadonnées
        if metadata:
            for meta in metadata:
                weight = 1.0  # Poids par défaut
                
                # Utiliser la source pour ajuster le poids
                source = meta.get('source', '').lower()
                if 'twitter' in source:
                    retweets = meta.get('retweet_count', 0)
                    favorites = meta.get('favorite_count', 0)
                    
                    # Plus de retweets/favoris = plus important
                    weight = 1.0 + 0.1 * np.log1p(retweets + favorites)
                elif 'news' in source or 'finnhub' in source:
                    # Les nouvelles économiques ont un poids légèrement plus élevé
                    weight = 1.5
                
                weights.append(weight)
        else:
            # Si pas de métadonnées, poids égaux
            weights = [1.0] * len(sentiment_scores)
        
        # Normaliser les poids
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Calculer la moyenne pondérée
        for i, sentiment in enumerate(sentiment_scores):
            for key in combined:
                if key in sentiment:
                    combined[key] += sentiment[key] * weights[i]
        
        return combined
    
    def analyze_news_sentiment(self, query: str, symbol: str = None, 
                              days_back: int = 1, max_age_hours: int = 1) -> Dict:
        """
        Analyse le sentiment des actualités pour un symbole ou une requête donnée.
        
        Args:
            query (str): Terme de recherche général.
            symbol (str): Symbole de l'entreprise.
            days_back (int): Nombre de jours à remonter.
            max_age_hours (int): Âge maximum du cache en heures.
            
        Returns:
            Dict: Résultats de l'analyse de sentiment.
        """
        # Vérifier si nous avons des résultats récents en cache
        cache_key = f"{query}_{symbol or ''}"
        
        if (
            cache_key in self.last_analysis and 
            cache_key in self.last_update_time and 
            (datetime.now() - self.last_update_time[cache_key]).total_seconds() < max_age_hours * 3600
        ):
            return self.last_analysis[cache_key]
        
        # Récupérer les actualités
        news = self.news_processor.get_cached_news(query, symbol, max_age_hours)
        
        if not news:
            self.logger.warning(f"Aucune actualité trouvée pour {query} {symbol}")
            return {
                'sentiment': {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0},
                'normalized_sentiment': 0.0,
                'count': 0,
                'trend': 'neutral',
                'confidence': 0.5,
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }
        
        # Extraire les textes et les métadonnées
        texts = [item['text'] for item in news]
        metadata = news
        
        # Analyser le sentiment
        sentiment_scores = self.sentiment_model.predict_batch_sentiment(texts)
        
        # Calculer le sentiment pondéré
        weighted_sentiment = self._get_weighted_sentiment(sentiment_scores, metadata)
        
        # Normaliser le sentiment
        normalized_sentiment = self._normalize_sentiment_score(weighted_sentiment)
        
        # Mettre à jour l'historique
        self.rolling_sentiment.append(normalized_sentiment)
        self.sentiment_history.append(normalized_sentiment)
        
        # Calculer la tendance du sentiment
        trend = 'neutral'
        confidence = 0.5
        
        if len(self.sentiment_history) >= 5:
            recent_sentiment = list(self.sentiment_history)[-5:]
            
            # Calculer la tendance à l'aide d'une régression linéaire simple
            x = np.arange(len(recent_sentiment))
            slope = np.polyfit(x, recent_sentiment, 1)[0]
            
            if slope > 0.01:
                trend = 'improving'
                confidence = min(0.5 + abs(slope) * 10, 0.9)  # Plus la pente est raide, plus la confiance est élevée
            elif slope < -0.01:
                trend = 'deteriorating'
                confidence = min(0.5 + abs(slope) * 10, 0.9)
            
            # Vérifier les niveaux absolus également
            avg_sentiment = np.mean(recent_sentiment)
            
            if avg_sentiment > self.sentiment_threshold:
                trend = 'bullish' if trend == 'neutral' else f"{trend}_bullish"
                confidence = max(confidence, min(0.5 + avg_sentiment, 0.9))
            elif avg_sentiment < -self.sentiment_threshold:
                trend = 'bearish' if trend == 'neutral' else f"{trend}_bearish"
                confidence = max(confidence, min(0.5 - avg_sentiment, 0.9))
        
        # Compiler les résultats
        sources = Counter([item.get('source', 'unknown') for item in news])
        
        result = {
            'sentiment': weighted_sentiment,
            'normalized_sentiment': normalized_sentiment,
            'count': len(news),
            'trend': trend,
            'confidence': confidence,
            'sources': dict(sources),
            'timestamp': datetime.now().isoformat()
        }
        
        # Mettre à jour le cache
        self.last_analysis[cache_key] = result
        self.last_update_time[cache_key] = datetime.now()
        
        return result
    
    def analyze_price_sentiment(self, prices: pd.DataFrame, window: int = 14) -> Dict:
        """
        Analyse le sentiment basé sur les prix et la volatilité.
        
        Args:
            prices (pd.DataFrame): DataFrame avec les données de prix.
            window (int): Taille de la fenêtre pour les calculs.
            
        Returns:
            Dict: Résultats de l'analyse de sentiment.
        """
        if len(prices) < window:
            self.logger.warning(f"Données insuffisantes pour l'analyse de prix (minimum {window} points)")
            return {
                'volatility': 0.0,
                'price_trend': 'neutral',
                'price_momentum': 0.0,
                'price_sentiment': 0.0,
                'confidence': 0.5
            }
        
        # Calculer la volatilité
        close_prices = prices['Close'].iloc[-window:]
        volatility = self._calculate_volatility(close_prices)
        
        # Mettre à jour l'historique de volatilité
        self.volatility_history.append(volatility)
        
        # Calculer la tendance des prix
        price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        
        # Calculer le momentum (accélération des prix)
        momentum = 0.0
        if len(close_prices) >= window:
            first_half = close_prices.iloc[:window//2]
            second_half = close_prices.iloc[window//2:]
            
            first_change = (first_half.iloc[-1] - first_half.iloc[0]) / first_half.iloc[0]
            second_change = (second_half.iloc[-1] - second_half.iloc[0]) / second_half.iloc[0]
            
            momentum = second_change - first_change
        
        # Calculer le sentiment de prix (combinaison de tendance et volatilité)
        price_sentiment = 0.0
        if price_change > 0:
            # Sentiment positif, mais diminué par une volatilité élevée
            price_sentiment = price_change / (1 + volatility * 5)
        else:
            # Sentiment négatif, amplifié par une volatilité élevée
            price_sentiment = price_change * (1 + volatility * 3)
        
        # Déterminer la tendance des prix
        price_trend = 'neutral'
        if price_change > self.sentiment_threshold:
            price_trend = 'bullish'
        elif price_change < -self.sentiment_threshold:
            price_trend = 'bearish'
        
        # Modifier en fonction du momentum
        if abs(momentum) > 0.01:
            if momentum > 0:
                price_trend = f"{price_trend}_accelerating" if price_trend != 'neutral' else 'accelerating'
            else:
                price_trend = f"{price_trend}_decelerating" if price_trend != 'neutral' else 'decelerating'
        
        # Calculer la confiance
        confidence = min(0.5 + abs(price_change) * 5, 0.9)  # Plus le changement est grand, plus la confiance est élevée
        
        return {
            'volatility': float(volatility),
            'price_trend': price_trend,
            'price_momentum': float(momentum),
            'price_sentiment': float(price_sentiment),
            'confidence': float(confidence)
        }
    
    def detect_market_regime(self, price_data: pd.DataFrame = None, 
                            news_query: str = None, symbol: str = None) -> Dict:
        """
        Détecte le régime de marché actuel en combinant l'analyse de sentiment et l'analyse technique.
        
        Args:
            price_data (pd.DataFrame): Données de prix pour l'analyse technique.
            news_query (str): Requête pour l'analyse de sentiment des actualités.
            symbol (str): Symbole de l'entreprise.
            
        Returns:
            Dict: Information sur le régime de marché détecté.
        """
        # Analyse du sentiment des actualités
        news_sentiment = None
        if news_query:
            news_sentiment = self.analyze_news_sentiment(news_query, symbol)
        
        # Analyse du sentiment des prix
        price_sentiment = None
        if price_data is not None and len(price_data) > 0:
            price_sentiment = self.analyze_price_sentiment(price_data)
        
        # Si aucune des deux analyses n'est disponible
        if news_sentiment is None and price_sentiment is None:
            self.logger.warning("Données insuffisantes pour détecter le régime de marché")
            return {
                'regime': 'unknown',
                'confidence': 0.5,
                'explanation': "Données insuffisantes"
            }
        
        # Initialiser les valeurs par défaut
        sentiment_value = 0.0
        volatility = 0.0
        
        # Combiner les résultats si disponibles
        if news_sentiment:
            sentiment_value = news_sentiment['normalized_sentiment']
        
        if price_sentiment:
            volatility = price_sentiment['volatility']
            
            # Si pas de sentiment d'actualités, utiliser le sentiment de prix
            if news_sentiment is None:
                sentiment_value = price_sentiment['price_sentiment']
            # Sinon, combiner les deux avec une pondération
            else:
                sentiment_value = 0.7 * news_sentiment['normalized_sentiment'] + 0.3 * price_sentiment['price_sentiment']
        
        # Déterminer le régime de marché
        regime = 'neutral'
        confidence = 0.6  # Confiance de base
        explanation = "Régime neutre par défaut"
        
        # Analyse du régime basée sur la volatilité et le sentiment
        if volatility > self.volatility_threshold:
            # Marché volatil
            if sentiment_value > self.sentiment_threshold:
                regime = 'volatile_bullish'
                explanation = "Marché volatil avec sentiment positif"
                confidence = min(0.6 + volatility * 10 + sentiment_value, 0.95)
            elif sentiment_value < -self.sentiment_threshold:
                regime = 'volatile_bearish'
                explanation = "Marché volatil avec sentiment négatif"
                confidence = min(0.6 + volatility * 10 - sentiment_value, 0.95)
            else:
                regime = 'volatile_neutral'
                explanation = "Marché volatil sans direction claire"
                confidence = min(0.6 + volatility * 5, 0.9)
        else:
            # Marché stable
            if sentiment_value > self.sentiment_threshold:
                regime = 'stable_bullish'
                explanation = "Marché stable avec sentiment positif"
                confidence = min(0.6 + sentiment_value * 2, 0.9)
            elif sentiment_value < -self.sentiment_threshold:
                regime = 'stable_bearish'
                explanation = "Marché stable avec sentiment négatif"
                confidence = min(0.6 - sentiment_value * 2, 0.9)
            else:
                regime = 'stable_neutral'
                explanation = "Marché stable sans direction claire"
                confidence = 0.7
        
        # Recherche de divergences entre sentiment et prix
        if news_sentiment and price_sentiment:
            news_trend = news_sentiment['trend']
            price_trend = price_sentiment['price_trend']
            
            if ('bullish' in news_trend and 'bearish' in price_trend) or ('bearish' in news_trend and 'bullish' in price_trend):
                regime = f"{regime}_divergent"
                explanation += " avec divergence entre actualités et prix"
                confidence = max(0.5, confidence - 0.1)  # Moins confiant en cas de divergence
        
        # Détection de changement de régime
        if len(self.sentiment_history) > 5 and len(self.volatility_history) > 5:
            old_sentiment = np.mean(list(self.sentiment_history)[:-5])
            new_sentiment = np.mean(list(self.sentiment_history)[-5:])
            sentiment_shift = new_sentiment - old_sentiment
            
            old_volatility = np.mean(list(self.volatility_history)[:-5])
            new_volatility = np.mean(list(self.volatility_history)[-5:])
            volatility_shift = new_volatility - old_volatility
            
            if abs(sentiment_shift) > self.sentiment_threshold or volatility_shift > self.volatility_threshold:
                regime = f"{regime}_transition"
                direction = "positive" if sentiment_shift > 0 else "négative"
                volatility_direction = "croissante" if volatility_shift > 0 else "décroissante"
                explanation += f" en transition ({direction}, volatilité {volatility_direction})"
                
                # Réduire légèrement la confiance pendant les transitions
                confidence = max(0.5, confidence - 0.05)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'explanation': explanation,
            'sentiment': sentiment_value,
            'volatility': volatility,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_state(self, filepath: str) -> bool:
        """
        Sauvegarde l'état actuel de l'analyseur.
        
        Args:
            filepath (str): Chemin du fichier de sauvegarde.
            
        Returns:
            bool: True si la sauvegarde est réussie, False sinon.
        """
        try:
            state = {
                'sentiment_history': list(self.sentiment_history),
                'rolling_sentiment': list(self.rolling_sentiment),
                'volatility_history': list(self.volatility_history),
                'last_analysis': self.last_analysis,
                'last_update_time': {k: v.isoformat() for k, v in self.last_update_time.items()},
                'config': {
                    'sentiment_window': self.sentiment_window,
                    'volatility_window': self.volatility_window,
                    'sentiment_threshold': self.sentiment_threshold,
                    'volatility_threshold': self.volatility_threshold
                },
                'saved_at': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'état: {str(e)}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Charge un état précédemment sauvegardé.
        
        Args:
            filepath (str): Chemin du fichier de sauvegarde.
            
        Returns:
            bool: True si le chargement est réussi, False sinon.
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Fichier d'état introuvable: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restaurer l'historique
            self.sentiment_history = deque(state['sentiment_history'], maxlen=100)
            self.rolling_sentiment = deque(state['rolling_sentiment'], maxlen=self.sentiment_window)
            self.volatility_history = deque(state['volatility_history'], maxlen=self.volatility_window)
            
            # Restaurer les analyses
            self.last_analysis = state['last_analysis']
            
            # Convertir les timestamps en datetime
            self.last_update_time = {
                k: datetime.fromisoformat(v) for k, v in state['last_update_time'].items()
            }
            
            # Restaurer la configuration si nécessaire
            if 'config' in state:
                self.sentiment_window = state['config'].get('sentiment_window', self.sentiment_window)
                self.volatility_window = state['config'].get('volatility_window', self.volatility_window)
                self.sentiment_threshold = state['config'].get('sentiment_threshold', self.sentiment_threshold)
                self.volatility_threshold = state['config'].get('volatility_threshold', self.volatility_threshold)
            
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de l'état: {str(e)}")
            return False
    
    def plot_sentiment_history(self, save_path: str = None) -> None:
        """
        Génère un graphique de l'historique du sentiment.
        
        Args:
            save_path (str, optional): Chemin pour sauvegarder l'image.
        """
        if len(self.sentiment_history) < 2:
            self.logger.warning("Données insuffisantes pour générer le graphique")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Tracer l'historique du sentiment
        values = list(self.sentiment_history)
        plt.plot(values, label='Sentiment', color='blue', linewidth=2)
        
        # Tracer la moyenne mobile
        window = min(len(values), self.sentiment_window)
        if window > 1:
            rolling_avg = pd.Series(values).rolling(window=window).mean().values
            plt.plot(rolling_avg, label=f'Moyenne mobile ({window} points)', color='red', linestyle='--')
        
        # Ajouter une ligne horizontale à zéro
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Ajouter des lignes de seuil
        plt.axhline(y=self.sentiment_threshold, color='green', linestyle='--', alpha=0.5, label='Seuil positif')
        plt.axhline(y=-self.sentiment_threshold, color='red', linestyle='--', alpha=0.5, label='Seuil négatif')
        
        # Embellir le graphique
        plt.title('Évolution du sentiment au fil du temps')
        plt.xlabel('Points de données')
        plt.ylabel('Score de sentiment (-1 à 1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_trading_signals(self, symbol: str, news_query: str = None, price_data: pd.DataFrame = None) -> Dict:
        """
        Génère des signaux de trading basés sur l'analyse de sentiment.
        
        Args:
            symbol (str): Symbole de l'entreprise.
            news_query (str): Requête pour l'analyse de sentiment des actualités.
            price_data (pd.DataFrame): Données de prix.
            
        Returns:
            Dict: Signaux de trading avec recommandations.
        """
        # Détecter le régime de marché
        regime_info = self.detect_market_regime(price_data, news_query, symbol)
        
        # Déterminer le signal de base
        signal = 'HOLD'  # Signal par défaut
        strength = 0.0  # Force du signal (0 à 1)
        explanation = "Signal neutre par défaut"
        stop_loss = None
        take_profit = None
        
        # Dériver le signal à partir du régime
        regime = regime_info['regime']
        sentiment = regime_info.get('sentiment', 0.0)
        volatility = regime_info.get('volatility', 0.0)
        confidence = regime_info.get('confidence', 0.5)
        
        # Déterminer le signal de trading
        if 'bullish' in regime and confidence > 0.6:
            signal = 'BUY'
            strength = min(confidence * abs(sentiment) * 2, 1.0)
            explanation = f"Signal d'achat basé sur un sentiment haussier ({sentiment:.2f}) et un régime {regime}"
            
            # Si nous avons des données de prix, calculer les niveaux
            if price_data is not None and len(price_data) > 0:
                current_price = price_data['Close'].iloc[-1]
                
                # Stop loss plus serré en cas de volatilité élevée
                if 'volatile' in regime:
                    stop_loss = current_price * (1 - 0.02 - volatility)
                    take_profit = current_price * (1 + 0.04 + volatility)
                else:
                    stop_loss = current_price * 0.97
                    take_profit = current_price * 1.05
        
        elif 'bearish' in regime and confidence > 0.6:
            signal = 'SELL'
            strength = min(confidence * abs(sentiment) * 2, 1.0)
            explanation = f"Signal de vente basé sur un sentiment baissier ({sentiment:.2f}) et un régime {regime}"
            
            # Si nous avons des données de prix, calculer les niveaux
            if price_data is not None and len(price_data) > 0:
                current_price = price_data['Close'].iloc[-1]
                
                # Stop loss plus serré en cas de volatilité élevée
                if 'volatile' in regime:
                    stop_loss = current_price * (1 + 0.02 + volatility)
                    take_profit = current_price * (1 - 0.04 - volatility)
                else:
                    stop_loss = current_price * 1.03
                    take_profit = current_price * 0.95
        
        # Si nous sommes en transition ou en divergence, atténuer le signal
        if 'transition' in regime or 'divergent' in regime:
            if signal != 'HOLD':
                explanation += f" (Signal atténué en raison de {regime})"
                strength *= 0.7  # Réduire la force du signal
                
                # Rendre le stop loss plus serré
                if stop_loss is not None and take_profit is not None:
                    if signal == 'BUY':
                        stop_loss = price_data['Close'].iloc[-1] * 0.98
                    else:
                        stop_loss = price_data['Close'].iloc[-1] * 1.02
        
        # Calculer le ratio risque/récompense
        risk_reward_ratio = None
        if stop_loss is not None and take_profit is not None:
            current_price = price_data['Close'].iloc[-1]
            
            if signal == 'BUY':
                risk = (current_price - stop_loss) / current_price
                reward = (take_profit - current_price) / current_price
            else:
                risk = (stop_loss - current_price) / current_price
                reward = (current_price - take_profit) / current_price
            
            if risk > 0:
                risk_reward_ratio = reward / risk
        
        # Position sizing recommandé
        position_size = None
        if strength > 0.3 and signal != 'HOLD':
            # Position de base: 5% à 20% selon la force du signal
            base_size = 0.05 + strength * 0.15
            
            # Ajuster en fonction de la volatilité
            volatility_factor = 1.0 - min(volatility * 10, 0.5)  # Réduire la taille en cas de volatilité élevée
            
            position_size = base_size * volatility_factor
            
            # Limiter à 20% maximum
            position_size = min(position_size, 0.2)
        
        return {
            'symbol': symbol,
            'signal': signal,
            'strength': float(strength),
            'explanation': explanation,
            'stop_loss': float(stop_loss) if stop_loss else None,
            'take_profit': float(take_profit) if take_profit else None,
            'risk_reward_ratio': float(risk_reward_ratio) if risk_reward_ratio else None,
            'recommended_position_size': float(position_size) if position_size else None,
            'regime': regime,
            'confidence': float(confidence),
            'sentiment': float(sentiment),
            'volatility': float(volatility),
            'timestamp': datetime.now().isoformat()
        } 
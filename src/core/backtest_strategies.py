import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable, Tuple, Any
try:
    import talib
except ImportError:
    try:
        # Essayer d'importer talib-binary s'il est installé
        import talib.abstract as talib
    except ImportError:
        raise ImportError("Le module 'talib' n'est pas installé. Veuillez installer 'talib-binary' avec pip: pip install talib-binary")
from .advanced_backtesting import TradingStrategy
from ..models.sentiment import MarketSentimentAnalyzer
from ..models.rl.advanced_rl_agent import RLAgentManager

class SentimentBasedStrategy(TradingStrategy):
    """
    Stratégie de trading basée sur l'analyse de sentiment du marché.
    Utilise l'analyse de sentiment pour prendre des décisions d'investissement.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None, sentiment_threshold: float = 0.2,
                 position_sizing: str = 'proportional', max_position: float = 1.0):
        """
        Initialise la stratégie basée sur le sentiment.
        
        Args:
            api_keys: Clés API pour les sources de données sentiment
            sentiment_threshold: Seuil de sentiment pour déclencher des transactions
            position_sizing: Méthode de dimensionnement des positions ('fixed', 'proportional')
            max_position: Position maximale (entre 0 et 1)
        """
        self.sentiment_analyzer = MarketSentimentAnalyzer(api_keys=api_keys)
        self.sentiment_threshold = sentiment_threshold
        self.position_sizing = position_sizing
        self.max_position = max_position
        self.last_analysis = None
        
    def get_action(self, observation: Dict) -> float:
        """
        Détermine l'action à prendre en fonction de l'analyse de sentiment.
        
        Args:
            observation: Observation actuelle du marché
            
        Returns:
            float: Valeur entre -1 et 1 représentant l'action à prendre
        """
        # Récupérer les données de prix actuelles
        main_tf = list(observation.keys())[0]
        if main_tf == 'portfolio':
            main_tf = list(observation.keys())[1]
        
        ohlcv_data = observation[main_tf]['ohlcv']
        timestamps = observation[main_tf]['timestamp']
        
        # Convertir les données en DataFrame pour l'analyse
        df = pd.DataFrame(ohlcv_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = timestamps
        df.set_index('Date', inplace=True)
        
        # Analyser le sentiment
        symbol = "BTCUSD"  # À remplacer par une valeur réelle
        news_query = "Bitcoin crypto"  # À personnaliser selon le marché
        regime_info = self.sentiment_analyzer.detect_market_regime(df, news_query, symbol)
        self.last_analysis = regime_info
        
        # Convertir le sentiment en action
        sentiment_value = regime_info.get('sentiment', 0.0)
        confidence = regime_info.get('confidence', 0.5)
        
        # Règles de trading basées sur le sentiment et le régime
        regime = regime_info['regime']
        
        # Position de base en fonction du sentiment
        if abs(sentiment_value) < self.sentiment_threshold:
            # Sentiment neutre, position neutre
            base_position = 0.0
        else:
            # Sentiment significatif, prendre position
            if self.position_sizing == 'fixed':
                # Position fixe, uniquement direction
                base_position = np.sign(sentiment_value) * self.max_position
            else:
                # Position proportionnelle au sentiment et à la confiance
                base_position = sentiment_value * confidence * self.max_position
        
        # Ajuster la position en fonction du régime de marché
        if 'volatile' in regime:
            # Réduire l'exposition dans les marchés volatils
            position = base_position * 0.7
        elif 'transition' in regime or 'divergent' in regime:
            # Réduire davantage l'exposition en période de transition
            position = base_position * 0.5
        else:
            # Position normale dans les marchés stables
            position = base_position
        
        # Limiter la position à [-max_position, max_position]
        position = np.clip(position, -self.max_position, self.max_position)
        
        return position
    
    def reset(self):
        """Réinitialise l'état interne de la stratégie."""
        self.last_analysis = None

class TechnicalStrategy(TradingStrategy):
    """
    Stratégie de trading basée sur des indicateurs techniques.
    Combine plusieurs indicateurs pour générer des signaux.
    """
    
    def __init__(self, rsi_period: int = 14, rsi_overbought: float = 70, rsi_oversold: float = 30,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 position_sizing: str = 'proportional', max_position: float = 1.0):
        """
        Initialise la stratégie basée sur des indicateurs techniques.
        
        Args:
            rsi_period: Période pour le calcul du RSI
            rsi_overbought: Niveau de surachat du RSI
            rsi_oversold: Niveau de survente du RSI
            macd_fast: Période rapide pour le MACD
            macd_slow: Période lente pour le MACD
            macd_signal: Période du signal MACD
            position_sizing: Méthode de dimensionnement des positions ('fixed', 'proportional')
            max_position: Position maximale (entre 0 et 1)
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.position_sizing = position_sizing
        self.max_position = max_position
        
        self.last_signal = None
        
    def get_action(self, observation: Dict) -> float:
        """
        Détermine l'action à prendre en fonction des indicateurs techniques.
        
        Args:
            observation: Observation actuelle du marché
            
        Returns:
            float: Valeur entre -1 et 1 représentant l'action à prendre
        """
        # Récupérer les données de prix actuelles
        main_tf = list(observation.keys())[0]
        if main_tf == 'portfolio':
            main_tf = list(observation.keys())[1]
        
        ohlcv_data = observation[main_tf]['ohlcv']
        
        # Convertir les données en arrays pour les indicateurs
        close_prices = np.array([candle[3] for candle in ohlcv_data])
        high_prices = np.array([candle[1] for candle in ohlcv_data])
        low_prices = np.array([candle[2] for candle in ohlcv_data])
        
        # Calculer les indicateurs
        # RSI
        rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)
        current_rsi = rsi[-1]
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        current_macd = macd[-1]
        current_macd_signal = macd_signal[-1]
        current_macd_hist = macd_hist[-1]
        
        # Moyenne mobile simple à 50 et 200 périodes
        if len(close_prices) >= 50:
            sma50 = talib.SMA(close_prices, timeperiod=50)
            current_sma50 = sma50[-1]
        else:
            current_sma50 = close_prices[-1]
            
        if len(close_prices) >= 200:
            sma200 = talib.SMA(close_prices, timeperiod=200)
            current_sma200 = sma200[-1]
        else:
            current_sma200 = close_prices[-1]
        
        # ATR pour la volatilité
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        current_atr = atr[-1]
        atr_percent = current_atr / close_prices[-1]
        
        # Générer des scores pour chaque indicateur (entre -1 et 1)
        
        # Score RSI: varie de -1 (survente extrême) à 1 (surachat extrême)
        # 50 est neutre (0), 30 est -0.5, 0 est -1, 70 est 0.5, 100 est 1
        rsi_score = (current_rsi - 50) / 50
        
        # Score MACD: positif quand MACD > Signal, négatif sinon
        # Intensité basée sur l'histogramme normalisé par la valeur du prix
        macd_score = current_macd_hist / (close_prices[-1] * 0.01)  # Normaliser par 1% du prix
        macd_score = np.clip(macd_score, -1, 1)  # Limiter entre -1 et 1
        
        # Score de tendance: basé sur la position relative du prix par rapport aux moyennes mobiles
        trend_score = 0.0
        # Tendance primaire (SMA200)
        if close_prices[-1] > current_sma200:
            trend_score += 0.3
        else:
            trend_score -= 0.3
        
        # Tendance secondaire (SMA50)
        if close_prices[-1] > current_sma50:
            trend_score += 0.2
        else:
            trend_score -= 0.2
        
        # Croisement SMA50/SMA200 (Golden/Death Cross)
        if current_sma50 > current_sma200:
            trend_score += 0.2
        else:
            trend_score -= 0.2
        
        # Combiner les scores avec différentes pondérations
        # RSI a moins d'importance en forte tendance
        if abs(trend_score) > 0.5:
            rsi_weight = 0.2
            macd_weight = 0.3
            trend_weight = 0.5
        else:
            rsi_weight = 0.3
            macd_weight = 0.3
            trend_weight = 0.4
        
        # Calculer le score combiné
        combined_score = (
            rsi_weight * -rsi_score +  # Inverser le RSI (surachat = signal de vente)
            macd_weight * macd_score + 
            trend_weight * trend_score
        )
        
        # Normaliser le score final entre -1 et 1
        combined_score = np.clip(combined_score, -1, 1)
        
        # Dimensionnement de la position
        if self.position_sizing == 'fixed':
            # Position fixe, uniquement direction
            if abs(combined_score) < 0.2:
                position = 0  # Zone neutre, pas de position
            else:
                position = np.sign(combined_score) * self.max_position
        else:
            # Position proportionnelle au score
            position = combined_score * self.max_position
        
        # Stocker le dernier signal
        self.last_signal = {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_macd_signal,
            'macd_hist': current_macd_hist,
            'trend': 'bullish' if trend_score > 0 else 'bearish',
            'combined_score': combined_score,
            'position': position
        }
        
        return position
    
    def reset(self):
        """Réinitialise l'état interne de la stratégie."""
        self.last_signal = None

class RLBasedStrategy(TradingStrategy):
    """
    Stratégie de trading basée sur l'apprentissage par renforcement.
    Utilise un agent RL pré-entraîné pour prendre des décisions.
    """
    
    def __init__(self, agent_id: str, model_dir: str = 'saved_models/rl_agents',
                 model_type: str = 'PPO', use_market_regime: bool = True,
                 api_keys: Dict[str, str] = None):
        """
        Initialise la stratégie basée sur l'apprentissage par renforcement.
        
        Args:
            agent_id: Identifiant de l'agent à utiliser
            model_dir: Répertoire des modèles sauvegardés
            model_type: Type du modèle ('PPO', 'SAC', 'TD3', 'Custom')
            use_market_regime: Si True, utilise des agents spécifiques aux régimes de marché
            api_keys: Clés API pour les sources de données sentiment (si use_market_regime=True)
        """
        self.agent_manager = RLAgentManager(model_dir=model_dir)
        self.agent_id = agent_id
        self.model_type = model_type
        self.use_market_regime = use_market_regime
        
        # Pour l'adaptation au régime de marché
        if use_market_regime:
            self.sentiment_analyzer = MarketSentimentAnalyzer(api_keys=api_keys)
            self.current_regime = "stable_neutral"  # Régime par défaut
        
        # Charger l'agent principal
        try:
            self.agent = self.agent_manager.load_agent(agent_id, model_type)
        except FileNotFoundError:
            raise ValueError(f"Agent {agent_id} non trouvé dans {model_dir}")
    
    def _preprocess_observation(self, observation: Dict) -> np.ndarray:
        """
        Prétraite l'observation pour l'agent RL.
        
        Args:
            observation: Observation brute du marché
            
        Returns:
            np.ndarray: Observation prétraitée pour l'agent
        """
        # Extraire les données pertinentes
        features = []
        
        # Données de prix (utiliser le timeframe principal)
        main_tf = list(observation.keys())[0]
        if main_tf == 'portfolio':
            main_tf = list(observation.keys())[1]
        
        ohlcv_data = np.array(observation[main_tf]['ohlcv'])
        
        # Calculer les rendements
        close_prices = ohlcv_data[:, 3]
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = np.append(0, returns)  # Ajouter un 0 au début pour maintenir la taille
        
        # Normaliser OHLCV (diviser par le dernier prix de clôture)
        norm_factor = close_prices[-1]
        normalized_ohlcv = ohlcv_data / norm_factor
        
        # Ajouter les rendements
        normalized_ohlcv = np.column_stack((normalized_ohlcv, returns))
        
        # Aplatir les données
        features.extend(normalized_ohlcv.flatten())
        
        # Ajouter des informations de portefeuille
        portfolio = observation['portfolio']
        current_position = portfolio['position']
        current_capital = portfolio['capital']
        unrealized_pnl = portfolio['unrealized_pnl']
        
        # Normaliser et ajouter les informations de portefeuille
        features.append(current_position)
        features.append(current_capital / 100000)  # Normaliser par rapport au capital initial typique
        features.append(unrealized_pnl / current_capital if current_capital > 0 else 0)
        
        # Convertir en array numpy
        return np.array(features, dtype=np.float32)
    
    def get_action(self, observation: Dict) -> float:
        """
        Détermine l'action à prendre en utilisant l'agent RL.
        
        Args:
            observation: Observation actuelle du marché
            
        Returns:
            float: Valeur entre -1 et 1 représentant l'action à prendre
        """
        # Détecter le régime de marché si nécessaire
        if self.use_market_regime:
            # Récupérer les données de prix
            main_tf = list(observation.keys())[0]
            if main_tf == 'portfolio':
                main_tf = list(observation.keys())[1]
            
            ohlcv_data = observation[main_tf]['ohlcv']
            timestamps = observation[main_tf]['timestamp']
            
            # Convertir en DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = timestamps
            df.set_index('Date', inplace=True)
            
            # Détecter le régime
            regime_info = self.sentiment_analyzer.detect_market_regime(df)
            self.current_regime = regime_info['regime']
            
            # Choisir l'agent spécifique au régime si disponible
            regime_agent_id = f"regime_{self.current_regime}"
            try:
                regime_agent = self.agent_manager.load_agent(regime_agent_id, self.model_type)
                # Utiliser l'agent spécifique au régime
                agent_to_use = regime_agent
            except (FileNotFoundError, ValueError):
                # Utiliser l'agent par défaut si l'agent spécifique n'est pas disponible
                agent_to_use = self.agent
        else:
            # Utiliser l'agent par défaut
            agent_to_use = self.agent
        
        # Prétraiter l'observation
        processed_obs = self._preprocess_observation(observation)
        
        # Obtenir l'action de l'agent
        action = self.agent_manager.predict(self.agent_id, processed_obs)
        
        # L'action est un array, prendre la première valeur
        if isinstance(action, np.ndarray):
            action = action[0]
        
        # S'assurer que l'action est dans la plage [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        return float(action)
    
    def reset(self):
        """Réinitialise l'état interne de la stratégie."""
        if self.use_market_regime:
            self.current_regime = "stable_neutral"

class HybridStrategy(TradingStrategy):
    """
    Stratégie hybride combinant analyse technique, sentiment et RL.
    """
    
    def __init__(self, tech_weight: float = 0.4, sentiment_weight: float = 0.3, 
                 rl_weight: float = 0.3, api_keys: Dict[str, str] = None,
                 agent_id: str = 'default_agent', model_dir: str = 'saved_models/rl_agents'):
        """
        Initialise la stratégie hybride.
        
        Args:
            tech_weight: Poids de la composante technique
            sentiment_weight: Poids de la composante sentiment
            rl_weight: Poids de la composante RL
            api_keys: Clés API pour les sources de données sentiment
            agent_id: Identifiant de l'agent RL à utiliser
            model_dir: Répertoire des modèles RL sauvegardés
        """
        self.tech_weight = tech_weight
        self.sentiment_weight = sentiment_weight
        self.rl_weight = rl_weight
        
        # Normaliser les poids
        total_weight = tech_weight + sentiment_weight + rl_weight
        self.tech_weight /= total_weight
        self.sentiment_weight /= total_weight
        self.rl_weight /= total_weight
        
        # Initialiser les sous-stratégies
        self.tech_strategy = TechnicalStrategy()
        self.sentiment_strategy = SentimentBasedStrategy(api_keys=api_keys)
        self.rl_strategy = RLBasedStrategy(agent_id=agent_id, model_dir=model_dir, api_keys=api_keys)
        
        self.last_actions = {}
    
    def get_action(self, observation: Dict) -> float:
        """
        Détermine l'action à prendre en combinant les différentes stratégies.
        
        Args:
            observation: Observation actuelle du marché
            
        Returns:
            float: Valeur entre -1 et 1 représentant l'action à prendre
        """
        # Obtenir les actions des sous-stratégies
        tech_action = self.tech_strategy.get_action(observation)
        sentiment_action = self.sentiment_strategy.get_action(observation)
        rl_action = self.rl_strategy.get_action(observation)
        
        # Stocker les dernières actions
        self.last_actions = {
            'technical': tech_action,
            'sentiment': sentiment_action,
            'rl': rl_action
        }
        
        # Combiner les actions avec leurs poids respectifs
        combined_action = (
            self.tech_weight * tech_action +
            self.sentiment_weight * sentiment_action +
            self.rl_weight * rl_action
        )
        
        # S'assurer que l'action est dans la plage [-1, 1]
        combined_action = np.clip(combined_action, -1.0, 1.0)
        
        return float(combined_action)
    
    def reset(self):
        """Réinitialise l'état interne de la stratégie et des sous-stratégies."""
        self.tech_strategy.reset()
        self.sentiment_strategy.reset()
        self.rl_strategy.reset()
        self.last_actions = {} 
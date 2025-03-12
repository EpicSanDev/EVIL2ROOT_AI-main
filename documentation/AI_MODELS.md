# Modèles d'Intelligence Artificielle

Ce document détaille les différents modèles d'IA et d'apprentissage automatique utilisés dans le système EVIL2ROOT Trading Bot.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [PricePredictionModel](#pricepredictionmodel)
3. [IndicatorManagementModel](#indicatormanagementmodel)
4. [RiskManagementModel](#riskmanagementmodel)
5. [TpSlManagementModel](#tpslmanagementmodel)
6. [RLTradingModel](#rltradingmodel)
7. [SentimentAnalyzer](#sentimentanalyzer)
8. [TransformerModel](#transformermodel)
9. [NewsRetrieval](#newsretrieval)
10. [Claude 3.7 Integration](#claude-37-integration)
11. [Entraînement et évaluation](#entraînement-et-évaluation)
12. [Infrastructures IA](#infrastructures-ia)

## Vue d'ensemble

Le système EVIL2ROOT utilise une approche multi-modèles où différentes techniques d'IA collaborent pour générer des signaux de trading fiables. Cette architecture permet de combiner les avantages de différentes approches et de compenser les faiblesses individuelles de chaque modèle.

L'approche hybride combine :
- Deep learning pour la prédiction de séries temporelles
- Machine learning classique pour la classification et la régression
- Apprentissage par renforcement pour l'optimisation des décisions
- Traitement du langage naturel pour l'analyse de sentiment et d'actualités
- LLM avancé (Claude 3.7) pour la validation et l'analyse approfondie

## PricePredictionModel

**Fichier** : `app/models/price_prediction.py`

**Objectif** : Prédire les mouvements de prix futurs pour différents horizons temporels.

**Architecture** :
- Réseaux LSTM et GRU bidirectionnels
- Architecture avec couches de dropout et de normalisation par lots
- Optimisation dynamique des hyperparamètres via recherche bayésienne

**Implémentation** :
```python
def build_model(self, params, input_shape, output_shape=1):
    model_type = params.get('model_type', 'lstm')
    layers = params.get('layers', 2)
    units = params.get('units', 64)
    dropout = params.get('dropout', 0.2)
    learning_rate = params.get('learning_rate', 0.001)
    
    model = Sequential()
    
    if model_type == 'lstm':
        model.add(Bidirectional(LSTM(units, 
                                    return_sequences=(layers > 1),
                                    input_shape=input_shape,
                                    recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        for i in range(layers - 1):
            return_sequences = i < layers - 2
            model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
    
    elif model_type == 'gru':
        model.add(Bidirectional(GRU(units, 
                                    return_sequences=(layers > 1),
                                    input_shape=input_shape,
                                    recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        for i in range(layers - 1):
            return_sequences = i < layers - 2
            model.add(Bidirectional(GRU(units, return_sequences=return_sequences)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(output_shape))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model
```

**Features utilisées** :
- Prix OHLCV (Open, High, Low, Close, Volume)
- Indicateurs techniques (RSI, MACD, Bollinger Bands, etc.)
- Caractéristiques dérivées (rendements logarithmiques, volatilité, etc.)
- Moyennes mobiles sur différentes périodes

**Processus d'entraînement** :
- Validation croisée temporelle pour éviter le look-ahead bias
- Early stopping pour éviter le surapprentissage
- Réduction adaptative du taux d'apprentissage
- Sauvegarde du meilleur modèle

**Performance** :
- Métriques : MSE, MAE, R²
- Validation sur des données hors échantillon
- Backtesting sur données historiques
- Évaluation comparative contre des modèles de référence

## IndicatorManagementModel

**Fichier** : `app/models/indicator_management.py`

**Objectif** : Analyser les indicateurs techniques et détecter les patterns chartistes.

**Techniques utilisées** :
- Modèles d'ensemble (Random Forest, XGBoost, etc.)
- Détection de patterns basée sur des règles
- Analyse par fenêtre glissante

**Implémentation principale** :
```python
class IndicatorManagementModel:
    def __init__(self):
        self.indicators = {
            'trend': ['sma', 'ema', 'macd', 'adx'],
            'momentum': ['rsi', 'stoch', 'cci', 'mfi'],
            'volatility': ['bbands', 'atr', 'keltner'],
            'volume': ['obv', 'vwap', 'ad', 'cmf']
        }
        self.models = {}
        self.pattern_detectors = self._init_pattern_detectors()
        
    def _init_pattern_detectors(self):
        detectors = {
            'head_shoulders': HeadAndShouldersDetector(),
            'double_top': DoubleTopDetector(),
            'double_bottom': DoubleBottomDetector(),
            'triangle': TriangleDetector(),
            'rectangle': RectangleDetector(),
            'channel': ChannelDetector(),
            'fibonacci': FibonacciRetracementDetector()
        }
        return detectors
    
    def analyze_indicators(self, data, symbol):
        results = {}
        
        # Trend analysis
        trend_score = self._analyze_trend_indicators(data)
        results['trend'] = trend_score
        
        # Momentum analysis
        momentum_score = self._analyze_momentum_indicators(data)
        results['momentum'] = momentum_score
        
        # Volatility analysis
        volatility_score = self._analyze_volatility_indicators(data)
        results['volatility'] = volatility_score
        
        # Volume analysis
        volume_score = self._analyze_volume_indicators(data)
        results['volume'] = volume_score
        
        # Pattern detection
        patterns = self._detect_patterns(data)
        results['patterns'] = patterns
        
        # Calculate overall signal score
        signal_score = self._calculate_signal_score(results)
        results['signal_score'] = signal_score
        
        return results
```

**Caractéristiques principales** :
- Classification des indicateurs par catégories (tendance, momentum, volatilité, volume)
- Détection de plus de 15 patterns chartistes différents
- Système de scoring pour évaluer la force du signal
- Analyse des divergences entre prix et indicateurs

## RiskManagementModel

**Fichier** : `app/models/risk_management.py`

**Objectif** : Évaluer les risques de marché et déterminer la taille optimale des positions.

**Approches** :
- Modélisation de la volatilité (GARCH, EWMA)
- Calcul de Value-at-Risk (VaR)
- Optimisation de portefeuille

**Algorithmes clés** :
- Modèle GARCH pour la prévision de volatilité
- Calcul de Kelly Criterion modifié pour le sizing des positions
- Modèle Monte Carlo pour l'évaluation des risques

**Implémentation de la gestion de position** :
```python
def calculate_position_size(self, symbol, action, account_balance, risk_per_trade, current_price, stop_loss_price):
    """
    Calcule la taille de position optimale en utilisant un critère de Kelly modifié
    et en tenant compte de la volatilité actuelle du marché.
    
    Args:
        symbol: Le symbole de trading
        action: 'buy' ou 'sell'
        account_balance: Solde du compte
        risk_per_trade: Pourcentage du compte à risquer par trade (ex: 0.02 pour 2%)
        current_price: Prix d'entrée actuel
        stop_loss_price: Prix du stop loss
        
    Returns:
        Taille de position recommandée en unités
    """
    # Calcul du risque monétaire maximum
    max_risk_amount = account_balance * risk_per_trade
    
    # Calcul du risque par unité
    price_risk = abs(current_price - stop_loss_price)
    if price_risk == 0:
        return 0  # Éviter la division par zéro
    
    # Taille de position de base (sans ajustement de volatilité)
    base_position_size = max_risk_amount / price_risk
    
    # Ajustement en fonction de la volatilité actuelle
    volatility_multiplier = self._get_volatility_adjustment(symbol)
    
    # Appliquer les contraintes de diversification
    diversification_factor = self._get_diversification_factor(symbol)
    
    # Calcul de la taille finale
    final_position_size = base_position_size * volatility_multiplier * diversification_factor
    
    # Arrondir à un nombre entier d'unités ou à la précision appropriée
    return self._round_position_size(final_position_size, symbol)
```

## TpSlManagementModel

**Fichier** : `app/models/tp_sl_management.py`

**Objectif** : Déterminer les niveaux optimaux de take-profit et stop-loss et gérer leur ajustement dynamique.

**Caractéristiques** :
- Calcul de niveaux basés sur la volatilité
- Identification des niveaux de support/résistance pour le placement
- Trailing stops intelligents qui s'adaptent aux conditions de marché

**Algorithmes notables** :
- Identification des niveaux de Fibonacci
- Modèle ML pour la prédiction de la volatilité future
- Système de trailing stop adaptatif

## RLTradingModel

**Fichier** : `app/models/rl_trading.py`

**Objectif** : Utiliser l'apprentissage par renforcement pour optimiser les décisions de trading.

**Architecture** :
- Environnement de trading personnalisé compatible gym
- Agent PPO (Proximal Policy Optimization)
- Réseau de neurones pour la politique et la fonction de valeur

**Implémentation de l'environnement** :
```python
class TradingEnvironment(gym.Env):
    """
    Environnement de trading pour l'apprentissage par renforcement.
    Conforme à l'interface gym standard.
    """
    
    def __init__(self, data, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Définir l'espace d'action : 0 (ne rien faire), 1 (acheter), 2 (vendre)
        self.action_space = spaces.Discrete(3)
        
        # Définir l'espace d'observation (état du marché + état du portefeuille)
        # État du marché: prix normalisés + indicateurs techniques + autres features
        n_features = data.shape[1]  # Nombre de colonnes dans les données
        
        # Ajouter 3 dimensions pour l'état du portefeuille (solde, position, prix d'entrée)
        self.observation_space = spaces.Box(
            low=np.negative(np.ones(n_features + 3)),
            high=np.ones(n_features + 3),
            dtype=np.float32
        )
        
        # Variables d'état
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0 = pas de position, positive = long, negative = short
        self.entry_price = 0
        self.total_reward = 0
        self.history = []
        
        return self._get_observation()
    
    def step(self, action):
        # Exécuter l'action
        self._take_action(action)
        
        # Avancer d'un pas
        self.current_step += 1
        
        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.data) - 1
        
        # Calculer la récompense
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Obtenir la nouvelle observation
        obs = self._get_observation()
        
        # Informations supplémentaires
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_reward': self.total_reward
        }
        
        return obs, reward, done, info
```

**Caractéristiques de l'apprentissage** :
- Récompenses basées sur le profit ajusté au risque (Sharpe ratio)
- Entraînement sur des données historiques diversifiées
- Fine-tuning sur des marchés spécifiques
- Techniques de généralisation pour éviter le surapprentissage

## SentimentAnalyzer

**Fichier** : `app/models/sentiment_analysis.py`

**Objectif** : Analyser le sentiment des actualités financières et des médias sociaux.

**Techniques** :
- Modèles de NLP pré-entraînés (DistilBERT, RoBERTa)
- Analyse de sentiment spécifique au domaine financier
- Agrégation et scoring multi-sources

**Implémentation clé** :
```python
class SentimentAnalyzer:
    def __init__(self):
        # Chargement des modèles
        try:
            self.general_sentiment_model = pipeline("sentiment-analysis", 
                                                 model="distilbert-base-uncased-finetuned-sst-2-english")
            
            # Modèle spécifique pour le sentiment financier
            self.finance_sentiment_model = pipeline("sentiment-analysis",
                                                 model="ProsusAI/finbert")
            
            # Modèle d'embedding pour l'analyse sémantique
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Classification d'impact
            self.impact_classifier = pipeline("zero-shot-classification",
                                           model="facebook/bart-large-mnli")
            
            self.initialized = True
        except Exception as e:
            logging.error(f"Error initializing sentiment models: {e}")
            self.initialized = False
    
    def analyze_news(self, news_items):
        """
        Analyse le sentiment de plusieurs articles d'actualité.
        
        Args:
            news_items: Liste de dictionnaires contenant 'title', 'description', et 'url'
            
        Returns:
            Dictionnaire avec scores de sentiment agrégés et par article
        """
        if not self.initialized:
            return {"error": "Sentiment models not initialized", "score": 0}
        
        results = []
        
        for item in news_items:
            # Construire le texte à analyser (titre + description)
            text = f"{item['title']}. {item['description']}"
            
            # Analyse du sentiment général
            general_sentiment = self.general_sentiment_model(text)
            
            # Analyse du sentiment financier
            finance_sentiment = self.finance_sentiment_model(text)
            
            # Classification de l'impact
            impact_classes = ["market moving", "significant impact", "minor impact", "no impact"]
            impact = self.impact_classifier(text, impact_classes)
            
            # Calculer un score composite
            sentiment_score = self._calculate_composite_score(general_sentiment, finance_sentiment, impact)
            
            # Stocker les résultats
            result = {
                "title": item["title"],
                "url": item["url"],
                "general_sentiment": general_sentiment,
                "finance_sentiment": finance_sentiment,
                "impact": impact,
                "composite_score": sentiment_score
            }
            
            results.append(result)
        
        # Calculer le score agrégé
        if results:
            aggregate_score = self._calculate_aggregate_score(results)
        else:
            aggregate_score = 0
        
        return {
            "individual_results": results,
            "aggregate_score": aggregate_score
        }
```

**Sources de données** :
- Actualités financières majeures
- Articles spécialisés
- Twitter/Reddit sentiment
- Rapports d'analystes financiers

## TransformerModel

**Fichier** : `app/models/transformer_model.py`

**Objectif** : Utiliser l'architecture Transformer pour l'analyse de séries temporelles financières.

**Architecture** :
- Transformers adaptés pour les séries temporelles
- Mécanisme d'attention pour capter les dépendances temporelles
- Architecture multi-tête pour l'analyse multi-échelle

**Capacités** :
- Capture des dépendances à long terme dans les données
- Gestion efficace des valeurs manquantes
- Fusion de données multimodales (prix, volume, indicateurs)
- Analyse de corrélations complexes

## NewsRetrieval

**Fichier** : `app/models/news_retrieval.py`

**Objectif** : Collecter et filtrer les actualités financières pertinentes.

**Fonctionnalités** :
- Collecte automatisée via différentes API
- Filtrage par pertinence pour le trading
- Classification par impact potentiel sur le marché
- Extraction d'informations clés

**Sources** :
- API News génériques
- Services d'actualités financières spécialisés
- Flux RSS de sites financiers
- API Twitter pour les actualités en temps réel

## Claude 3.7 Integration

**Fichier** : `app/ai_trade_validator.py`

**Objectif** : Utiliser Claude 3.7 pour une analyse approfondie et une validation des décisions de trading.

**Mise en œuvre** :
- Communication via l'API OpenRouter
- Prompts structurés pour l'analyse de trading
- Extraction et interprétation des réponses
- Combinaison avec d'autres modèles pour une décision finale

**Exemple de prompt** :
```python
def _generate_claude_prompt(self, symbol, proposed_action, proposed_price, proposed_tp, proposed_sl, market_data):
    """
    Génère un prompt structuré pour Claude 3.7 afin d'analyser une décision de trading.
    """
    # Format des données de marché pour le prompt
    market_data_recent = market_data.tail(20).reset_index()
    market_data_str = market_data_recent.to_string(index=False)
    
    # Calcul des indicateurs clés pour enrichir le prompt
    current_price = market_data.iloc[-1]['Close']
    price_change_1d = ((current_price / market_data.iloc[-2]['Close']) - 1) * 100
    price_change_5d = ((current_price / market_data.iloc[-6]['Close']) - 1) * 100
    price_change_20d = ((current_price / market_data.iloc[-21]['Close']) - 1) * 100
    
    # RSI actuel
    rsi = market_data.iloc[-1].get('RSI', 'N/A')
    
    # Informations générales sur le symbole
    instrument_type = "cryptocurrency" if "-USD" in symbol else "stock"
    
    # Construction du prompt structuré
    prompt = f"""
    Je suis un expert en trading professionnel avec 20 ans d'expérience sur les marchés financiers. 
    Je dois évaluer cette décision de trading et déterminer si elle est judicieuse.
    
    ## Contexte du marché
    - Symbole: {symbol} ({instrument_type})
    - Date actuelle: {datetime.now().strftime('%Y-%m-%d')}
    - Prix actuel: ${current_price:.2f}
    - Variation 1 jour: {price_change_1d:.2f}%
    - Variation 5 jours: {price_change_5d:.2f}%
    - Variation 20 jours: {price_change_20d:.2f}%
    - RSI actuel: {rsi}
    
    ## Décision de trading proposée
    - Action: {proposed_action.upper()}
    - Prix d'entrée proposé: ${proposed_price:.2f}
    - Take-profit proposé: ${proposed_tp:.2f} ({((proposed_tp/proposed_price)-1)*100:.2f}%)
    - Stop-loss proposé: ${proposed_sl:.2f} ({((proposed_sl/proposed_price)-1)*100:.2f}%)
    - Ratio risque/récompense: {abs((proposed_tp-proposed_price)/(proposed_sl-proposed_price)):.2f}
    
    ## Données récentes du marché
    ```
    {market_data_str}
    ```
    
    ## Tâche
    Analyse cette décision de trading et détermine si elle est valide en te basant sur:
    1. L'analyse technique (tendance, momentum, niveaux de support/résistance)
    2. Le timing de l'entrée
    3. Le placement du take-profit et du stop-loss
    4. Le ratio risque/récompense
    5. La cohérence globale de la stratégie
    
    Fournis une réponse structurée avec:
    1. Un bref résumé de la situation du marché
    2. Une analyse détaillée de la décision proposée
    3. Des points forts et des points faibles de cette décision
    4. Une conclusion claire avec OUI ou NON pour valider la transaction
    5. Un score de confiance entre 0 et 1
    
    Ta réponse doit se terminer par une ligne formatée exactement comme:
    DÉCISION: [OUI/NON], CONFIANCE: [0.XX]
    """
    
    return prompt
```

## Entraînement et évaluation

**Fichier** : `app/model_trainer.py`

**Objectif** : Entraîner, optimiser et évaluer les modèles d'IA.

**Processus d'entraînement** :
- Division des données respectant la temporalité
- Optimisation des hyperparamètres
- Validation croisée temporelle
- Early stopping et régularisation

**Métriques d'évaluation** :
- Précision, rappel et F1-score pour les modèles de classification
- MSE, MAE pour les modèles de régression
- Profit/perte et Sharpe ratio pour les modèles de trading
- Drawdown et autres métriques de risque

**Sauvegarde et versionnage** :
- Stockage des modèles entraînés avec leurs métadonnées
- Suivi des performances au fil du temps
- Rollback possible en cas de dégradation des performances

## Infrastructures IA

**Besoins matériels** :
- GPU pour l'entraînement des modèles profonds
- CPU multi-cœurs pour les prédictions en temps réel
- RAM suffisante pour manipuler de grands ensembles de données
- Stockage rapide pour les données historiques et les modèles

**Optimisations** :
- Quantification des modèles pour des inférences plus rapides
- Optimisation des opérations de convolution et des couches récurrentes
- Parallélisation des calculs d'indicateurs techniques
- Mise en cache des résultats intermédiaires 
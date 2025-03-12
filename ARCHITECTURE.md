# Architecture du Système EVIL2ROOT Trading Bot

Ce document fournit une vue d'ensemble détaillée de l'architecture technique du système de trading EVIL2ROOT, pour aider les développeurs à comprendre sa structure et ses interactions.

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture des microservices](#architecture-des-microservices)
- [Flux de données](#flux-de-données)
- [Composants principaux](#composants-principaux)
  - [Trading Bot](#trading-bot)
  - [AI Validator](#ai-validator)
  - [Web UI](#web-ui)
  - [Daily Analysis Bot](#daily-analysis-bot)
  - [Market Analysis Scheduler](#market-analysis-scheduler)
- [Base de données](#base-de-données)
- [Modèles d'IA](#modèles-dia)
- [Sécurité](#sécurité)
- [Scalabilité](#scalabilité)
- [Dépendances externes](#dépendances-externes)

## Vue d'ensemble

EVIL2ROOT Trading Bot est conçu comme une architecture de microservices communiquant via Redis et utilisant PostgreSQL comme stockage persistant. Cette architecture permet une haute disponibilité, une séparation claire des responsabilités et une scalabilité facilitée.

## Architecture des microservices

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Web (Browser)                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Reverse Proxy (Nginx)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌───────────────────────┐   ┌───────────────────────┐
│                       │   │                       │
│       Web UI          │   │    REST API           │
│       (Flask)         │   │    (Flask)            │
│                       │   │                       │
└───────────┬───────────┘   └───────────┬───────────┘
            │                           │
            └───────────┬───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                       Message Bus                           │
│                        (Redis)                              │
└────┬─────────────────────┬──────────────────────┬───────────┘
     │                     │                      │
     ▼                     ▼                      ▼
┌──────────────┐    ┌──────────────┐     ┌────────────────┐
│              │    │              │     │                │
│ Trading Bot  │    │ AI Validator │     │ Daily Analysis │
│              │    │              │     │     Bot        │
└──────┬───────┘    └──────┬───────┘     └────────┬───────┘
       │                   │                      │
       │                   │                      │
       │                   │                      │
       ▼                   ▼                      ▼
┌────────────┐    ┌─────────────────┐     ┌────────────────┐
│            │    │                 │     │                │
│ Market     │    │ Model Trainer   │     │ Telegram Bot   │
│ Scheduler  │    │                 │     │                │
└──────┬─────┘    └──────┬──────────┘     └────────┬───────┘
       │                 │                         │
       └─────────┬───────┴─────────────┬──────────┘
                 │                     │
                 ▼                     ▼
┌─────────────────────────┐     ┌────────────────────────┐
│                         │     │                        │
│  PostgreSQL Database    │     │  External APIs         │
│                         │     │  (Market Data, etc.)   │
└─────────────────────────┘     └────────────────────────┘
```

## Flux de données

1. **Acquisition des données**:
   - Le Data Manager (intégré dans différents composants) récupère les données de marché via des API externes (yfinance, etc.)
   - Les données sont nettoyées, normalisées et enrichies avec des indicateurs techniques
   - Les données traitées sont stockées dans PostgreSQL et partagées via Redis

2. **Génération des signaux**:
   - Le Trading Bot analyse les données avec différents modèles (ML, RL, analyse technique)
   - Les signaux générés sont combinés pour produire des décisions de trading potentielles
   - Chaque signal est enregistré dans la base de données pour analyse ultérieure

3. **Validation IA**:
   - Les décisions de trading sont envoyées au service AI Validator via Redis
   - L'AI Validator effectue une analyse approfondie (Claude 3.7, modèles ML internes)
   - Les résultats de validation sont renvoyés au Trading Bot avec un score de confiance

4. **Exécution des transactions**:
   - Les transactions validées sont exécutées (ou simulées en mode backtest)
   - Les transactions sont journalisées dans la base de données
   - Les notifications sont envoyées via Telegram

5. **Analyses planifiées**:
   - Le Market Analysis Scheduler déclenche des analyses à intervalles réguliers
   - Le Daily Analysis Bot effectue des analyses quotidiennes approfondies
   - Les résultats sont enregistrés et communiqués via Telegram/Interface Web

6. **Présentation Web**:
   - L'interface Web interroge la base de données pour les données historiques
   - Les données en temps réel sont fournies via des connexions Redis
   - Les graphiques et tableaux de bord sont générés dynamiquement

## Composants principaux

### Trading Bot

**Objectif**: Analyser les marchés, générer des signaux de trading et exécuter des transactions.

**Fichiers clés**:
- `app/trading.py`: Logique principale du bot de trading
- `app/models/`: Modèles d'IA et d'apprentissage automatique
- `app/model_trainer.py`: Entraînement des modèles

**Fonctionnalités**:
- Génération de signaux via plusieurs approches (techniques, ML, RL)
- Gestion des positions ouvertes (stop-loss, take-profit)
- Calcul des tailles de position et gestion du risque
- Communication avec le service de validation IA

### AI Validator

**Objectif**: Fournir une validation indépendante des décisions de trading.

**Fichiers clés**:
- `app/ai_trade_validator.py`: Service de validation des transactions
- `app/models/sentiment_analysis.py`: Analyse du sentiment du marché
- `app/models/news_retrieval.py`: Collecte et analyse des actualités financières

**Fonctionnalités**:
- Validation multi-facteur des décisions de trading
- Utilisation de Claude 3.7 pour analyse avancée
- Vérification de compatibilité avec les tendances de marché
- Calcul de scores de confiance
- Intégration de l'analyse de sentiment et d'actualités

### Web UI

**Objectif**: Fournir une interface utilisateur pour surveiller et configurer le système.

**Fichiers clés**:
- `app/routes.py`: Routes Flask pour l'interface web
- `app/templates/`: Templates HTML pour les pages web
- `app/static/`: Ressources statiques (CSS, JS, images)

**Fonctionnalités**:
- Tableau de bord de performance en temps réel
- Graphiques et visualisations des données de marché
- Configuration des paramètres du bot
- Historique des transactions et signaux
- Tableaux de bord avancés et analytics

### Daily Analysis Bot

**Objectif**: Réaliser des analyses quotidiennes approfondies des marchés.

**Fichiers clés**:
- `app/daily_analysis_bot.py`: Logique principale du bot d'analyse quotidienne
- `start_daily_analysis.py`: Point d'entrée pour l'analyse quotidienne

**Fonctionnalités**:
- Analyse technique et fondamentale approfondie
- Génération de rapports détaillés
- Identification des opportunités de trading à moyen terme
- Envoi automatisé de rapports par Telegram

### Market Analysis Scheduler

**Objectif**: Planifier et coordonner les analyses de marché à intervalles réguliers.

**Fichiers clés**:
- `app/market_analysis_scheduler.py`: Gestionnaire des analyses planifiées
- `start_market_scheduler.sh`: Script de démarrage du planificateur

**Fonctionnalités**:
- Planification des analyses à intervalles configurables
- Coordination des tâches d'analyse
- Optimisation des ressources système
- Priorisation des analyses selon les conditions de marché

## Base de données

Le système utilise PostgreSQL avec le schéma suivant:

**Tables principales**:
- `trade_history`: Enregistre toutes les transactions exécutées
- `trading_signals`: Stocke les signaux générés par les modèles
- `market_data`: Contient des snapshots de données de marché historiques
- `performance_metrics`: Métriques de performance quotidiennes
- `bot_settings`: Configuration du bot de trading
- `open_positions`: Positions actuellement ouvertes
- `analysis_results`: Résultats des analyses quotidiennes et planifiées
- `model_performance`: Métriques de performance des modèles d'IA

## Modèles d'IA

Le système utilise plusieurs modèles d'IA complémentaires:

1. **PricePredictionModel** (`app/models/price_prediction.py`):
   - Réseaux LSTM et GRU bidirectionnels pour prédire les mouvements de prix
   - Architecture avancée avec couches de normalisation et dropout
   - Optimisation bayésienne des hyperparamètres
   - Entrée: Données OHLCV et indicateurs techniques
   - Sortie: Prédiction de prix pour différents horizons temporels

2. **IndicatorManagementModel** (`app/models/indicator_management.py`):
   - Analyse des indicateurs techniques classiques (RSI, MACD, etc.)
   - Détection de patterns graphiques
   - Reconnaissance de configurations techniques spécifiques

3. **RiskManagementModel** (`app/models/risk_management.py`):
   - Évaluation du risque de marché actuel
   - Recommandation de taille de position
   - Analyse de la volatilité et gestion dynamique des risques

4. **TpSlManagementModel** (`app/models/tp_sl_management.py`):
   - Calcul des niveaux optimaux de take-profit et stop-loss
   - Ajustement dynamique des trailing stops
   - Stratégies de sortie en plusieurs étapes

5. **RLTradingModel** (`app/models/rl_trading.py`):
   - Agent d'apprentissage par renforcement pour les décisions de trading
   - Utilise Stable Baselines pour l'implémentation
   - Environnement d'apprentissage personnalisé pour le trading

6. **SentimentAnalyzer** (`app/models/sentiment_analysis.py`):
   - Analyse du sentiment à partir des news financières et médias sociaux
   - Utilise des modèles de NLP pré-entraînés
   - Intégration avec des transformers pour l'analyse de texte

7. **TransformerModel** (`app/models/transformer_model.py`):
   - Architecture Transformer pour analyse de séries temporelles
   - Mécanisme d'attention pour identifier les corrélations importantes
   - Capacité à intégrer des données multimodales

8. **NewsRetrieval** (`app/models/news_retrieval.py`):
   - Collecte automatisée d'actualités financières
   - Filtrage et classification des nouvelles pertinentes
   - Extraction d'insights pour la prise de décision de trading

## Sécurité

- Toutes les communications sont chiffrées
- Les clés API sont stockées dans des variables d'environnement
- L'accès à la base de données est limité par conteneur
- Les données sensibles ne sont jamais exposées via l'API
- Validation des entrées pour prévenir les injections SQL et autres attaques

## Scalabilité

L'architecture permet plusieurs options de scaling:

- **Vertical**: Augmentation des ressources par conteneur
- **Horizontal**: Déploiement de plusieurs instances de chaque service
- **Shard**: Division des responsabilités par marché/instrument
- **Caching**: Utilisation de Redis pour réduire la charge de la base de données
- **Traitement asynchrone**: Utilisation de files d'attente pour les tâches intensives

## Dépendances externes

- **yfinance**: Récupération des données de marché
- **OpenRouter API**: Accès à Claude 3.7 pour la validation IA
- **Telegram API**: Notifications en temps réel
- **TensorFlow/Keras**: Frameworks pour les modèles d'apprentissage profond
- **scikit-learn**: Bibliothèque pour les modèles ML classiques
- **Stable Baselines**: Framework pour l'apprentissage par renforcement
- **pandas-ta/talib**: Calcul d'indicateurs techniques
- **Flask**: Framework web pour l'interface utilisateur et l'API
- **Redis**: Message bus et cache pour la communication inter-services
- **PostgreSQL**: Base de données relationnelle pour le stockage persistant
- **Transformers/SentenceTransformer**: Bibliothèques pour l'analyse de texte et NLP 
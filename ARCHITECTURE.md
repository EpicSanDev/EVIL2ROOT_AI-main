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
│                     Client Web (Browser)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Reverse Proxy (Nginx)                    │
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
│                       Message Bus                            │
│                        (Redis)                               │
└────┬─────────────────────┬──────────────────────┬───────────┘
     │                     │                      │
     ▼                     ▼                      ▼
┌──────────────┐    ┌──────────────┐     ┌───────────────┐
│              │    │              │     │               │
│ Trading Bot  │    │ AI Validator │     │ Data Manager  │
│              │    │              │     │               │
└──────┬───────┘    └──────┬───────┘     └───────┬───────┘
       │                   │                     │
       └───────────┬───────┴─────────────┬──────┘
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
   - Le Data Manager récupère les données de marché via des API externes (yfinance, etc.)
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

5. **Présentation Web**:
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

**Fonctionnalités**:
- Validation multi-facteur des décisions de trading
- Utilisation de Claude 3.7 pour analyse avancée
- Vérification de compatibilité avec les tendances de marché
- Calcul de scores de confiance

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

## Base de données

Le système utilise PostgreSQL avec le schéma suivant:

**Tables principales**:
- `trade_history`: Enregistre toutes les transactions exécutées
- `trading_signals`: Stocke les signaux générés par les modèles
- `market_data`: Contient des snapshots de données de marché historiques
- `performance_metrics`: Métriques de performance quotidiennes
- `bot_settings`: Configuration du bot de trading
- `open_positions`: Positions actuellement ouvertes

## Modèles d'IA

Le système utilise plusieurs modèles d'IA complémentaires:

1. **PricePredictionModel** (app/models/price_prediction.py):
   - Réseaux LSTM pour prédire les mouvements de prix
   - Entrée: Données OHLCV et indicateurs techniques
   - Sortie: Prédiction de prix pour différents horizons temporels

2. **IndicatorManagementModel** (app/models/indicator_management.py):
   - Analyse des indicateurs techniques classiques (RSI, MACD, etc.)
   - Détection de patterns graphiques

3. **RiskManagementModel** (app/models/risk_management.py):
   - Évaluation du risque de marché actuel
   - Recommandation de taille de position

4. **TpSlManagementModel** (app/models/tp_sl_management.py):
   - Calcul des niveaux optimaux de take-profit et stop-loss
   - Ajustement dynamique des trailing stops

5. **RLTradingModel** (app/models/rl_trading.py):
   - Agent d'apprentissage par renforcement pour les décisions de trading
   - Utilise Stable Baselines pour l'implémentation

6. **SentimentAnalyzer** (app/models/sentiment_analysis.py):
   - Analyse du sentiment à partir des news financières
   - Utilise des modèles de NLP pré-entraînés
   - Intégration avec des transformers pour l'analyse de texte

## Sécurité

- Toutes les communications sont chiffrées
- Les clés API sont stockées dans des variables d'environnement
- L'accès à la base de données est limité par conteneur
- Les données sensibles ne sont jamais exposées via l'API

## Scalabilité

L'architecture permet plusieurs options de scaling:

- **Vertical**: Augmentation des ressources par conteneur
- **Horizontal**: Déploiement de plusieurs instances de chaque service
- **Shard**: Division des responsabilités par marché/instrument
- **Caching**: Utilisation de Redis pour réduire la charge de la base de données

## Dépendances externes

- **yfinance**: Récupération des données de marché
- **OpenRouter API**: Accès à Claude 3.7 pour la validation IA
- **Telegram API**: Notifications en temps réel
- **TensorFlow/PyTorch**: Frameworks pour les modèles d'apprentissage automatique
- **Stable Baselines**: Framework pour l'apprentissage par renforcement
- **pandas-ta/talib**: Calcul d'indicateurs techniques 
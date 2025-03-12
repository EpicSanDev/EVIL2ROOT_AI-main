# Architecture du Système EVIL2ROOT Trading Bot

## Vue d'ensemble

EVIL2ROOT Trading Bot est conçu comme une architecture de microservices communiquant via Redis et utilisant PostgreSQL comme stockage persistant. Cette architecture permet une haute disponibilité, une séparation claire des responsabilités et une scalabilité facilitée.

## Diagramme d'architecture

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

### 1. Acquisition des données

- **Sources de données** : 
  - API yfinance pour les données historiques et en temps réel
  - APIs financières alternatives pour les données de marché
  - Sources d'actualités financières via News API
  - Données de sentiment des médias sociaux

- **Traitement des données** :
  - Nettoyage et suppression des valeurs aberrantes
  - Normalisation des données à l'aide de StandardScaler et MinMaxScaler
  - Calcul d'indicateurs techniques via talib et pandas-ta
  - Génération de features avancées par ingénierie de caractéristiques

- **Stockage des données** :
  - Stockage des données brutes et traitées dans PostgreSQL
  - Mise en cache des données récentes dans Redis
  - Stockage des séries temporelles optimisé pour l'accès rapide

### 2. Génération des signaux

- **Analyse technique** :
  - Application d'indicateurs techniques (RSI, MACD, Bandes de Bollinger, etc.)
  - Détection de motifs chartistes (triangles, têtes-épaules, etc.)
  - Identification des niveaux de support et résistance

- **Modèles ML/DL** :
  - Prédiction de prix via des réseaux LSTM et GRU bidirectionnels
  - Modèles de reconnaissance de patterns avec CNN 1D
  - Modèles d'ensemble (Random Forest, XGBoost, etc.) pour la classification
  - Transformers pour l'analyse des séquences temporelles

- **Apprentissage par renforcement** :
  - Agent RL entraîné dans un environnement de marché simulé
  - Optimisation de la politique avec PPO (Proximal Policy Optimization)
  - Utilisation de Stable Baselines pour l'implémentation

- **Fusion de signaux** :
  - Combinaison pondérée des signaux de différentes sources
  - Allocation dynamique des poids selon les performances historiques
  - Meta-modèle pour la décision finale de trading

### 3. Validation IA

- **Analyse complète par Claude 3.7** :
  - Analyse du contexte global du marché 
  - Évaluation des risques macroéconomiques
  - Vérification de la cohérence des signaux avec l'actualité

- **Vérifications multi-facteurs** :
  - Compatibilité avec la tendance du marché
  - Évaluation du rapport risque/récompense
  - Analyse des corrélations inter-marchés
  - Détection d'anomalies potentielles

- **Analyse de sentiment** :
  - Traitement des actualités financières récentes
  - Évaluation du sentiment des médias sociaux
  - Analyse des rapports d'analystes

### 4. Exécution des transactions

- **Gestion des risques** :
  - Calcul dynamique de la taille des positions
  - Limites d'exposition par symbole et globales
  - Diversification intelligente du portefeuille

- **Exécution et suivi** :
  - Exécution des ordres validés
  - Gestion des stop-loss et take-profit
  - Ajustement dynamique des trailing stops
  - Journalisation détaillée de toutes les actions

### 5. Analyses planifiées

- **Analyses quotidiennes** :
  - Analyse approfondie des marchés chaque jour
  - Génération de rapports détaillés
  - Ajustement des paramètres de trading

- **Planification des analyses** :
  - Coordination des tâches d'analyse
  - Optimisation des ressources système
  - Priorisation selon les conditions de marché

## Composants techniques

### 1. Trading Bot (app/trading.py)

- **Rôle** : Composant central qui gère la logique de trading
- **Responsabilités** :
  - Récupération et traitement des données de marché
  - Exécution des modèles et génération des signaux
  - Envoi des demandes de validation à l'AI Validator
  - Exécution des transactions validées
  - Gestion des positions ouvertes

### 2. AI Validator (app/ai_trade_validator.py)

- **Rôle** : Système de validation des décisions de trading
- **Responsabilités** :
  - Analyse approfondie des propositions de trading
  - Communication avec Claude 3.7 via OpenRouter
  - Vérification multi-facteur des décisions
  - Calcul des scores de confiance
  - Validation ou rejet des transactions

### 3. Web UI (app/routes.py)

- **Rôle** : Interface utilisateur et API REST
- **Responsabilités** :
  - Affichage du tableau de bord de performance
  - Configuration du système de trading
  - Visualisation des graphiques et indicateurs
  - Historique des transactions
  - Points d'API pour l'interaction avec le système

### 4. Daily Analysis Bot (app/daily_analysis_bot.py)

- **Rôle** : Analyse quotidienne approfondie des marchés
- **Responsabilités** :
  - Analyse technique et fondamentale
  - Génération de rapports détaillés
  - Identification des opportunités à moyen terme
  - Envoi des rapports par Telegram

### 5. Market Analysis Scheduler (app/market_analysis_scheduler.py)

- **Rôle** : Planification et coordination des analyses
- **Responsabilités** :
  - Planification temporelle des analyses
  - Gestion des ressources système
  - Priorisation des tâches d'analyse
  - Coordination entre les différents services

### 6. Model Trainer (app/model_trainer.py)

- **Rôle** : Entraînement et optimisation des modèles ML/DL
- **Responsabilités** :
  - Entraînement régulier des modèles
  - Optimisation des hyperparamètres
  - Évaluation des performances des modèles
  - Gestion des versions des modèles

## Infrastructure et déploiement

### Docker et conteneurisation

- Architecture complète conteneurisée avec Docker
- Services isolés pour une meilleure résilience
- Orchestration via Docker Compose
- Volumes pour la persistance des données

### Base de données

- PostgreSQL pour le stockage principal des données
- Schéma optimisé pour les données de trading et les séries temporelles
- Backup automatisé et stratégie de reprise après incident
- Indexes optimisés pour les requêtes fréquentes

### Messagerie et communication

- Redis comme bus de messages entre les services
- Pub/Sub pour la communication asynchrone
- Stockage temporaire des données en cache
- Files d'attente pour les tâches de longue durée

### Monitoring et logging

- Logging centralisé de tous les composants
- Métriques de performance collectées en temps réel
- Alertes sur les anomalies système
- Tableaux de bord de surveillance

## Sécurité

- Gestion sécurisée des secrets et clés API
- Communication chiffrée entre les composants
- Validation des entrées pour prévenir les injections
- Authentification pour l'accès à l'interface et aux API
- Audit des actions sensibles

## Scalabilité

L'architecture est conçue pour permettre une scalabilité horizontale :

- Services indépendants pouvant être déployés sur plusieurs instances
- Base de données pouvant être mise à l'échelle ou répliquée
- Redis pouvant être configuré en cluster
- Charge de travail distribuable sur plusieurs serveurs

## Perspectives d'évolution

- Intégration de modèles IA plus avancés
- Support pour davantage de marchés et d'actifs
- Architecture encore plus distribuée avec Kubernetes
- Système de backtesting plus sophistiqué
- Interface utilisateur encore plus riche 
# EVIL2ROOT AI Trading System - Structure du projet

Ce projet a été réorganisé pour améliorer la maintenabilité et la clarté du code. Voici la structure des dossiers et leur fonction :

## Structure principale

- **`/app`** : Dossier principal de l'application
  - **`/analytics`** : Composants d'analyse et de validation
  - **`/api`** : Routes API pour l'accès au système
  - **`/bots`** : Bots Telegram et d'analyse quotidienne
  - **`/core`** : Fonctionnalités principales (trading, utilitaires)
  - **`/models`** : Modèles utilisés par le système (voir sous-section)
  - **`/monitoring`** : Surveillance du système
  - **`/plugins`** : Système d'extension par plugins
  - **`/routes`** : Routes web Flask
  - **`/scripts`** : Scripts utilitaires
  - **`/services`** : Services (paiements, notifications, etc.)
  - **`/static`** : Fichiers statiques (CSS, JS, images)
  - **`/templates`** : Templates HTML
  - **`/tests`** : Tests du système
  - **`/ui`** : Composants d'interface utilisateur

## Structure des modèles

Les modèles ont été organisés en sous-dossiers thématiques :

- **`/models/ensemble`** : Modèles d'ensemble combinant plusieurs prédictions
- **`/models/indicators`** : Gestion des indicateurs et stop-loss/take-profit
- **`/models/ml`** : Modèles de machine learning
- **`/models/price`** : Modèles de prédiction de prix
- **`/models/sentiment`** : Analyse de sentiment
- **`/models/trading`** : Gestion des positions et backtesting
- **`/models/users`** : Modèles d'utilisateurs

## Structure des tests

Les tests sont également organisés par thématique :

- **`/tests/bots`** : Tests des bots
- **`/tests/data`** : Tests de récupération et traitement des données
- **`/tests/price_models`** : Tests des modèles de prédiction

## Modules importants

- `app.core.trading` : Classes principales TradingBot et DataManager
- `app.models.trading.position_manager` : Gestion des positions de trading
- `app.api.api` : API REST pour l'interface utilisateur
- `app.monitoring.core.monitoring` : Service de surveillance du système

Chaque dossier contient un fichier README.md qui détaille son contenu et sa fonction.

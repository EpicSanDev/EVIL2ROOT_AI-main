# Guide Utilisateur EVIL2ROOT Trading Bot

Ce guide vous aidera à installer, configurer et utiliser efficacement le système de trading EVIL2ROOT. Il couvre toutes les fonctionnalités principales et fournit des instructions détaillées pour tirer le meilleur parti du système.

## Table des matières

- [Installation](#installation)
  - [Prérequis](#prérequis)
  - [Installation standard](#installation-standard)
  - [Installation avec Docker](#installation-avec-docker)
- [Configuration](#configuration)
  - [Fichier .env](#fichier-env)
  - [Configuration des actifs à trader](#configuration-des-actifs-à-trader)
  - [Configuration de l'IA de validation](#configuration-de-lia-de-validation)
  - [Configuration des notifications](#configuration-des-notifications)
- [Démarrage du système](#démarrage-du-système)
  - [Démarrage standard](#démarrage-standard)
  - [Démarrage avec entraînement forcé](#démarrage-avec-entraînement-forcé)
  - [Planification des analyses de marché](#planification-des-analyses-de-marché)
- [Interface utilisateur web](#interface-utilisateur-web)
  - [Tableau de bord principal](#tableau-de-bord-principal)
  - [Tableau de bord avancé](#tableau-de-bord-avancé)
  - [Gestion des paramètres](#gestion-des-paramètres)
  - [Gestion des plugins](#gestion-des-plugins)
- [Analyse quotidienne du marché](#analyse-quotidienne-du-marché)
  - [Rapports d'analyse](#rapports-danalyse)
  - [Configuration de l'analyse](#configuration-de-lanalyse)
- [Modèles d'IA et entraînement](#modèles-dia-et-entraînement)
  - [Comprendre les modèles](#comprendre-les-modèles)
  - [Entraînement des modèles](#entraînement-des-modèles)
  - [Évaluation des performances](#évaluation-des-performances)
- [Gestion des risques](#gestion-des-risques)
  - [Paramètres de risque](#paramètres-de-risque)
  - [Stop-loss et Take-profit](#stop-loss-et-take-profit)
- [Backtesting](#backtesting)
  - [Exécution d'un backtest](#exécution-dun-backtest)
  - [Analyse des résultats](#analyse-des-résultats)
- [Dépannage](#dépannage)
  - [Problèmes courants](#problèmes-courants)
  - [Logs et diagnostics](#logs-et-diagnostics)
- [Meilleures pratiques](#meilleures-pratiques)
  - [Optimisation des performances](#optimisation-des-performances)
  - [Sécurité](#sécurité)

## Installation

### Prérequis

Avant d'installer le système EVIL2ROOT Trading Bot, assurez-vous que votre système répond aux exigences suivantes:

- **Matériel recommandé**:
  - CPU: 4 cœurs ou plus
  - RAM: 8 Go minimum, 16 Go recommandés
  - Disque: 10 Go d'espace libre minimum
  - Connexion Internet stable

- **Logiciels requis**:
  - Python 3.8 ou supérieur
  - Docker et Docker Compose (pour l'installation avec Docker)
  - Git
  - Compte OpenRouter (pour l'accès à Claude 3.7)
  - Compte Telegram (pour les notifications)

### Installation standard

1. Clonez le dépôt GitHub:
   ```bash
   git clone https://github.com/Evil2Root/EVIL2ROOT_AI.git
   cd EVIL2ROOT_AI
   ```

2. Créez et activez un environnement virtuel Python:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installez les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

4. Copiez le fichier d'exemple d'environnement:
   ```bash
   cp .env.example .env
   ```

5. Modifiez le fichier `.env` avec vos paramètres (voir section [Configuration](#configuration))

### Installation avec Docker

1. Clonez le dépôt GitHub:
   ```bash
   git clone https://github.com/Evil2Root/EVIL2ROOT_AI.git
   cd EVIL2ROOT_AI
   ```

2. Copiez le fichier d'exemple d'environnement:
   ```bash
   cp .env.example .env
   ```

3. Modifiez le fichier `.env` avec vos paramètres (voir section [Configuration](#configuration))

4. Définissez les permissions des scripts:
   ```bash
   chmod +x docker-entrypoint.sh
   chmod +x start_docker.sh
   chmod +x stop_docker.sh
   chmod +x start_market_scheduler.sh
   ```

5. Construisez et démarrez les conteneurs:
   ```bash
   docker compose up --build
   ```
   
   Ou utilisez le script fourni:
   ```bash
   ./start_docker.sh
   ```

## Configuration

### Fichier .env

Le fichier `.env` contient toutes les variables de configuration nécessaires au fonctionnement du système. Voici les principales variables à configurer:

#### Configuration générale

```
# Application configuration
SECRET_KEY=change_this_to_a_secure_random_string
FLASK_APP=run.py
FLASK_ENV=production

# Trading configuration
INITIAL_BALANCE=100000.0
TRADING_SYMBOLS=AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA
TRADING_ENABLED=true
USE_TRANSFORMER_MODEL=true
```

#### Configuration de la base de données

```
# Database configuration
DB_HOST=db
DB_PORT=5432
DB_USER=trading_user
DB_PASSWORD=your_secure_password
DB_NAME=trading_db
DATABASE_URI=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}
```

#### Configuration Redis

```
# Redis configuration
REDIS_HOST=redis
REDIS_PORT=6379
```

#### Configuration du risque

```
# Risk management
MAX_POSITIONS=10
RISK_PER_TRADE=0.02
TRAILING_STOP_ENABLED=true
TRAILING_ACTIVATION_PCT=1.0
TRAILING_DISTANCE_PCT=0.5

# Trading Bot Configuration
CONFIDENCE_THRESHOLD=0.65
ENABLE_LIVE_TRADING=false
```

#### Configuration de l'IA avancée

```
# Advanced AI Configuration (AI Trade Validator)
OPENROUTER_API_KEY=your_openrouter_api_key

# Claude Configuration - Using advanced model for increased precision
CLAUDE_MODEL=anthropic/claude-3.7-sonnet
```

### Configuration des actifs à trader

Vous pouvez configurer les actifs que vous souhaitez trader en modifiant la variable `SYMBOLS` dans le fichier `.env`:

```
# Supported Symbols (comma-separated)
SYMBOLS=AAPL,GOOGL,MSFT,AMZN,TSLA,BTC-USD,ETH-USD
```

Le système prend en charge:
- Actions (ticker standard)
- Crypto-monnaies (format: BTC-USD)
- ETFs
- Indices (format spécifique à yfinance)

### Configuration de l'IA de validation

Pour utiliser la validation avancée avec Claude 3.7, vous devez:

1. Créer un compte sur [OpenRouter](https://openrouter.ai/)
2. Obtenir une clé API
3. Définir cette clé dans la variable `OPENROUTER_API_KEY` du fichier `.env`
4. Configurer le modèle Claude souhaité avec la variable `CLAUDE_MODEL`

Vous pouvez ajuster le seuil de confiance pour la validation en modifiant `CONFIDENCE_THRESHOLD`. Une valeur plus élevée (ex: 0.8) rendra le système plus sélectif, tandis qu'une valeur plus basse (ex: 0.5) acceptera plus de transactions.

### Configuration des notifications

Le système peut envoyer des notifications via Telegram. Pour configurer:

1. Créez un bot Telegram en parlant à [@BotFather](https://t.me/BotFather)
2. Obtenez le token du bot
3. Définissez ce token dans la variable `TELEGRAM_TOKEN` du fichier `.env`
4. Démarrez une conversation avec votre bot
5. Utilisez l'option `/start` pour commencer à recevoir des notifications

## Démarrage du système

### Démarrage standard

Pour démarrer le système en utilisant Docker:

```bash
./start_docker.sh
```

Pour arrêter le système:

```bash
./stop_docker.sh
```

### Démarrage avec entraînement forcé

Si vous souhaitez forcer le système à réentraîner tous les modèles avant de démarrer:

```bash
./start_docker_force_train.sh
```

Cette option est utile après une mise à jour ou lorsque vous remarquez une baisse de performance des modèles.

### Planification des analyses de marché

Pour activer les analyses de marché automatiques planifiées:

```bash
./start_market_scheduler.sh
```

Pour arrêter les analyses planifiées:

```bash
./stop_market_scheduler.sh
```

Ces analyses s'exécutent selon le calendrier défini dans `app/market_analysis_scheduler.py` et envoient des rapports via Telegram et sur l'interface web.

## Interface utilisateur web

L'interface web est accessible à l'adresse http://localhost:5000/ après le démarrage du système.

### Tableau de bord principal

Le tableau de bord principal (`dashboard.html`) affiche:
- Aperçu des performances du portefeuille
- Transactions récentes
- Positions ouvertes
- Signaux de trading actuels
- Graphiques de performance

### Tableau de bord avancé

Le tableau de bord avancé (`advanced_dashboard.html`) offre:
- Visualisations détaillées des actifs
- Indicateurs techniques avancés
- Analyse en profondeur des performances des modèles
- Métriques de trading détaillées
- Rapports d'analyse de marché

Pour y accéder, cliquez sur "Dashboard avancé" dans le menu de navigation.

### Gestion des paramètres

La page de paramètres (`settings.html`) vous permet de:
- Configurer les paramètres de trading (seuils, limites)
- Ajuster les paramètres de gestion des risques
- Configurer les notifications
- Régler les intervalles d'analyse
- Activer/désactiver des fonctionnalités

### Gestion des plugins

La page des plugins (`plugins.html` et `plugin_settings.html`) permet de:
- Activer/désactiver des plugins d'analyse supplémentaires
- Configurer les paramètres des plugins
- Voir les résultats d'analyse des plugins

## Analyse quotidienne du marché

### Rapports d'analyse

Les rapports d'analyse quotidienne incluent:
- Analyse technique des actifs configurés
- Tendances de marché identifiées
- Opportunités de trading détectées
- Sentiment du marché et analyse des actualités
- Prévisions à court et moyen terme

Ces rapports sont accessibles:
- Via l'interface web (section "Analyses")
- Par notifications Telegram
- Dans les logs système

### Configuration de l'analyse

Vous pouvez configurer l'analyse quotidienne en modifiant les paramètres dans le fichier `.env`:

```
# Data Update Frequency (minutes)
DATA_UPDATE_INTERVAL=5
SCANNING_INTERVAL=60
```

Pour une analyse manuelle ponctuelle:

```bash
python start_daily_analysis.py
```

Pour forcer l'entraînement des modèles avant l'analyse:

```bash
python start_daily_analysis.py --force-train
```

## Modèles d'IA et entraînement

### Comprendre les modèles

Le système utilise plusieurs modèles d'IA complémentaires:

1. **Modèle de prédiction de prix**: Prédit les mouvements de prix futurs en utilisant des réseaux LSTM et GRU
2. **Modèle d'indicateurs techniques**: Analyse les indicateurs techniques classiques
3. **Modèle de gestion des risques**: Évalue le risque de chaque transaction
4. **Modèle TP/SL**: Calcule les niveaux optimaux de take-profit et stop-loss
5. **Modèle RL**: Agent d'apprentissage par renforcement pour les décisions de trading
6. **Analyseur de sentiment**: Analyse le sentiment du marché
7. **Modèle Transformer**: Utilise l'architecture Transformer pour analyse de séquence
8. **Récupération de news**: Collecte et analyse les actualités financières

### Entraînement des modèles

L'entraînement des modèles s'effectue automatiquement au démarrage initial du système, mais vous pouvez forcer un réentraînement:

```bash
./start_docker_force_train.sh
```

Ou via Python:

```python
from app.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train_all_models(force_train=True)
```

Les modèles entraînés sont sauvegardés dans le répertoire `saved_models/`.

### Évaluation des performances

Les performances des modèles sont évaluées automatiquement et disponibles:
- Dans l'interface web (section "Modèles")
- Dans les logs (`logs/model_training.log`)
- Dans la base de données (table `model_performance`)

## Gestion des risques

### Paramètres de risque

Les paramètres de gestion des risques sont configurables dans le fichier `.env`:

```
# Risk management
MAX_POSITIONS=10
RISK_PER_TRADE=0.02
TRAILING_STOP_ENABLED=true
TRAILING_ACTIVATION_PCT=1.0
TRAILING_DISTANCE_PCT=0.5
```

- `MAX_POSITIONS`: Nombre maximum de positions ouvertes simultanément
- `RISK_PER_TRADE`: Pourcentage du capital à risquer par transaction (0.02 = 2%)
- `TRAILING_STOP_ENABLED`: Activer/désactiver les trailing stops
- `TRAILING_ACTIVATION_PCT`: Pourcentage de profit avant activation du trailing stop
- `TRAILING_DISTANCE_PCT`: Distance du trailing stop par rapport au prix maximum

### Stop-loss et Take-profit

Le système calcule automatiquement des niveaux de stop-loss et take-profit pour chaque transaction en utilisant:
- L'analyse technique (niveaux de support/résistance)
- La volatilité de l'actif
- Le modèle TpSlManagementModel

Ces niveaux sont ajustables dans l'interface web pour les positions ouvertes.

## Backtesting

### Exécution d'un backtest

Pour exécuter un backtest sur les données historiques:

```bash
python -c "from app.models.backtesting import Backtester; bt = Backtester(); bt.run_backtest('AAPL', '2020-01-01', '2023-01-01')"
```

Vous pouvez aussi utiliser l'interface web:
1. Accédez à la section "Backtesting"
2. Sélectionnez les actifs à tester
3. Définissez la période et les paramètres
4. Exécutez le backtest

### Analyse des résultats

Les résultats du backtest incluent:
- ROI global
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Graphiques de performance
- Analyse transaction par transaction

Ces résultats sont sauvegardés et accessibles via l'interface web.

## Dépannage

### Problèmes courants

**Erreur de connexion à la base de données**
```
Vérifiez que PostgreSQL est en cours d'exécution et que les identifiants dans .env sont corrects.
```

**Erreur d'API OpenRouter**
```
Vérifiez votre clé API dans .env et assurez-vous que votre compte a des crédits suffisants.
```

**Modèles non chargés**
```
Assurez-vous que les modèles sont entraînés en vérifiant le répertoire saved_models/. Si nécessaire, forcez l'entraînement.
```

**Dockerisation échouée**
```
Vérifiez les logs avec 'docker compose logs' pour identifier le problème spécifique.
```

### Logs et diagnostics

Les logs du système sont stockés dans le répertoire `logs/` :
- `trading_bot.log`: Logs principaux du bot de trading
- `ai_validator.log`: Logs du validateur IA
- `model_training.log`: Logs d'entraînement des modèles
- `market_analysis.log`: Logs d'analyse de marché

Pour des diagnostics Docker:
```bash
docker compose logs
```

Pour cibler un service spécifique:
```bash
docker compose logs trading-bot
```

## Meilleures pratiques

### Optimisation des performances

1. **Ajustez les paramètres de risque** selon votre tolérance au risque
2. **Commencez en mode simulation** (ENABLE_LIVE_TRADING=false) avant de passer au trading réel
3. **Réentraînez régulièrement les modèles** pour les adapter aux conditions de marché
4. **Surveillez les performances** via l'interface web et les notifications
5. **Ajustez le CONFIDENCE_THRESHOLD** en fonction des performances observées

### Sécurité

1. **Utilisez toujours des mots de passe forts** pour la base de données et l'interface web
2. **Ne partagez jamais vos clés API** ou identifiants dans des environnements non sécurisés
3. **Sauvegardez régulièrement** la base de données et les modèles entraînés
4. **Mettez à jour régulièrement** le système pour bénéficier des dernières améliorations
5. **Utilisez un VPN** si vous accédez à l'interface web depuis des réseaux non sécurisés

---

Pour toute assistance supplémentaire ou questions, veuillez ouvrir une issue sur le dépôt GitHub du projet ou consulter la documentation détaillée. 
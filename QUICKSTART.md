# 🚀 EVIL2ROOT Trading Bot - Guide de démarrage rapide

Ce guide de démarrage rapide vous aidera à configurer et à utiliser EVIL2ROOT Trading Bot, un système de trading automatisé sophistiqué qui utilise l'intelligence artificielle et l'apprentissage par renforcement.

## Table des matières

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Utilisation](#utilisation)
5. [Modes de fonctionnement](#modes-de-fonctionnement)
6. [FAQ](#faq)

## Prérequis

Avant de commencer, assurez-vous d'avoir :

- **Système d'exploitation** : Linux, macOS ou Windows 10/11
- **Python** : Python 3.8+ (3.9 recommandé)
- **Docker** : (Optionnel mais recommandé) Docker et Docker Compose pour le déploiement containerisé
- **Base de données** : PostgreSQL 13+ (peut être installé automatiquement via Docker)
- **Redis** : Redis 6+ (peut être installé automatiquement via Docker)
- **Git** : Pour cloner le dépôt
- **Clés API** : Clés d'API pour les plateformes d'échange (Binance, Coinbase) et services (Claude/Anthropic)

## Installation

### Méthode 1 : Installation locale (Développement)

1. **Cloner le dépôt**

```bash
git clone https://github.com/votre-username/EVIL2ROOT_AI.git
cd EVIL2ROOT_AI
```

2. **Configurer un environnement virtuel Python**

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Linux/macOS:
source venv/bin/activate
# Sur Windows:
venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
# Installer d'abord les dépendances essentielles
pip install -r requirements-essential.txt

# Installer TA-Lib
# Sur Linux:
./docker/fix-talib-install.sh
# Sur macOS:
./docker/fix-talib-install-alt.sh
# Sur Windows:
python -m pip install --index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28

# Installer toutes les dépendances
pip install -r requirements.txt
```

4. **Initialiser la base de données**

```bash
# Si PostgreSQL est déjà installé
python scripts/init_database.py

# Si vous utilisez Docker pour PostgreSQL
docker-compose -f docker/docker-compose.yml up -d postgres redis
python scripts/init_database.py --docker
```

### Méthode 2 : Installation avec Docker (Recommandé pour la production)

1. **Cloner le dépôt**

```bash
git clone https://github.com/votre-username/EVIL2ROOT_AI.git
cd EVIL2ROOT_AI
```

2. **Configurer les variables d'environnement**

```bash
cp config/secrets.env config/secrets.env.local
# Éditer config/secrets.env.local avec vos clés API et configurations
```

3. **Lancer avec Docker Compose**

```bash
docker-compose up -d
```

Cette commande va construire l'image Docker, créer les conteneurs pour le bot de trading, PostgreSQL et Redis, et démarrer tous les services.

## Configuration

### Variables d'environnement

Les principales variables d'environnement à configurer sont :

- `BINANCE_API_KEY` et `BINANCE_API_SECRET` : Clés API pour Binance
- `COINBASE_API_KEY` et `COINBASE_API_SECRET` : Clés API pour Coinbase
- `ANTHROPIC_API_KEY` : Clé API pour Claude (Anthropic)
- `TELEGRAM_TOKEN` et `TELEGRAM_CHAT_ID` : Pour les notifications Telegram
- `MODE` : Mode de trading (live, paper, backtest)
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` : Configuration de la base de données

Consultez le fichier `config/secrets.env` pour voir toutes les variables disponibles.

### Configuration du trading

Le fichier `config/bot_config.json` contient les configurations principales du bot :

```json
{
  "general": {
    "name": "EVIL2ROOT Trading Bot",
    "version": "1.0.0",
    "mode": "paper",
    "log_level": "INFO"
  },
  "trading": {
    "exchanges": ["binance"],
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "timeframes": ["1h", "4h", "1d"],
    "max_open_positions": 5,
    "risk_per_trade": 0.02,
    "initial_capital": 10000
  },
  "strategies": {
    "default": "hybrid",
    "available": ["technical", "sentiment", "ml", "rl", "hybrid"]
  },
  "risk_management": {
    "max_drawdown": 0.15,
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_atr_multiplier": 3.0,
    "use_trailing_stop": true
  }
}
```

Modifiez ce fichier selon vos préférences de trading.

## Utilisation

### Démarrer le bot

```bash
# Mode papier (simulation sans trading réel)
python src/main.py --mode paper

# Mode backtest (test sur données historiques)
python src/main.py --mode backtest --symbol BTCUSDT --start-date 2023-01-01 --end-date 2023-12-31

# Mode live (trading réel)
python src/main.py --mode live
```

### Utiliser l'bot d'analyse quotidienne

```bash
# Démarrer le bot d'analyse quotidienne
python start_daily_analysis.py
```

Ce script exécutera l'analyse à intervalles réguliers et enverra des rapports via Telegram.

### Surveillance du système

1. **Interface web** : Accédez à http://localhost:5000 pour voir le dashboard (si le service web est activé)
2. **Logs** : Consultez les logs dans le dossier `logs/`
3. **Notifications Telegram** : Recevez des alertes sur votre téléphone via Telegram

## Modes de fonctionnement

### Mode Paper Trading

Le mode "paper" permet de simuler des trades sans risquer de vrais fonds :

```bash
python src/main.py --mode paper
```

Caractéristiques :
- Utilise des données de marché en temps réel
- Simule les exécutions d'ordres
- Calcule les P&L virtuels
- Parfait pour tester les stratégies en conditions réelles

### Mode Backtest

Le mode "backtest" teste les stratégies sur des données historiques :

```bash
python src/main.py --mode backtest --symbol BTCUSDT --start-date 2023-01-01 --end-date 2023-12-31 --strategy hybrid
```

Options disponibles :
- `--strategy` : `technical`, `sentiment`, `ml`, `rl`, ou `hybrid`
- `--symbol` : Paire de trading à tester
- `--start-date` et `--end-date` : Période de backtest
- `--timeframe` : Période temporelle (1h, 4h, 1d)

### Mode Live Trading

⚠️ **ATTENTION : Ce mode implique des trades réels et des risques financiers réels.**

```bash
python src/main.py --mode live
```

Prérequis supplémentaires :
- Clés API avec permissions de trading
- Configuration approfondie des règles de gestion des risques

## FAQ

### Comment puis-je configurer Telegram pour les notifications ?

1. Créez un bot Telegram via [@BotFather](https://t.me/botfather) et obtenez le token
2. Créez un groupe et ajoutez votre bot
3. Obtenez l'ID du chat (utilisez [@RawDataBot](https://t.me/rawdatabot))
4. Configurez les variables `TELEGRAM_TOKEN` et `TELEGRAM_CHAT_ID` dans votre fichier d'environnement

### Comment installer TA-Lib si les méthodes standard échouent ?

Le projet inclut plusieurs alternatives pour installer TA-Lib :

1. Utilisez les scripts fournis dans le dossier `docker/` :
   ```bash
   # Sur Linux
   ./docker/fix-talib-install.sh
   # Sur macOS
   ./docker/fix-talib-install-alt.sh
   ```

2. Utilisez l'image Docker dédiée :
   ```bash
   docker build -f docker/Dockerfile.talib -t talib-builder .
   ```

3. Utilisez l'implémentation mock qui est automatiquement activée en cas d'échec d'installation

### Comment puis-je ajouter une nouvelle stratégie ?

1. Créez une nouvelle classe de stratégie dans `src/core/strategies/`
2. Implémentez au minimum les méthodes `initialize()`, `calculate_signals()` et `generate_trades()`
3. Enregistrez votre stratégie dans le gestionnaire de stratégies
4. Configurez le bot pour utiliser votre stratégie dans `config/bot_config.json`

### Comment optimiser les hyperparamètres des modèles ?

Le projet inclut un système d'optimisation bayésienne :

```bash
python src/utils/optimizer.py --model price_prediction --symbol BTCUSDT --timeframe 1d
```

### Quelles sont les ressources système minimales requises ?

- **Développement** : 4 cœurs CPU, 8 Go RAM, 50 Go espace disque
- **Production légère** : 2 vCPU, 4 Go RAM, 100 Go espace disque
- **Production complète** : 4+ vCPU, 8+ Go RAM, 200+ Go espace disque

### Comment puis-je contribuer au projet ?

1. Forkez le dépôt sur GitHub
2. Clonez votre fork et créez une branche pour votre contribution
3. Développez et testez votre fonctionnalité 
4. Soumettez une Pull Request avec une description détaillée
5. Attendez la revue de code et les commentaires

---

Pour plus d'informations, consultez la [documentation complète](/DOCUMENTATION.md) et le [guide technique](/TECHNICAL_GUIDE.md).

⚠️ **Avertissement : Le trading financier comporte des risques significatifs. Ce bot est fourni à des fins éducatives et de recherche uniquement. Les performances passées ne garantissent pas les résultats futurs.**

# Fonctionnalités Avancées du Bot de Trading

Ce document présente les nouvelles fonctionnalités avancées ajoutées au bot de trading, ainsi que les instructions pour les utiliser et déployer le bot avec Docker et Kubernetes.

## Nouvelles Fonctionnalités

### 1. Apprentissage par Renforcement (RL)

Le bot intègre désormais un système d'apprentissage par renforcement avancé pour les décisions de trading :

- **Agents spécialisés par régime de marché** : Des agents RL sont entraînés pour différents régimes de marché (volatile, stable, bullish, bearish)
- **Multiples algorithmes** : Support pour PPO, SAC et TD3
- **Implémentation PyTorch personnalisée** : Performance optimisée et contrôle total sur les modèles

### 2. Analyse de Sentiment Avancée

Le système d'analyse de sentiment combine plusieurs approches pour une meilleure précision :

- **Modèles spécialisés pour la finance** : Utilisation de FinBERT et autres modèles de NLP
- **Sources multiples** : Agrégation de données de Twitter, NewsAPI, Finnhub
- **Analyse de régime de marché** : Détection intelligente des régimes basée sur le sentiment et la volatilité

### 3. Connexions aux Marchés en Temps Réel

Connexion robuste aux marchés en temps réel :

- **WebSockets performants** : Gestion optimisée des connexions avec reconnexion automatique
- **Support multi-timeframes** : Analyse sur différentes échelles temporelles (1h, 4h, 1d)
- **Implémentation pour Binance** : Prêt à l'emploi pour Binance, extensible à d'autres échanges

### 4. Backtesting Avancé

Fonctionnalités complètes de backtesting pour évaluer les stratégies :

- **Métriques de performance détaillées** : Sharpe ratio, Sortino ratio, drawdowns, etc.
- **Visualisations avancées** : Graphiques de performance, heatmaps mensuelles, analyses des trades
- **Comparaison de stratégies** : Outils pour comparer différentes approches

## Démarrage Rapide

### Configuration des Secrets

1. Copiez le modèle de secrets et ajoutez vos clés API :
   ```bash
   cp config/secrets.env config/secrets.env.local
   ```

2. Éditez le fichier `config/secrets.env.local` avec vos clés API réelles

### Lancement en Mode Backtest

Pour tester une stratégie sur des données historiques :

```bash
python src/main_trading_bot.py --backtest --symbol BTCUSDT --start-date 2023-01-01 --end-date 2023-12-31 --strategy hybrid
```

Options disponibles :
- `--strategy` : Choisissez parmi `technical`, `sentiment`, `rl`, ou `hybrid`
- `--symbol` : Le symbole à tester (par défaut: BTCUSDT)
- `--config` : Chemin vers un fichier de configuration personnalisé

### Lancement en Mode Trading Réel

Pour lancer le bot en mode trading réel :

```bash
python src/main_trading_bot.py
```

## Utilisation avec Docker

### Construction de l'Image

```bash
docker build -t trading-bot:latest .
```

### Exécution avec Docker

```bash
docker run -d --name trading-bot \
  --env-file config/secrets.env.local \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config:/app/config \
  trading-bot:latest
```

### Vérification des Logs

```bash
docker logs -f trading-bot
```

## Déploiement sur Kubernetes

### Prérequis

- kubectl installé et configuré
- Cluster Kubernetes accessible
- Registre Docker configuré

### Pousser l'Image vers un Registre

```bash
docker tag trading-bot:latest your-registry.com/trading-bot:latest
docker push your-registry.com/trading-bot:latest
```

### Déploiement avec le Script

Le script `kubernetes/deploy-trading-bot.sh` automatise le déploiement :

```bash
cd kubernetes
./deploy-trading-bot.sh --registry your-registry.com --namespace trading
```

Options disponibles :
- `--registry` : Spécifie le registre Docker (défaut: localhost:5000)
- `--tag` : Spécifie le tag de l'image (défaut: latest)
- `--namespace` : Spécifie le namespace Kubernetes (défaut: default)
- `--config` : Spécifie le fichier de config Kubernetes (défaut: trading-bot-deployment.yaml)
- `--secrets` : Spécifie le fichier de secrets (défaut: ../config/secrets.env)

### Vérification du Déploiement

```bash
kubectl get pods -n trading -l app=trading-bot
kubectl logs -f -l app=trading-bot -n trading
```

## Configuration Avancée

### Structure du Fichier de Configuration

Le fichier `config/bot_config.json` définit les paramètres du bot :

```json
{
  "trading": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "timeframes": ["1h", "4h", "1d"],
    "initial_capital": 10000,
    "leverage": 1.0,
    "transaction_fee": 0.001,
    "frequency_seconds": 60,
    "dry_run": true
  },
  "risk": {
    "max_position_size": 0.2,
    "max_drawdown": 0.1,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.1
  },
  "strategy": {
    "default": "hybrid",
    "weights": {
      "technical": 0.4,
      "sentiment": 0.3,
      "rl": 0.3
    }
  },
  "sentiment": {
    "update_interval_minutes": 60,
    "sources": ["newsapi", "finnhub"]
  },
  "rl": {
    "model_dir": "saved_models/rl_agents",
    "use_market_regime": true,
    "default_agent": "default_agent",
    "model_type": "PPO"
  }
}
```

### Configuration des Stratégies

#### Stratégie Technique

La stratégie technique utilise des indicateurs comme RSI, MACD et moyennes mobiles. Personnalisez ces paramètres dans le fichier `src/core/backtest_strategies.py`.

#### Stratégie Sentiment

La stratégie basée sur le sentiment utilise l'analyse de sentiment des actualités et réseaux sociaux. Configurez les sources et paramètres dans `config/bot_config.json`.

#### Stratégie RL

La stratégie RL utilise des agents d'apprentissage par renforcement. Vous pouvez entraîner vos propres agents et les spécifier dans la configuration.

#### Stratégie Hybride

La stratégie hybride combine les trois approches précédentes avec des poids configurables dans le fichier `config/bot_config.json`.

## Développement et Extension

### Ajouter un Nouvel Échange

Pour ajouter un nouvel échange, créez une classe qui hérite de `BaseMarketConnector` dans le dossier `src/services/market_data/`.

### Entraîner un Nouvel Agent RL

Pour entraîner un nouvel agent RL :

```python
from src.models.rl.advanced_rl_agent import RLAgentManager

# Créer l'environnement et les données d'entraînement
agent_manager = RLAgentManager(model_dir='saved_models/rl_agents')
agent = agent_manager.create_agent(
    agent_id='my_custom_agent',
    env=my_trading_env,
    model_type='PPO'
)
agent_manager.train_agent(
    agent_id='my_custom_agent',
    total_timesteps=100000
)
```

### Personnaliser l'Analyse de Sentiment

Pour personnaliser l'analyse de sentiment, vous pouvez modifier les classes dans `src/models/sentiment/` ou entraîner votre propre modèle avec la méthode `train_custom_model` de la classe `AdvancedSentimentModel`.

## Dépannage

### Problèmes de Connexion à l'API

Vérifiez que vos clés API sont correctement configurées dans le fichier `config/secrets.env.local`.

### Erreurs de Déploiement Kubernetes

- Vérifiez que les secrets sont bien créés : `kubectl get secrets -n your-namespace`
- Vérifiez les logs du pod : `kubectl logs -f -l app=trading-bot -n your-namespace`
- Vérifiez les événements du pod : `kubectl describe pod -l app=trading-bot -n your-namespace`

### Problèmes de Performance

Si vous rencontrez des problèmes de performance avec le bot, essayez de :
- Réduire le nombre de symboles surveillés
- Augmenter l'intervalle entre les mises à jour de sentiment
- Réduire la fréquence de trading 
# EVIL2ROOT Trading Bot 🤖📈

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)](https://www.postgresql.org/)

Un système de trading automatisé complet avec validation des décisions par IA, apprentissage par renforcement et analyse du sentiment de marché.

![Trading Dashboard](https://via.placeholder.com/800x400?text=EVIL2ROOT+Trading+Dashboard) <!-- Remplacez ceci par une capture d'écran réelle de votre interface -->

## 📋 Table des Matières

- [Caractéristiques](#-caractéristiques)
- [Architecture du Système](#-architecture-du-système)
- [Technologies Utilisées](#-technologies-utilisées)
- [Pour Commencer](#-pour-commencer)
  - [Prérequis](#prérequis)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Utilisation](#-utilisation)
  - [Démarrer le Bot](#démarrer-le-bot)
  - [Interface Web](#interface-web)
  - [Commandes Docker](#commandes-docker)
- [Composants du Système](#-composants-du-système)
  - [Trading Bot](#trading-bot)
  - [AI Validation](#service-de-validation-ia)
  - [Modèles IA](#modèles-ia)
  - [Base de Données](#schéma-de-la-base-de-données)
- [Surveillance et Logs](#-surveillance-et-logs)
- [Backtesting](#-backtesting)
- [Documentation API](#-documentation-api)
- [Développement](#-développement)
- [Comment Contribuer](#-comment-contribuer)
- [FAQ](#-faq)
- [Performances](#-performances)
- [Feuille de Route](#-feuille-de-route)
- [Licence](#-licence)
- [Sécurité](#-sécurité)
- [Avertissement](#-avertissement)
- [Contact](#-contact)
- [Fonctionnalités d'analyse avec entraînement préalable des modèles](#fonctionnalités-d'analyse-avec-entraînement-préalable-des-modèles)

## �� Caractéristiques

- **Modèles de Trading Multiples**: 
  - Indicateurs techniques traditionnels
  - Prédiction de prix par apprentage profond
  - Apprentissage par renforcement
  - Analyse de sentiment du marché
  - Analyse de news financières

- **Validation IA des Transactions**: 
  - Système IA secondaire validant chaque décision de trading
  - Utilisation de Claude 3.7 via l'API OpenRouter pour une analyse avancée
  - Vérification multi-facteurs des décisions de trading

- **Gestion Avancée des Risques**:
  - Calcul dynamique de la taille des positions
  - Gestion automatique des stop-loss et take-profit
  - Trailing stops intelligents

- **Persistance Complète**:
  - Base de données PostgreSQL pour toutes les données de trading
  - Historique complet des transactions et signaux
  - Métriques de performance stockées pour analyse

- **Interface Web Intuitive**:
  - Tableau de bord en temps réel
  - Visualisation avancée des métriques
  - Configuration facile des paramètres de trading

- **Notifications Temps Réel**:
  - Alertes Telegram pour les transactions et événements importants
  - Notifications configurables selon vos préférences

- **Infrastructure Robuste**:
  - Configuration Docker complète
  - Microservices bien isolés
  - Haute disponibilité et résilience

## 🏗 Architecture du Système

Le système est conçu comme une architecture microservices, avec plusieurs composants conteneurisés interagissant via Redis et une base de données PostgreSQL partagée:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Trading Bot   │◄──►│  AI Validator   │◄──►│     Web UI      │
│                 │    │                 │    │                 │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                     Redis Message Bus                       │
│                                                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                   PostgreSQL Database                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Composants Principaux:

- **Trading Bot**: Logique de trading principale et exécution de modèles
- **AI Validation**: Système IA secondaire qui valide les décisions de trading
- **Web UI**: Tableau de bord basé sur Flask pour la surveillance
- **PostgreSQL**: Base de données pour stocker les données de trading et les métriques
- **Redis**: Communication entre les services de trading

## 🔧 Technologies Utilisées

- **Backend**: Python 3.8+, Flask, Redis
- **Base de Données**: PostgreSQL
- **Modèles IA**: 
  - TensorFlow/Keras pour les réseaux de neurones
  - scikit-learn pour les modèles classiques
  - Stable Baselines pour l'apprentissage par renforcement
  - Claude 3.7 pour la validation IA avancée
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Infrastructure**: Docker, Docker Compose
- **APIs Externes**: 
  - yfinance pour les données de marché
  - OpenRouter pour l'accès à Claude 3.7
  - API Telegram pour les notifications
- **Librairies de Trading**: pandas-ta, talib

## 🚦 Pour Commencer

### Prérequis

- Docker et Docker Compose
- Python 3.8+ (pour le développement local)
- Compte OpenRouter pour l'API Claude (pour la validation IA)
- Minimum 8GB de RAM recommandé pour les performances optimales

### Installation

1. Clonez le dépôt:
   ```bash
   git clone https://github.com/Evil2Root/EVIL2ROOT_AI.git
   cd EVIL2ROOT_AI
   ```

2. Configurez les variables d'environnement:
   ```bash
   cp .env.example .env
   ```
   Modifiez le fichier `.env` avec vos paramètres et clés API.

3. Définissez les permissions des scripts d'entrée:
   ```bash
   chmod +x docker/services/entrypoint-*.sh
   ```

4. Construisez et démarrez les conteneurs:
   ```bash
   # Utiliser docker compose directement
   docker compose up --build
   
   # OU en utilisant le Makefile fourni
   make build
   make up
   ```

### Configuration

Options de configuration clés dans `.env`:

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `ENABLE_LIVE_TRADING` | Activer le trading en direct | `false` |
| `RISK_PER_TRADE` | Pourcentage de risque par transaction | `0.02` (2%) |
| `CONFIDENCE_THRESHOLD` | Confiance minimale de l'IA pour valider | `0.75` |
| `SYMBOLS` | Liste de symboles à trader | `AAPL,MSFT,GOOGL` |
| `TELEGRAM_TOKEN` | Token du bot Telegram | - |
| `OPENROUTER_API_KEY` | Clé API OpenRouter pour Claude | - |
| `CLAUDE_MODEL` | ID du modèle Claude | `anthropic/claude-3.7` |

Consultez le fichier `.env.example` pour la liste complète des variables de configuration.

## 🎮 Utilisation

### Démarrer le Bot

```bash
# Démarrer tous les services en arrière-plan
make up

# Démarrer avec les logs visibles dans le terminal
make up-log
```

### Interface Web

Accédez à l'interface web à l'adresse http://localhost:5001/ pour:
- Voir le tableau de bord des performances
- Consulter l'historique des transactions
- Configurer les paramètres du bot
- Visualiser les graphiques et indicateurs
- Surveiller les positions ouvertes

### Commandes Docker (via Makefile)

| Commande | Description |
|---------|-------------|
| `make build` | Construire ou reconstruire tous les conteneurs |
| `make up` | Démarrer tous les services en arrière-plan |
| `make up-log` | Démarrer tous les services avec logs visibles |
| `make down` | Arrêter tous les services |
| `make logs` | Afficher les logs de tous les services |
| `make logs-SERVICE` | Afficher les logs d'un service spécifique (ex., `make logs-trading-bot`) |
| `make ps` | Lister les conteneurs en cours d'exécution et leur statut |
| `make restart` | Redémarrer tous les services |
| `make restart-SERVICE` | Redémarrer un service spécifique (ex., `make restart-web-ui`) |
| `make clean` | Supprimer tous les conteneurs et volumes |
| `make shell-SERVICE` | Ouvrir un shell dans un conteneur (ex., `make shell-trading-bot`) |
| `make backup` | Sauvegarder la base de données dans un fichier SQL |
| `make db-cli` | Ouvrir l'interface en ligne de commande PostgreSQL |
| `make redis-cli` | Ouvrir l'interface en ligne de commande Redis |
| `make test` | Exécuter les tests à l'intérieur du conteneur |

## 🧩 Composants du Système

### Trading Bot

Le composant de trading principal qui:
- Récupère et traite les données de marché en temps réel
- Exécute plusieurs modèles de trading pour générer des signaux
- Envoie des demandes de transactions au service de validation IA
- Exécute les transactions validées
- Gère les positions ouvertes avec trailing stops et take-profits

#### Modèles IA

Le système intègre plusieurs modèles d'IA pour différents aspects du trading:

1. **PricePredictionModel**: Utilise des réseaux de neurones LSTM pour prédire les mouvements de prix
2. **IndicatorManagementModel**: Analyse les indicateurs techniques classiques
3. **RiskManagementModel**: Évalue le risque de chaque transaction potentielle
4. **TpSlManagementModel**: Détermine les niveaux optimaux de take-profit et stop-loss
5. **RLTradingModel**: Agent d'apprentissage par renforcement pour les décisions de trading
6. **SentimentAnalyzer**: Analyse le sentiment du marché à partir des news et médias sociaux

### Service de Validation IA

Un service IA secondaire qui:
- Valide les décisions de trading du bot principal
- Vérifie si la transaction s'aligne avec les tendances du marché sur plusieurs périodes
- S'assure que les niveaux de risque sont acceptables
- Fournit des scores de confiance pour les décisions de trading
- Utilise Claude 3.7 via l'API OpenRouter pour une analyse avancée

### Schéma de la Base de Données

La base de données PostgreSQL comprend:
- `trade_history`: Historique de toutes les transactions
- `trading_signals`: Signaux de trading générés par les modèles
- `market_data`: Snapshots de données historiques du marché
- `performance_metrics`: Statistiques quotidiennes de performance de trading
- `bot_settings`: Paramètres de configuration pour le bot de trading

## 📊 Surveillance et Logs

- **Interface web**: http://localhost:5001/
- **Logs de trading**: Consultez `logs/trading_bot.log`
- **Logs de validation IA**: Consultez `logs/ai_validator.log`
- **Logs des conteneurs**: `make logs` ou `make logs-SERVICE`
- **Métriques de performance**: Disponibles dans l'interface web et en base de données

## 📈 Backtesting

Le système inclut des capacités de backtesting complètes pour évaluer les stratégies:

```bash
# Utilisation du Makefile
make shell-trading-bot
python -c "from app.trading import TradingBot; bot = TradingBot(); bot.run_backtest('data/market_data_cleaned.csv')"

# Ou directement avec docker compose
docker compose run trading-bot python -c "from app.trading import TradingBot; bot = TradingBot(); bot.run_backtest('data/market_data_cleaned.csv')"
```

Le backtesting génère un rapport détaillé incluant:
- Return on Investment (ROI) global
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Graphiques de performance

## 📘 Documentation API

Le système expose une API REST pour l'intégration avec d'autres services:

- `/api/v1/trade-history`: Récupérer l'historique des transactions
- `/api/v1/active-positions`: Consulter les positions ouvertes
- `/api/v1/performance`: Obtenir les métriques de performance
- `/api/v1/signals`: Récupérer les signaux de trading récents

Documentation complète de l'API disponible à l'adresse `/api/docs` dans l'interface web.

## 💻 Développement

Pour le développement local en dehors de Docker:

```bash
# Installer les dépendances
pip install -r requirements.txt

# Exécuter le bot en mode développement
python run.py
```

### Tests

```bash
# Exécuter tous les tests
python -m pytest tests/

# Exécuter des tests spécifiques
python -m pytest tests/test_trading.py
```

## 🤝 Comment Contribuer

Les contributions sont les bienvenues! Veuillez suivre ces étapes:

1. Forker le dépôt
2. Créer une branche de fonctionnalité (`git checkout -b feature/fonctionnalite-incroyable`)
3. Validez vos modifications (`git commit -m 'Ajouter une fonctionnalité incroyable'`)
4. Poussez vers la branche (`git push origin feature/fonctionnalite-incroyable`)
5. Ouvrez une Pull Request

Veuillez consulter [CONTRIBUTING.md](CONTRIBUTING.md) pour plus de détails sur notre code de conduite et notre processus de soumission de pull requests.

## ❓ FAQ

**Q: Le bot peut-il négocier sur des marchés de crypto-monnaies?**  
R: Oui, le système prend en charge les actions, les crypto-monnaies et les forex. Configurez les marchés souhaités dans le fichier `.env`.

**Q: Quelles sont les exigences matérielles minimales?**  
R: 4GB de RAM et 2 cœurs CPU sont le minimum recommandé. 8GB de RAM et 4 cœurs sont optimaux pour l'exécution de tous les modèles.

**Q: Le système peut-il fonctionner sans la validation Claude IA?**  
R: Oui, définissez `ENABLE_CLAUDE_VALIDATION=false` dans votre fichier `.env`. Le système utilisera alors uniquement les modèles internes.

**Q: Quelle est la fréquence de mise à jour des données?**  
R: Par défaut, le système actualise les données de marché toutes les 5 minutes, mais c'est configurable via `UPDATE_INTERVAL` dans `.env`.

## 📝 Performances

Les performances varient selon les marchés et la configuration, mais nos tests montrent typiquement:

- ROI annualisé: 15-25% (backtesting)
- Sharpe Ratio: 1.2-1.8
- Maximum Drawdown: 10-15%
- Win Rate: 55-65%

*Note: Les performances passées ne garantissent pas les résultats futurs.*

## 🛣 Feuille de Route

- [ ] Intégration de modèles IA génératifs supplémentaires
- [ ] Support pour les options et les futures
- [ ] Application mobile de surveillance
- [ ] Optimisation automatique des hyperparamètres
- [ ] Support multi-comptes
- [ ] Interface d'administration améliorée
- [ ] Support pour des courtiers supplémentaires

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🔒 Sécurité

Si vous découvrez une vulnérabilité de sécurité, veuillez envoyer un e-mail à evil2root@protonmail.com au lieu d'utiliser l'outil de suivi des problèmes. Nous prendrons les mesures nécessaires pour résoudre le problème rapidement.

Consultez [SECURITY.md](SECURITY.md) pour plus de détails sur notre politique de sécurité.

## ⚠️ Avertissement

Ce logiciel est fourni à des fins éducatives uniquement. Le trading comporte des risques inhérents. Les auteurs ne sont pas responsables des pertes financières pouvant résulter de l'utilisation de ce logiciel. Utilisez-le à vos propres risques et consultez toujours un conseiller financier professionnel.

## 📬 Contact


---

<p align="center">
  Développé avec ❤️ par l'équipe EVIL2ROOT
</p>

## Fonctionnalités d'analyse avec entraînement préalable des modèles

Le système d'analyse prend désormais en charge l'entraînement obligatoire des modèles avant de commencer les analyses. Cette fonctionnalité garantit que les modèles sont correctement entraînés avant d'envoyer des analyses, ce qui améliore la qualité et la fiabilité des prédictions.

### Utilisation

Vous pouvez utiliser le script `start_train_and_analyze.sh` pour lancer le bot d'analyse avec un entraînement forcé des modèles :

```bash
./start_train_and_analyze.sh
```

Alternativement, vous pouvez utiliser l'option `--force-train` avec le script Python directement :

```bash
python3 start_daily_analysis.py --force-train
```

### Fonctionnement

Lorsque cette fonctionnalité est activée :

1. Le système vérifie si des modèles existants sont présents dans le répertoire `saved_models`
2. Si l'option `--force-train` est utilisée, les modèles existants sont ignorés et de nouveaux modèles sont entraînés
3. Le système envoie une notification via Telegram pour informer que l'entraînement des modèles est en cours
4. Une fois l'entraînement terminé, les analyses sont générées et envoyées

### Configuration Docker

Pour Docker, vous pouvez également forcer l'entraînement des modèles en définissant la variable d'environnement `FORCE_MODEL_TRAINING=true` dans votre fichier `.env` ou dans la commande Docker :

```bash
docker-compose run -e FORCE_MODEL_TRAINING=true analysis-bot
```

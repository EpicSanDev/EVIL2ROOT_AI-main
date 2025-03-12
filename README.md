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
- [Fonctionnalités d'analyse avec entraînement préalable des modèles](#fonctionnalités-danalyse-avec-entraînement-préalable-des-modèles)

## 🚀 Caractéristiques

- **Modèles de Trading Multiples**: 
  - Indicateurs techniques traditionnels
  - Prédiction de prix par apprentage profond (LSTM, GRU, Transformers)
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

- **Analyse Quotidienne et Planifiée**:
  - Analyses automatiques programmées du marché
  - Rapports d'analyse détaillés
  - Adaptabilité aux conditions changeantes du marché

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
- **AI Validator**: Système IA secondaire qui valide les décisions de trading
- **Web UI**: Tableau de bord basé sur Flask pour la surveillance
- **PostgreSQL**: Base de données pour stocker les données de trading et les métriques
- **Redis**: Communication entre les services de trading

## 🔧 Technologies Utilisées

- **Backend**: Python 3.8+, Flask, Redis
- **Base de Données**: PostgreSQL
- **Modèles IA**: 
  - TensorFlow/Keras pour les réseaux de neurones (LSTM, GRU, Conv1D)
  - scikit-learn pour les modèles classiques
  - Stable Baselines pour l'apprentissage par renforcement
  - Claude 3.7 pour la validation IA avancée
  - Transformers et SentenceTransformer pour l'analyse de texte
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
   chmod +x docker-entrypoint.sh
   chmod +x start_docker.sh
   chmod +x stop_docker.sh
   ```

4. Construisez et démarrez les conteneurs:
   ```bash
   # Utiliser docker compose directement
   docker compose up --build
   
   # OU en utilisant le script de démarrage
   ./start_docker.sh
   ```

### Configuration

Options de configuration clés dans `.env`:

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `ENABLE_LIVE_TRADING` | Activer le trading en direct | `false` |
| `RISK_PER_TRADE` | Pourcentage de risque par transaction | `0.02` (2%) |
| `CONFIDENCE_THRESHOLD` | Confiance minimale de l'IA pour valider | `0.65` |
| `SYMBOLS` | Liste de symboles à trader | `AAPL,MSFT,GOOGL,AMZN,TSLA,BTC-USD,ETH-USD` |
| `TELEGRAM_TOKEN` | Token du bot Telegram | - |
| `OPENROUTER_API_KEY` | Clé API OpenRouter pour Claude | - |
| `CLAUDE_MODEL` | ID du modèle Claude | `anthropic/claude-3.7-sonnet` |

Consultez le fichier `.env.example` pour la liste complète des variables de configuration.

## 🎮 Utilisation

### Démarrer le Bot

```bash
# Démarrer tous les services en arrière-plan
./start_docker.sh

# Pour arrêter tous les services
./stop_docker.sh

# Pour démarrer avec entraînement forcé des modèles
./start_docker_force_train.sh

# Pour démarrer l'analyse de marché planifiée
./start_market_scheduler.sh
```

### Interface Web

Accédez à l'interface web à l'adresse http://localhost:5000/ pour:
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

Fichiers principaux: `app/trading.py`, `app/market_analysis_scheduler.py`, `app/daily_analysis_bot.py`

### Service de Validation IA

Fournit une validation indépendante des décisions de trading:
- Analyse les signaux de trading à l'aide de multiples modèles
- Utilise Claude 3.7 pour une analyse avancée des décisions
- Génère des scores de confiance et des explications détaillées
- Intègre l'analyse du sentiment de marché dans les décisions

Fichier principal: `app/ai_trade_validator.py`

### Modèles IA

Le système intègre plusieurs modèles d'IA pour différents aspects du trading:

1. **PricePredictionModel** (`app/models/price_prediction.py`): Utilise des réseaux de neurones LSTM et GRU pour prédire les mouvements de prix
2. **IndicatorManagementModel** (`app/models/indicator_management.py`): Analyse les indicateurs techniques classiques
3. **RiskManagementModel** (`app/models/risk_management.py`): Évalue le risque de chaque transaction potentielle
4. **TpSlManagementModel** (`app/models/tp_sl_management.py`): Détermine les niveaux optimaux de take-profit et stop-loss
5. **RLTradingModel** (`app/models/rl_trading.py`): Agent d'apprentissage par renforcement pour les décisions de trading
6. **SentimentAnalyzer** (`app/models/sentiment_analysis.py`): Analyse le sentiment du marché à partir des news et médias sociaux
7. **TransformerModel** (`app/models/transformer_model.py`): Utilise l'architecture Transformer pour analyse de séquence
8. **NewsRetrieval** (`app/models/news_retrieval.py`): Système de collecte et analyse des actualités financières

## 📊 Surveillance et Logs

Le système offre plusieurs niveaux de surveillance:

- **Logs Détaillés**: Tous les services génèrent des logs complets dans le répertoire `logs/`
- **Interface Web**: Métriques et états visualisés en temps réel
- **Notifications Telegram**: Alertes configurables pour les événements importants
- **Monitoring Système**: Suivi des performances et de l'état des services via `app/monitoring.py`

## 🧪 Backtesting

Le système inclut des fonctionnalités avancées de backtesting:

- Test des stratégies sur données historiques
- Évaluation des performances des modèles
- Optimisation des paramètres de trading
- Génération de rapports détaillés

Utilisez le module `app/models/backtesting.py` pour ces fonctionnalités.

## 💻 Développement

Pour contribuer au développement du projet:

1. Créez une branche pour votre fonctionnalité
2. Suivez les conventions de codage du projet
3. Ajoutez des tests pour vos nouvelles fonctionnalités
4. Soumettez une pull request

Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour plus de détails.

## 🔒 Sécurité

Le projet prend la sécurité au sérieux:

- Toutes les communications sont chiffrées
- Les clés API sont stockées dans des variables d'environnement
- L'accès à la base de données est limité par conteneur

Consultez [SECURITY.md](SECURITY.md) pour plus d'informations.

## ⚠️ Avertissement

Ce système est fourni à des fins éducatives et de recherche. Le trading comporte des risques financiers significatifs. Utilisez à vos propres risques.

## 📞 Contact

Pour toute question ou support, veuillez ouvrir une issue sur le dépôt GitHub ou contacter les mainteneurs via les coordonnées indiquées dans le projet.

## 🔄 Fonctionnalités d'analyse avec entraînement préalable des modèles

Le système inclut des fonctionnalités d'analyse avancées qui nécessitent un entraînement préalable des modèles:

- Analyse quotidienne automatisée des marchés via `start_daily_analysis.py`
- Entraînement périodique des modèles pour maintenir leur précision
- Analyse planifiée du marché avec `market_analysis_scheduler.py`
- Option d'entraînement forcé via `start_docker_force_train.sh`

Ces fonctionnalités permettent au système de s'adapter continuellement aux conditions changeantes du marché et d'améliorer ses performances au fil du temps.

## 🔧 Résumé de la solution

Problème identifié : Le problème principal était l'absence du module skopt (scikit-optimize), qui empêchait l'importation de presque tous les modules de l'application.

Dépendances manquantes : Nous avons identifié et installé les dépendances manquantes :
- scikit-optimize (qui fournit le module skopt)
- prometheus_client
- psycopg2-binary

Vérification des signatures : Nous avons vérifié que toutes les méthodes train dans les classes de modèles exigent bien le paramètre symbol, et que les appels à ces méthodes dans le code fournissent correctement ce paramètre.

Tests : Nous avons créé et exécuté plusieurs scripts de test pour vérifier que le problème est résolu :
- minimal_test.py : Pour tester l'importation des modules
- standalone_test.py : Pour tester le fonctionnement des modèles sans dépendre des modules de l'application
- check_signatures.py : Pour vérifier les signatures des méthodes train

Résultat : L'application fonctionne maintenant correctement, et nous avons confirmé que le problème était bien lié aux dépendances manquantes et non à un bug dans le code.

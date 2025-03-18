# 🤖 EVIL2ROOT Trading Bot 📈

<div align="center">

![EVIL2ROOT Trading Banner](https://via.placeholder.com/1200x300?text=EVIL2ROOT+Trading+Bot)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)](https://www.postgresql.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Claude 3.7](https://img.shields.io/badge/Claude-3.7-blueviolet.svg)](https://claude.ai/)

**Un système de trading automatisé complet avec validation des décisions par IA, 
apprentissage par renforcement et analyse du sentiment de marché.**

</div>

## 🌟 Points Forts

- **Intelligence Artificielle Avancée** : Utilisation de modèles d'ensemble combinant apprentissage profond, réseaux de neurones et LLM pour des analyses multidimensionnelles
- **Analyse de Sentiment Multi-sources** : Traitement en temps réel des actualités, médias sociaux et rapports financiers
- **Architecture Évolutive** : Système modulaire facilement extensible et adaptable aux changements de marché
- **Backtesting Robuste** : Simulation précise sur données historiques avec ajustement pour la liquidité et le slippage
- **Performance Optimisée** : Trading haute fréquence avec latence minimisée et exécution efficace
- **Sécurité Renforcée** : Chiffrement bout-en-bout, authentification à deux facteurs et audit de sécurité
- **Multi-exchange Support** : Compatible avec les principales plateformes d'échange (Binance, Coinbase, FTX, etc.)
- **Multi-actifs** : Trading sur cryptomonnaies, actions, forex, matières premières et dérivés
- **API Complète** : Intégration facile avec des systèmes externes via une API RESTful documentée
- **Gestion Dynamique des Risques** : Calcul intelligent des tailles de position et niveaux stop-loss/take-profit
- **Surveillance en Temps Réel** : Interface web intuitive avec tableaux de bord complets et alertes instantanées

## 🚀 Architecture de Déploiement

Le bot utilise une architecture optimisée pour DigitalOcean qui sépare le build de l'exécution :

```mermaid
graph LR
    A[GitHub] -->|Push| B[GitHub Actions]
    B -->|Copie le code| C[Droplet Builder]
    C -->|Build| D[Image Docker]
    D -->|Push| E[Container Registry]
    E -->|Deploy| F[App Platform]
```

### Configuration initiale (à faire UNE SEULE FOIS)

Pour mettre en place l'environnement de build :

```bash
# Exécuter UNE SEULE FOIS pour configurer l'environnement
./scripts/setup-builder-droplet.sh VOTRE_TOKEN_DIGITALOCEAN
```

Ce script va :
1. Créer une Droplet DigitalOcean dédiée au build
2. Configurer un Container Registry
3. Préparer les secrets pour GitHub Actions

### Workflow automatique à chaque push

Une fois la configuration initiale terminée, à chaque push sur la branche main :

1. GitHub Actions déclenche le build sur la Droplet existante
2. L'image Docker est construite et poussée vers le Container Registry
3. App Platform déploie automatiquement la nouvelle version

⚠️ **Important** : Le script `setup-builder-droplet.sh` ne doit être exécuté qu'une seule fois lors de la configuration initiale.

---

## 📋 Table des Matières

- [🚀 Présentation](#-présentation)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🏗️ Architecture du Système](#️-architecture-du-système)
- [🧠 Modèles IA Intégrés](#-modèles-ia-intégrés)
- [🚦 Pour Commencer](#-pour-commencer)
- [🎮 Guide d'Utilisation](#-guide-dutilisation)
- [⚙️ Configuration](#️-configuration)
- [📊 Visualisation et Surveillance](#-visualisation-et-surveillance)
- [📚 Documentation](#-documentation)
- [💻 Développement](#-développement)
- [📝 Licence](#-licence)
- [⚠️ Avertissement](#️-avertissement)

---

## 🚀 Présentation

EVIL2ROOT Trading Bot est une plateforme de trading automatisé conçue pour exploiter les dernières avancées en intelligence artificielle, en traitement des données et en analyse de marché. Le système exploite les synergies entre l'analyse technique traditionnelle, le deep learning et l'analyse de sentiment pour générer des signaux de trading robustes et fiables.

Ce qui distingue EVIL2ROOT des autres solutions de trading est sa couche de validation IA qui utilise Claude 3.7 pour analyser et vérifier chaque décision de trading en fonction de multiples facteurs (conditions de marché, données macroéconomiques, actualités financières et plus encore).

<div align="center">
  <img src="https://via.placeholder.com/800x400?text=EVIL2ROOT+Trading+Dashboard" alt="Dashboard EVIL2ROOT"/>
  <p><em>Interface de trading intuitive avec analyse en temps réel</em></p>
</div>

---

## ✨ Fonctionnalités

### 🔮 Prédiction et Analyse
- **Prédiction de Prix Avancée** : Modèles LSTM, GRU et Transformer pour l'analyse de séries temporelles
- **Analyse Technique Automatisée** : Plus de 50 indicateurs calculés et analysés en temps réel
- **Analyse de Sentiment** : Traitement du langage naturel sur les actualités financières et les médias sociaux
- **Détection de Patterns** : Reconnaissance automatique des figures chartistes et configurations de prix

### 🛡️ Gestion des Risques
- **Sizing Dynamique** : Ajustement automatique de la taille des positions selon le risque
- **Stop-Loss Intelligents** : Placement optimal des stops basé sur la volatilité et les supports/résistances
- **Take-Profit Adaptatifs** : Objectifs de profit ajustés selon les conditions de marché
- **Trailing Stops** : Suivi dynamique des positions gagnantes pour maximiser les profits

### 🔍 Validation et Décision
- **Double Validation IA** : Chaque signal est validé par un système IA secondaire
- **Analyse Multi-actifs** : Corrélations entre marchés pour des décisions plus robustes
- **Filtres de Volatilité** : Protection contre les mouvements erratiques du marché
- **Scores de Confiance** : Attribution de niveaux de confiance à chaque signal généré

### 📱 Interface et Notifications
- **Dashboard en Temps Réel** : Visualisation claire de toutes les positions et analyses
- **Notifications Configurables** : Alertes Telegram pour chaque action importante
- **Journalisation Détaillée** : Historique complet des transactions et des décisions
- **Rapports d'Analyse** : Génération automatique de rapports quotidiens et hebdomadaires

---

## 🏗️ Architecture du Système

EVIL2ROOT Trading Bot est structuré selon une architecture microservices moderne, permettant une scalabilité optimale et une maintenance facilitée.

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

### 📂 Nouvelle Structure du Projet

Nous avons récemment restructuré le projet pour améliorer son organisation et sa maintenabilité :

```
EVIL2ROOT_AI-main/
│
├── src/                    # Code source principal
│   ├── core/               # Fonctionnalités principales et moteur de trading
│   ├── models/             # Modèles d'IA et de prédiction
│   ├── services/           # Services divers (notifications, paiements)
│   ├── api/                # API et endpoints
│   ├── utils/              # Utilitaires communs
│   └── ui/                 # Interface utilisateur
│
├── tests/                  # Tests unitaires et d'intégration
├── config/                 # Fichiers de configuration
├── scripts/                # Scripts utilitaires
├── docs/                   # Documentation
├── docker/                 # Fichiers Docker
└── kubernetes/             # Configuration Kubernetes
```

Pour plus de détails sur la structure, consultez [le document de restructuration](docs/NOUVELLE_STRUCTURE.md).

---

## 🧠 Modèles IA Intégrés

### 🔄 Modèles Prédictifs
- **LSTM & GRU** : Réseaux récurrents pour la prévision de mouvements de prix
- **Transformer** : Architecture d'attention pour l'analyse de séquences temporelles
- **Conv1D** : Réseaux convolutifs pour la détection de patterns dans les graphiques

### 📊 Modèles d'Analyse Technique
- **Modèles d'Ensemble** : Random Forest et XGBoost pour l'analyse d'indicateurs
- **Détecteurs de Patterns** : Reconnaissance des figures chartistes classiques
- **Analyseurs de Tendance** : Identification des phases de marché et retournements

### 📰 Modèles d'Analyse de Sentiment
- **BERT & RoBERTa** : Modèles de langage pour l'analyse d'actualités financières
- **SentenceTransformer** : Extraction de sentiment à partir des médias sociaux
- **Analyseur de Volatilité Implicite** : Évaluation de la peur/avidité du marché

### 🤖 Validation IA
- **Claude 3.7** : Grand modèle de langage pour la validation avancée des décisions
- **Système de Raisonnement Critique** : Évaluation multi-facteurs des opportunités
- **Analyseur de Contexte Macro** : Prise en compte des facteurs économiques globaux

<div align="center">
  <img src="https://via.placeholder.com/800x500?text=EVIL2ROOT+AI+Models" alt="Modèles IA EVIL2ROOT"/>
  <p><em>Architecture des modèles IA intégrés</em></p>
</div>

---

## 🚦 Pour Commencer

### 📋 Prérequis
- Docker et Docker Compose
- Python 3.8+ (pour le développement local)
- Compte OpenRouter pour l'API Claude (validation IA)
- Minimum 8GB de RAM recommandé (16GB pour l'entraînement des modèles)

### 🔧 Installation Standard

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/EpicSanDev/EVIL2ROOT_AI-main.git
   cd EVIL2ROOT_AI-main
   ```

2. Exécutez le script d'installation :
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. Configurez votre environnement :
   ```bash
   # Modifiez le fichier .env avec vos paramètres
   nano .env
   ```

4. Lancez l'application :
   ```bash
   source venv/bin/activate
   python src/main.py
   ```

### 🐳 Installation avec Docker

1. Clonez le dépôt et configurez l'environnement :
   ```bash
   git clone https://github.com/EpicSanDev/EVIL2ROOT_AI-main.git
   cd EVIL2ROOT_AI-main
   cp config/environments/.env.example .env
   ```

2. Personnalisez votre configuration :
   ```bash
   # Modifiez le fichier .env avec vos paramètres
   nano .env
   ```

3. Démarrez avec Docker Compose :
   ```bash
   cd docker
   docker-compose up -d
   ```

---

## 🎮 Guide d'Utilisation

### 💻 Interface Web
Accédez à l'interface web à l'adresse http://localhost:5000/ pour :
- Visualiser le tableau de bord des performances
- Consulter l'historique des transactions
- Configurer les paramètres du bot
- Surveiller les positions ouvertes

### 🚀 Modes d'Exécution
- **Mode Paper Trading** :
  ```bash
  python src/main.py --mode paper
  ```

- **Mode Backtest** :
  ```bash
  python src/main.py --mode backtest
  ```

- **Mode Trading Réel** (à utiliser avec précaution) :
  ```bash
  python src/main.py --mode live
  ```

- **Mode Analyse de Marché** :
  ```bash
  python src/main.py --mode analysis
  ```

### 🔄 Options Avancées
- **Entraînement Forcé des Modèles** :
  ```bash
  python src/main.py --force-train
  ```

- **Sélection de Symboles Spécifiques** :
  ```bash
  python src/main.py --symbols BTC-USD,ETH-USD,AAPL
  ```

- **Mode Debug** :
  ```bash
  python src/main.py --debug
  ```

---

## ⚙️ Configuration

### 📝 Paramètres Principaux
Les principales configurations se trouvent dans le fichier `.env` :

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `ENABLE_LIVE_TRADING` | Active le trading réel | `false` |
| `RISK_PER_TRADE` | % du capital par trade | `0.02` |
| `CONFIDENCE_THRESHOLD` | Seuil de confiance IA | `0.65` |
| `SYMBOLS` | Symboles à trader | `BTC-USD,ETH-USD,AAPL...` |
| `TELEGRAM_TOKEN` | Token bot Telegram | - |
| `OPENROUTER_API_KEY` | Clé API pour Claude | - |
| `CLAUDE_MODEL` | Version de Claude | `anthropic/claude-3.7-sonnet` |

### 🧮 Configuration des Modèles
Personnalisez les paramètres des modèles dans les fichiers de configuration dédiés :
- `config/models/price_prediction.json`

# EVIL2ROOT Trading Bot - Service Web Complet

Un service web complet pour le bot de trading EVIL2ROOT, comprenant une API RESTful et une interface utilisateur moderne.

## Fonctionnalités

- **API RESTful complète** pour interagir avec le bot de trading
- **Interface utilisateur moderne** développée avec React et Material UI
- **Tableau de bord de trading** avec visualisation en temps réel
- **Gestion des utilisateurs** avec authentification sécurisée
- **Système d'abonnement** avec différents niveaux de service
- **Backtesting** pour tester vos stratégies sur des données historiques
- **Analyses de performance** détaillées
- **Notifications** par email et dans l'application

## Architecture

Le projet est organisé en deux parties principales :

- **Backend**: API Python FastAPI avec PostgreSQL et Redis
- **Frontend**: Application React TypeScript avec Material UI

## Prérequis

- Python 3.8+
- Node.js 16+
- PostgreSQL
- Redis
- Connexion Internet (pour les données de marché en temps réel)

## Installation

### Backend (API)

1. Clonez le dépôt :
```bash
git clone https://github.com/yourusername/EVIL2ROOT_AI.git
cd EVIL2ROOT_AI
```

2. Créez un environnement virtuel et installez les dépendances :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configurez les variables d'environnement :
```bash
cp .env.example .env
# Modifiez .env avec vos paramètres
```

4. Lancez l'API :
```bash
python src/api/run_api.py
```

L'API sera disponible à l'adresse http://localhost:8000.

### Frontend

1. Allez dans le dossier frontend :
```bash
cd frontend
```

2. Installez les dépendances :
```bash
npm install
```

3. Configurez les variables d'environnement :
```bash
cp .env.example .env
# Modifiez .env avec vos paramètres
```

4. Lancez l'application :
```bash
npm start
```

L'interface sera disponible à l'adresse http://localhost:3000.

## Utilisation

### API

L'API est documentée avec Swagger UI, accessible à l'adresse http://localhost:8000/docs.

Points d'entrée principaux :
- `/api/auth/*` - Authentification et gestion des utilisateurs
- `/api/trading/*` - Opérations de trading
- `/api/dashboard/*` - Données du tableau de bord
- `/api/settings/*` - Configuration du bot
- `/api/subscriptions/*` - Gestion des abonnements
- `/api/backtest/*` - Backtesting

### Interface utilisateur

L'interface utilisateur comprend :
- Page d'accueil publique
- Pages d'authentification (connexion, inscription)
- Tableau de bord principal
- Page de trading
- Gestion des positions et ordres
- Analyse des signaux
- Backtesting
- Paramètres du compte et du bot
- Gestion de l'abonnement

## Déploiement

### Production

Pour un déploiement en production :

1. Construisez le frontend :
```bash
cd frontend
npm run build
```

2. Servez les fichiers statiques avec un serveur web comme Nginx.

3. Exécutez l'API avec un serveur ASGI comme uvicorn avec plusieurs workers :
```bash
uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000 --workers 4
```

4. Configurez un proxy inverse pour diriger les requêtes `/api` vers le backend.

## Licence

© 2023 EVIL2ROOT. Tous droits réservés.

## Contact

Pour toute question ou suggestion, veuillez contacter support@evil2root.com.
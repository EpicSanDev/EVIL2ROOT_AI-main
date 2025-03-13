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

- **Intelligence Artificielle Avancée** : Modèles de deep learning LSTM, GRU et Transformers pour la prédiction des prix
- **Validation Multi-niveaux** : Chaque décision de trading est validée par un système IA secondaire utilisant Claude 3.7
- **Ensemble Sophistiqué** : Combinaison de modèles d'analyse technique, fondamentale et d'IA
- **Gestion Dynamique des Risques** : Calcul intelligent des tailles de position et niveaux stop-loss/take-profit
- **Surveillance en Temps Réel** : Interface web intuitive avec tableaux de bord complets et alertes instantanées

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
- `config/models/price_prediction.json` : Configuration des modèles de prix
- `config/models/sentiment_analysis.json` : Paramètres d'analyse de sentiment
- `config/models/risk_management.json` : Règles de gestion des risques

### 🚦 Stratégies de Trading
Configurez vos stratégies dans `config/strategies/` :
- Paramètres d'entrée et de sortie
- Combinaisons d'indicateurs
- Règles de validation de signaux
- Périodes d'analyse

---

## 📊 Visualisation et Surveillance

### 📈 Dashboard Temps Réel
- Graphiques interactifs avec indicateurs techniques
- Vue consolidée du portefeuille et des performances
- Analyse de corrélation entre actifs
- Signaux de trading récents et historiques

### 🔔 Système de Notifications
- Alertes Telegram pour chaque transaction
- Rapports quotidiens de performance
- Notifications d'événements critiques
- Alertes de risque personnalisables

### 📉 Métriques de Performance
- Ratio de Sharpe et Sortino
- Drawdown maximum
- Gain moyen vs perte moyenne
- Taux de réussite par stratégie
- Performance par type d'actif

<div align="center">
  <img src="https://via.placeholder.com/800x400?text=EVIL2ROOT+Performance+Metrics" alt="Métriques de performance"/>
  <p><em>Métriques de performance et analyse de risque</em></p>
</div>

---

## 📚 Documentation

- [📖 Guide Utilisateur](docs/USER_GUIDE.md) - Guide complet d'utilisation du système
- [🏗️ Architecture du Système](docs/ARCHITECTURE.md) - Documentation technique détaillée
- [🔌 Documentation API](docs/api/) - Référence des endpoints API
- [🚀 Guide de Déploiement](docs/DEPLOYMENT.md) - Instructions de déploiement détaillées
- [☸️ Configuration Kubernetes](docs/KUBERNETES.md) - Guide de déploiement sur Kubernetes
- [🧠 Documentation des Modèles IA](docs/ENSEMBLE_AI_DOCUMENTATION.md) - Détails sur les modèles d'IA

Pour une documentation spécifique aux modèles et stratégies, consultez le répertoire [docs/](docs/).

---

## 💻 Développement

### 🛠️ Installation pour le Développement
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances de développement
pip install -r requirements-dev.txt

# Installer le package en mode développement
pip install -e .
```

### 🧪 Exécution des Tests
```bash
# Exécuter tous les tests
pytest

# Exécuter les tests unitaires uniquement
pytest tests/unit/

# Exécuter avec couverture de code
pytest --cov=src
```

### 🔄 Flux de Travail Git
1. Créez une branche pour votre fonctionnalité :
   ```bash
   git checkout -b feature/ma-fonctionnalite
   ```

2. Développez et testez votre code

3. Soumettez une pull request vers la branche main

Pour plus d'informations, consultez [CONTRIBUTING.md](docs/CONTRIBUTING.md).

---

## 📝 Licence

Ce projet est distribué sous la licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## ⚠️ Avertissement

**AVIS DE RISQUE** : Le trading automatisé comporte des risques financiers substantiels. Ce logiciel est fourni à des fins éducatives et de recherche uniquement. Les performances passées ne garantissent pas les résultats futurs. Utilisez à vos propres risques.

**REMARQUE IMPORTANTE** : Ce système n'est pas conçu pour être un conseiller financier. Toujours consulter un professionnel qualifié avant de prendre des décisions d'investissement.

---

<div align="center">
  <p>Développé avec ❤️ par l'équipe EVIL2ROOT</p>
  <p>
    <a href="https://github.com/EpicSanDev/EVIL2ROOT_AI-main/issues">Signaler un problème</a> •
    <a href="docs/CONTRIBUTING.md">Contribuer</a> •
    <a href="docs/CHANGELOG.md">Changelog</a>
  </p>
</div> 
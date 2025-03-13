# EVIL2ROOT Trading Bot 🤖📈

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)](https://www.postgresql.org/)

Un système de trading automatisé complet avec validation des décisions par IA, apprentissage par renforcement et analyse du sentiment de marché.

## 📋 Table des Matières

- [Caractéristiques](#-caractéristiques)
- [Nouvelle Structure](#-nouvelle-structure)
- [Pour Commencer](#-pour-commencer)
  - [Installation Standard](#installation-standard)
  - [Installation avec Docker](#installation-avec-docker)
- [Utilisation](#-utilisation)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Développement](#-développement)
- [Licence](#-licence)

## 🚀 Caractéristiques

- **Modèles de Trading Multiples**: Indicateurs techniques, deep learning, RL, analyse de sentiment
- **Validation IA des Transactions**: Système IA secondaire validant chaque décision
- **Gestion Avancée des Risques**: Taille des positions dynamique, stop-loss et take-profit automatiques
- **Interface Web Intuitive**: Tableau de bord en temps réel avec visualisations avancées
- **Notifications en Temps Réel**: Alertes Telegram pour les transactions importantes
- **Infrastructure Robuste**: Docker, microservices, haute disponibilité
- **Analyse Quotidienne et Planifiée**: Rapports d'analyse automatisés du marché

## 🏗 Nouvelle Structure

Le projet a été restructuré pour améliorer l'organisation, la lisibilité et la maintenabilité:

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

Pour plus de détails sur la nouvelle structure, consultez [le document de restructuration](./NOUVELLE_STRUCTURE.md).

## 🚦 Pour Commencer

### Installation Standard

1. Clonez le dépôt:
   ```bash
   git clone https://github.com/EpicSanDev/EVIL2ROOT_AI-main.git
   cd EVIL2ROOT_AI-main
   ```

2. Exécutez le script d'installation:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. Configurez le fichier `.env` avec vos paramètres:
   ```bash
   nano .env
   ```

4. Lancez l'application:
   ```bash
   source venv/bin/activate
   python src/main.py
   ```

### Installation avec Docker

1. Clonez le dépôt et configurez l'environnement:
   ```bash
   git clone https://github.com/EpicSanDev/EVIL2ROOT_AI-main.git
   cd EVIL2ROOT_AI-main
   cp config/environments/.env.example .env
   ```

2. Modifiez le fichier `.env` avec vos configurations

3. Démarrez avec Docker Compose:
   ```bash
   cd docker
   docker-compose up -d
   ```

## 🎮 Utilisation

- **Interface Web**: Accédez à l'interface web à http://localhost:5000/
- **Démarrer le bot**: `python src/main.py --mode paper`
- **Backtest**: `python src/main.py --mode backtest`
- **Trading réel**: `python src/main.py --mode live` (utilisez avec précaution)
- **Analyse de marché**: `python src/main.py --mode analysis`

## ⚙️ Configuration

Les principales configurations se trouvent dans le fichier `.env`:

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `ENABLE_LIVE_TRADING` | Active le trading réel | `false` |
| `RISK_PER_TRADE` | % du capital par trade | `0.02` |
| `CONFIDENCE_THRESHOLD` | Seuil de confiance IA | `0.65` |
| `SYMBOLS` | Symboles à trader | `BTC-USD,ETH-USD,AAPL...` |

## 📚 Documentation

- [Guide Utilisateur](./USER_GUIDE.md)
- [Architecture du Système](./ARCHITECTURE.md)
- [Documentation API](./api/)
- [Guide Déploiement](./DEPLOYMENT.md)
- [Kubernetes](./KUBERNETES.md)

## 💻 Développement

Pour contribuer au projet:

1. Créez une branche pour votre fonctionnalité:
   ```bash
   git checkout -b feature/ma-fonctionnalite
   ```

2. Installez les dépendances de développement:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Exécutez les tests:
   ```bash
   pytest tests/
   ```

Consultez [CONTRIBUTING.md](./CONTRIBUTING.md) pour plus d'informations.

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](../LICENSE) pour plus de détails.

## ⚠️ Avertissement

Ce logiciel est fourni à des fins éducatives et de recherche uniquement. Le trading automatisé comporte des risques financiers importants. Utilisez à vos propres risques. 
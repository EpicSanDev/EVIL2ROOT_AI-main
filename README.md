# EVIL2ROOT Trading Bot

Un système de trading automatisé complet avec validation des décisions par IA, apprentissage par renforcement et analyse du sentiment de marché.

## Caractéristiques

- **Modèles de Trading Multiples**: Indicateurs techniques, prédiction de prix, apprentissage par renforcement et analyse de sentiment
- **Validation IA des Transactions**: Système IA secondaire validant les décisions de trading avec Claude 3 Opus via l'API OpenRouter
- **Suivi des Performances**: Métriques détaillées sur les performances de trading
- **Support Docker**: Configuration entièrement conteneurisée pour un déploiement fiable
- **Persistance en Base de Données**: Données de trading stockées dans une base de données PostgreSQL
- **Interface Web**: Surveillez l'activité de trading depuis une interface web intuitive
- **Notifications Telegram**: Alertes de trading en temps réel

## Architecture du Système

Le système comprend plusieurs services conteneurisés:

- **Trading Bot**: Logique de trading principale et exécution de modèles
- **AI Validation**: Système IA secondaire qui valide les décisions de trading
- **Web UI**: Tableau de bord basé sur Flask pour la surveillance
- **PostgreSQL**: Base de données pour stocker les données de trading et les métriques de performance
- **Redis**: Communication entre les services de trading

## Pour Commencer

### Prérequis

- Docker et Docker Compose
- Python 3.8+ (pour le développement en dehors de Docker)
- Compte OpenRouter pour l'API Claude (pour la validation IA)

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

- `ENABLE_LIVE_TRADING`: Définir à `true` pour activer le trading en direct (défaut: `false`)
- `RISK_PER_TRADE`: Pourcentage de risque par transaction (défaut: `0.02` = 2%)
- `CONFIDENCE_THRESHOLD`: Confiance minimale de l'IA pour valider les transactions
- `SYMBOLS`: Liste de symboles à trader séparés par des virgules
- `TELEGRAM_TOKEN`: Token du bot Telegram pour les notifications
- `OPENROUTER_API_KEY`: Clé API OpenRouter pour accéder à Claude pour la validation IA
- `CLAUDE_MODEL`: ID du modèle Claude à utiliser (défaut: `anthropic/claude-3-opus-20240229`)

## Composants du Système

### Trading Bot

Le composant de trading principal qui:
- Récupère et traite les données de marché
- Exécute plusieurs modèles de trading
- Envoie des demandes de transactions au service de validation IA
- Exécute les transactions validées
- Gère les positions ouvertes

### Service de Validation IA

Un service IA séparé qui:
- Valide les décisions de trading du bot principal
- Vérifie si la transaction s'aligne avec les tendances du marché sur plusieurs périodes
- S'assure que les niveaux de risque sont acceptables
- Fournit des scores de confiance pour les décisions de trading
- Utilise Claude via l'API OpenRouter pour une analyse avancée

### Schéma de la Base de Données

La base de données PostgreSQL comprend:
- `trade_history`: Historique de toutes les transactions
- `trading_signals`: Signaux de trading générés par les modèles
- `market_data`: Snapshots de données historiques du marché
- `performance_metrics`: Statistiques quotidiennes de performance de trading
- `bot_settings`: Paramètres de configuration pour le bot de trading

## Utilisation

### Démarrer le Bot

```bash
# Démarrer tous les services en arrière-plan
make up

# Démarrer avec les logs visibles dans le terminal
make up-log
```

### Commandes Docker (via Makefile)

Le projet inclut un Makefile pour simplifier les opérations Docker:

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

### Surveillance

- Interface web: http://localhost:5001/
- Logs de trading: Consultez `logs/trading_bot.log`
- Logs de validation IA: Consultez `logs/ai_validator.log`
- Logs des conteneurs: `make logs` ou `make logs-SERVICE`

### Développement

Pour le développement local en dehors de Docker:

```bash
pip install -r requirements.txt
python run.py
```

## Backtesting

Pour exécuter le backtesting:

```bash
# Utilisation du Makefile
make shell-trading-bot
python -c "from app.trading import TradingBot; bot = TradingBot(); bot.run_backtest('data/market_data_cleaned.csv')"

# Ou directement avec docker compose
docker compose run trading-bot python -c "from app.trading import TradingBot; bot = TradingBot(); bot.run_backtest('data/market_data_cleaned.csv')"
```

## Comment Contribuer

Les contributions sont les bienvenues! Veuillez suivre ces étapes:

1. Forker le dépôt
2. Créer une branche de fonctionnalité (`git checkout -b feature/fonctionnalite-incroyable`)
3. Validez vos modifications (`git commit -m 'Ajouter une fonctionnalité incroyable'`)
4. Poussez vers la branche (`git push origin feature/fonctionnalite-incroyable`)
5. Ouvrez une Pull Request

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Sécurité

Si vous découvrez une vulnérabilité de sécurité, veuillez envoyer un e-mail à evil2root@protonmail.com au lieu d'utiliser l'outil de suivi des problèmes. Nous prendrons les mesures nécessaires pour résoudre le problème rapidement.

## Avertissement

Ce logiciel est fourni à des fins éducatives uniquement. Le trading comporte des risques inhérents. Les auteurs ne sont pas responsables des pertes financières pouvant résulter de l'utilisation de ce logiciel. Utilisez-le à vos propres risques et consultez toujours un conseiller financier professionnel.

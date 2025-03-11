# Bot d'Analyse Quotidienne - Guide d'Utilisation Docker

Ce document explique comment configurer et exécuter le bot d'analyse quotidienne avec Docker.

## Présentation

Le Bot d'Analyse Quotidienne est un outil qui génère et envoie des analyses financières complètes via Telegram à des moments prédéfinis de la journée (9h00, 12h30, 16h30, 20h00). Ces analyses intègrent :

- Analyse technique (indicateurs, tendances)
- Analyse fondamentale (données financières)
- Analyse de sentiment (basée sur les nouvelles)
- Prédiction de prix (modèles d'IA)
- Évaluation des risques
- Actualités pertinentes

Toutes ces analyses sont générées avec l'aide de plusieurs modèles d'IA, dont Claude 3.7 via OpenRouter.

## Prérequis

- [Docker](https://www.docker.com/get-started) et [Docker Compose](https://docs.docker.com/compose/install/)
- Un [token de bot Telegram](https://core.telegram.org/bots#how-do-i-create-a-bot)
- Un [ID de chat Telegram](https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id)
- Une [clé API OpenRouter](https://openrouter.ai/keys) (pour l'accès à Claude 3.7)

## Configuration

1. Créez un fichier `.env` à la racine du projet avec les variables suivantes :

```
# Configuration de base
TELEGRAM_TOKEN=votre_token_telegram
TELEGRAM_CHAT_ID=votre_chat_id
OPENROUTER_API_KEY=votre_clé_openrouter

# Symboles à analyser (séparés par des virgules)
SYMBOLS=AAPL,GOOGL,MSFT,AMZN,TSLA,BTC-USD,ETH-USD

# Configuration du modèle Claude
CLAUDE_MODEL=anthropic/claude-3.7-sonnet

# Base de données (pour le stockage des données)
DB_NAME=trading_db
DB_USER=trader
DB_PASSWORD=secure_password
DB_HOST=db
DB_PORT=5432

# Redis (pour la mise en cache et la communication entre services)
REDIS_HOST=redis
REDIS_PORT=6379
```

## Démarrage Rapide

Un script utilitaire est fourni pour faciliter le démarrage du bot. Utilisez simplement :

```bash
./start_analysis_docker.sh
```

Le script vérifiera que tout est correctement configuré, puis démarrera le bot d'analyse en tant que service Docker.

### Options

- `--build` : Reconstruit l'image Docker avant de démarrer le service
  ```bash
  ./start_analysis_docker.sh --build
  ```

## Démarrage Manuel

Si vous préférez utiliser Docker Compose directement :

1. Construisez l'image (uniquement nécessaire la première fois ou après des modifications) :
   ```bash
   docker-compose build analysis-bot
   ```

2. Démarrez le service :
   ```bash
   docker-compose up -d analysis-bot
   ```

3. Consultez les logs :
   ```bash
   docker-compose logs -f analysis-bot
   ```

## Architecture Docker

Le système utilise plusieurs services Docker :

- `analysis-bot` : Le bot d'analyse quotidienne
- `web` : Interface web du système de trading (facultatif pour le bot d'analyse)
- `db` : Base de données PostgreSQL pour stocker les données
- `redis` : Cache Redis pour les performances
- `prometheus` & `grafana` : Monitoring (facultatif)

## Personnalisation

### Horaires d'analyse

Les horaires d'analyse sont définis dans `app/daily_analysis_bot.py`. Pour les modifier, éditez ce fichier et reconstruisez l'image Docker.

```python
self.analysis_schedule = [
    "09:00",  # Analyse pré-marché
    "12:30",  # Analyse de mi-journée
    "16:30",  # Analyse de clôture
    "20:00"   # Analyse récapitulative
]
```

### Symboles analysés

Les symboles à analyser peuvent être configurés dans le fichier `.env` via la variable `SYMBOLS`.

## Dépannage

### Problèmes courants

1. **Le bot ne se connecte pas à Telegram**
   - Vérifiez que votre token Telegram est correct
   - Assurez-vous que le bot a été démarré dans la conversation

2. **Erreurs d'analyse**
   - Vérifiez les logs Docker pour identifier l'erreur spécifique
   - Assurez-vous que l'API OpenRouter est correctement configurée

3. **Problèmes de base de données**
   - Vérifiez que le service PostgreSQL est bien démarré
   - Vérifiez les identifiants dans le fichier `.env`

### Consultation des logs

```bash
docker-compose logs -f analysis-bot
```

## Ressources additionnelles

- [Documentation Telegram Bot API](https://core.telegram.org/bots/api)
- [Documentation OpenRouter](https://openrouter.ai/docs)
- [Documentation Docker Compose](https://docs.docker.com/compose/) 
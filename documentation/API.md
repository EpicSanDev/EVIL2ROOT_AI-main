# API et Interfaces du Système EVIL2ROOT

Ce document détaille les différentes API et interfaces exposées par le système EVIL2ROOT Trading Bot, permettant l'interaction avec le système et l'intégration avec d'autres applications.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [API REST](#api-rest)
3. [Webhooks](#webhooks)
4. [Interface Web](#interface-web)
5. [Interface Telegram](#interface-telegram)
6. [Bus de messages Redis](#bus-de-messages-redis)
7. [Authentification et sécurité](#authentification-et-sécurité)
8. [Limites de taux et performance](#limites-de-taux-et-performance)
9. [Intégration avec des systèmes externes](#intégration-avec-des-systèmes-externes)

## Vue d'ensemble

Le système EVIL2ROOT Trading Bot expose plusieurs interfaces permettant l'interaction et l'intégration avec d'autres systèmes :

- **API REST** : Interface principale pour les intégrations programmatiques
- **Interface Web** : Interface utilisateur graphique pour la surveillance et la configuration
- **Webhooks** : Points de terminaison pour la notification d'événements externes
- **Interface Telegram** : Bot Telegram pour les notifications et les commandes simples
- **Bus de messages Redis** : Communication interne entre les composants

## API REST

L'API REST est le principal point d'intégration programmatique avec le système EVIL2ROOT. Elle est basée sur les principes REST, utilise JSON pour la sérialisation des données et suit les conventions standard HTTP.

### Base URL

```
https://<host>:<port>/api/v1
```

### Authentification

L'authentification à l'API utilise des tokens JWT (JSON Web Tokens). Les tokens sont obtenus via le endpoint `/auth/token` et doivent être inclus dans l'en-tête `Authorization` de chaque requête :

```
Authorization: Bearer <token>
```

### Endpoints

#### Authentification

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| POST | `/auth/token` | Obtenir un token d'authentification | `username`, `password` |
| POST | `/auth/refresh` | Rafraîchir un token expiré | `refresh_token` |
| POST | `/auth/revoke` | Révoquer un token | - |

#### Système

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/system/status` | Obtenir l'état du système | - |
| GET | `/system/health` | Vérifier la santé du système | - |
| GET | `/system/version` | Obtenir la version du système | - |
| GET | `/system/logs` | Obtenir les journaux du système | `level`, `component`, `start_date`, `end_date`, `limit` |

#### Configuration

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/config` | Obtenir la configuration actuelle | - |
| PUT | `/config` | Mettre à jour la configuration | Configuration en JSON |
| GET | `/config/{parameter}` | Obtenir un paramètre spécifique | - |
| PUT | `/config/{parameter}` | Mettre à jour un paramètre | `value` |

#### Trading

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/trading/symbols` | Obtenir la liste des symboles disponibles | - |
| GET | `/trading/status` | Obtenir l'état du système de trading | - |
| POST | `/trading/enable` | Activer le trading | `symbols` (optionnel) |
| POST | `/trading/disable` | Désactiver le trading | `symbols` (optionnel) |
| POST | `/trading/trade` | Créer une transaction manuelle | `symbol`, `action`, `quantity`, `price`, `stop_loss`, `take_profit` |

#### Positions

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/positions` | Obtenir toutes les positions ouvertes | `symbol` (optionnel) |
| GET | `/positions/{position_id}` | Obtenir une position spécifique | - |
| PUT | `/positions/{position_id}` | Modifier une position | `stop_loss`, `take_profit` |
| DELETE | `/positions/{position_id}` | Fermer une position | `price` (optionnel) |

#### Transactions

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/trades` | Obtenir l'historique des transactions | `symbol`, `start_date`, `end_date`, `limit`, `offset` |
| GET | `/trades/{trade_id}` | Obtenir une transaction spécifique | - |
| GET | `/trades/stats` | Obtenir des statistiques sur les transactions | `symbol`, `period` |

#### Signaux

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/signals` | Obtenir les signaux récents | `symbol`, `signal_type`, `start_date`, `end_date`, `limit` |
| GET | `/signals/latest` | Obtenir les derniers signaux par symbole | `symbol` (optionnel) |

#### Analyse

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/analysis/daily` | Obtenir les analyses quotidiennes | `symbol`, `date` |
| POST | `/analysis/run` | Exécuter une analyse à la demande | `symbol`, `timeframe`, `analysis_type` |
| GET | `/analysis/schedule` | Obtenir le planning des analyses | - |

#### Modèles

| Méthode | Endpoint | Description | Paramètres |
|---------|----------|-------------|------------|
| GET | `/models` | Obtenir la liste des modèles disponibles | - |
| GET | `/models/{model_id}` | Obtenir les détails d'un modèle | - |
| GET | `/models/{model_id}/performance` | Obtenir les performances d'un modèle | `symbol`, `period` |
| POST | `/models/{model_id}/train` | Déclencher l'entraînement d'un modèle | `symbol`, `parameters` |

### Exemples de requêtes

#### Obtenir un token d'authentification

```bash
curl -X POST https://example.com/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'
```

Réponse :
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Obtenir les positions ouvertes

```bash
curl -X GET https://example.com/api/v1/positions \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

Réponse :
```json
{
  "positions": [
    {
      "id": 123,
      "symbol": "AAPL",
      "entry_price": 150.25,
      "entry_time": "2023-04-12T14:30:00Z",
      "quantity": 10,
      "stop_loss": 145.50,
      "take_profit": 160.00,
      "current_price": 152.75,
      "unrealized_pnl": 25.00,
      "unrealized_pnl_percent": 1.66
    },
    {
      "id": 124,
      "symbol": "BTC-USD",
      "entry_price": 28500.50,
      "entry_time": "2023-04-13T09:15:00Z",
      "quantity": 0.5,
      "stop_loss": 27000.00,
      "take_profit": 31000.00,
      "current_price": 29250.25,
      "unrealized_pnl": 374.88,
      "unrealized_pnl_percent": 2.63
    }
  ],
  "count": 2
}
```

#### Créer une transaction manuelle

```bash
curl -X POST https://example.com/api/v1/trading/trade \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "MSFT",
    "action": "BUY",
    "quantity": 5,
    "price": 280.50,
    "stop_loss": 270.00,
    "take_profit": 300.00
  }'
```

Réponse :
```json
{
  "trade_id": 789,
  "symbol": "MSFT",
  "action": "BUY",
  "quantity": 5,
  "entry_price": 280.50,
  "stop_loss": 270.00,
  "take_profit": 300.00,
  "entry_time": "2023-04-15T10:45:32Z",
  "status": "OPEN"
}
```

### Gestion des erreurs

L'API utilise les codes d'état HTTP standard pour indiquer le succès ou l'échec d'une requête :

- `200 OK` : Requête réussie
- `201 Created` : Ressource créée avec succès
- `400 Bad Request` : Paramètres invalides
- `401 Unauthorized` : Authentification requise
- `403 Forbidden` : Accès refusé
- `404 Not Found` : Ressource non trouvée
- `429 Too Many Requests` : Limite de taux dépassée
- `500 Internal Server Error` : Erreur serveur

Les réponses d'erreur suivent un format standard :

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid value for parameter 'quantity'",
    "details": "Quantity must be greater than 0"
  }
}
```

## Webhooks

Les webhooks permettent à des systèmes externes de recevoir des notifications en temps réel des événements du système EVIL2ROOT.

### Configuration des webhooks

Les webhooks sont configurés via l'API REST :

```bash
curl -X POST https://example.com/api/v1/webhooks \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-system.com/callback",
    "events": ["trade.new", "trade.closed", "signal.new"],
    "secret": "your_webhook_secret"
  }'
```

### Événements disponibles

| Événement | Description |
|-----------|-------------|
| `trade.new` | Nouvelle transaction ouverte |
| `trade.closed` | Transaction fermée |
| `trade.updated` | Transaction mise à jour |
| `signal.new` | Nouveau signal généré |
| `position.updated` | Position mise à jour |
| `analysis.completed` | Analyse terminée |
| `model.trained` | Modèle entraîné |
| `system.alert` | Alerte système |

### Format des notifications

Les notifications webhook sont envoyées en format JSON avec une signature pour la vérification :

```
X-EVIL2ROOT-Signature: sha256=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce373eec5
```

Exemple de payload :
```json
{
  "event": "trade.new",
  "timestamp": "2023-04-15T10:45:32Z",
  "data": {
    "trade_id": 789,
    "symbol": "MSFT",
    "action": "BUY",
    "quantity": 5,
    "entry_price": 280.50,
    "stop_loss": 270.00,
    "take_profit": 300.00
  }
}
```

### Vérification de la signature

Pour vérifier l'authenticité des webhooks, calculez une signature HMAC-SHA256 du corps de la requête en utilisant votre secret webhook :

```python
import hmac
import hashlib

def verify_webhook_signature(payload_body, header_signature, secret):
    calculated_signature = hmac.new(
        secret.encode('utf-8'),
        payload_body.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    expected_signature = header_signature.split('=')[1]
    
    return hmac.compare_digest(calculated_signature, expected_signature)
```

## Interface Web

L'interface web fournit une interface utilisateur graphique pour surveiller et configurer le système EVIL2ROOT.

### URL

```
https://<host>:<port>/
```

### Pages principales

| Page | URL | Description |
|------|-----|-------------|
| Tableau de bord | `/` | Vue d'ensemble du système |
| Transactions | `/trades` | Historique des transactions |
| Positions | `/positions` | Positions ouvertes |
| Signaux | `/signals` | Signaux récents |
| Analyses | `/analysis` | Résultats d'analyses |
| Symboles | `/symbols` | Détails par symbole |
| Modèles | `/models` | Modèles et performances |
| Configuration | `/settings` | Configuration du système |
| Journal | `/logs` | Journaux du système |

### Fonctionnalités clés

- **Tableau de bord** :
  - Vue d'ensemble des performances
  - Graphiques de performance
  - Positions ouvertes
  - Alertes et notifications
  - Statut du système

- **Transactions** :
  - Historique complet des transactions
  - Filtrage et recherche
  - Visualisation des transactions sur graphiques
  - Statistiques et métriques

- **Positions** :
  - Positions actuellement ouvertes
  - Statut en temps réel
  - Modification des SL/TP
  - Fermeture manuelle

- **Signaux** :
  - Visualisation des signaux récents
  - Analyse des sources de signaux
  - Superposition des signaux sur les graphiques
  - Statistiques de performance des signaux

- **Configuration** :
  - Paramètres du système
  - Gestion des symboles tradés
  - Configuration des modèles
  - Planification des analyses
  - Gestion des utilisateurs et des accès

### Authentification

L'accès à l'interface web est protégé par une authentification basée sur des sessions :

- Formulaire de connexion standard (username/password)
- Support 2FA (TOTP) optionnel
- Gestion des sessions avec expiration automatique
- Journalisation des connexions et des activités

## Interface Telegram

Le bot Telegram fournit une interface mobile pour recevoir des notifications et envoyer des commandes simples au système.

### Configuration

Pour configurer le bot Telegram :
1. Créer un bot via [@BotFather](https://t.me/botfather) sur Telegram
2. Configurer le token du bot dans le fichier `.env` : `TELEGRAM_TOKEN=your_token`
3. Démarrer une conversation avec le bot
4. S'authentifier avec le bot en utilisant la commande `/auth your_secret_key`

### Commandes disponibles

| Commande | Description | Exemple |
|----------|-------------|---------|
| `/help` | Afficher l'aide | `/help` |
| `/auth` | S'authentifier avec le bot | `/auth your_secret_key` |
| `/status` | Obtenir l'état du système | `/status` |
| `/positions` | Lister les positions ouvertes | `/positions` |
| `/position` | Détails d'une position | `/position AAPL` |
| `/trade` | Voir l'historique des transactions | `/trade` ou `/trade AAPL 5` |
| `/report` | Générer un rapport rapide | `/report` ou `/report AAPL` |
| `/performance` | Obtenir les performances | `/performance` ou `/performance daily` |
| `/alert` | Configurer une alerte | `/alert AAPL > 150` |
| `/configure` | Modifier un paramètre | `/configure risk_per_trade 0.02` |
| `/stop` | Arrêter le trading | `/stop` ou `/stop AAPL` |
| `/start` | Démarrer le trading | `/start` ou `/start AAPL` |

### Notifications

Le bot Telegram envoie automatiquement des notifications pour les événements importants :

- Nouvelles transactions
- Positions fermées
- Alertes de prix
- Signaux importants
- Analyses quotidiennes
- Alertes système

### Sécurité

- Authentification par code secret 
- Autorisation par ID utilisateur Telegram
- Liste blanche d'utilisateurs autorisés
- Limitation des commandes par niveau d'accès
- Journalisation de toutes les interactions

## Bus de messages Redis

Le bus de messages Redis est utilisé pour la communication interne entre les composants du système. Bien qu'il ne soit pas directement exposé aux clients, il peut être utilisé pour des intégrations avancées.

### Canaux principaux

| Canal | Description | Format du message |
|-------|-------------|-------------------|
| `trade_requests` | Demandes de validation de transactions | JSON |
| `trade_responses` | Réponses de validation | JSON |
| `trade_executed` | Transactions exécutées | JSON |
| `market_data` | Données de marché en temps réel | JSON |
| `signals` | Signaux générés | JSON |
| `system_alerts` | Alertes système | JSON |
| `analysis_requests` | Demandes d'analyse | JSON |
| `analysis_results` | Résultats d'analyse | JSON |

### Format des messages

Les messages suivent un format standard :

```json
{
  "id": "msg_12345",
  "timestamp": "2023-04-15T10:45:32.123Z",
  "type": "trade_request",
  "source": "trading_bot",
  "destination": "ai_validator",
  "data": {
    // Données spécifiques au type de message
  }
}
```

### Exemple d'utilisation (Python)

```python
import redis
import json
import time
import uuid
from datetime import datetime

# Connexion à Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Fonction d'envoi de message
def send_message(channel, msg_type, data, destination=None):
    message = {
        "id": f"msg_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.utcnow().isoformat(),
        "type": msg_type,
        "source": "external_system",
        "destination": destination,
        "data": data
    }
    redis_client.publish(channel, json.dumps(message))
    return message["id"]

# Fonction d'écoute
def listen_for_messages(channel):
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            print(f"Received: {data}")
            # Traitement du message
            
# Exemple d'envoi d'une demande d'analyse
msg_id = send_message(
    channel="analysis_requests",
    msg_type="analysis_request",
    destination="daily_analysis_bot",
    data={
        "symbol": "AAPL",
        "timeframe": "1D",
        "analysis_type": "technical",
        "parameters": {
            "depth": "full"
        }
    }
)
```

## Authentification et sécurité

### Méthodes d'authentification

- **API REST** : Authentification JWT (JSON Web Tokens)
- **Interface Web** : Authentification par session + support 2FA optionnel
- **Webhooks** : Signatures HMAC pour la vérification
- **Telegram** : Clé secrète + liste blanche d'utilisateurs

### Gestion des autorisations

Les autorisations sont gérées via un système de rôles :

| Rôle | Description | Accès |
|------|-------------|-------|
| `admin` | Administrateur | Accès complet à toutes les fonctionnalités |
| `trader` | Trader | Gestion des positions, transactions manuelles, configuration limitée |
| `analyst` | Analyste | Accès en lecture seule, exécution d'analyses personnalisées |
| `viewer` | Lecteur | Accès en lecture seule uniquement |
| `system` | Système | Accès programmatique pour les intégrations |

### Sécurité des communications

- Toutes les communications externes utilisent TLS/SSL
- Validation stricte des entrées
- Protection contre les attaques CSRF pour l'interface web
- Rate limiting pour prévenir les abus
- Journalisation de toutes les actions sensibles

## Limites de taux et performance

### Limites de taux API

Des limites de taux sont appliquées pour éviter les abus et assurer la performance du système :

| Endpoint | Limite standard | Limite élevée |
|----------|----------------|---------------|
| Toutes les API | 100 req/min | 500 req/min |
| `/trading/*` | 60 req/min | 300 req/min |
| `/positions/*` | 120 req/min | 600 req/min |
| Demandes d'analyse | 10 req/min | 30 req/min |

Les limites peuvent être ajustées selon les besoins.

### En-têtes de limite de taux

Les réponses de l'API incluent des en-têtes indiquant les limites de taux et l'usage actuel :

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1618495532
```

### Performance et latence

- Temps de réponse moyen de l'API REST : < 100ms
- Débit maximum supporté : 1000 requêtes/seconde
- Latence de propagation des événements Redis : < 10ms
- Délai de notification Telegram : < 2 secondes

## Intégration avec des systèmes externes

### Exemples d'intégration

#### Intégration avec des tableurs (Google Sheets, Excel)

Utilisez l'API REST pour récupérer les données et les intégrer dans des tableurs :

```javascript
// Exemple Google Apps Script pour Google Sheets
function fetchTradingData() {
  const token = "your_api_token";
  const url = "https://example.com/api/v1/trades?limit=100";
  
  const options = {
    method: 'get',
    headers: {
      'Authorization': 'Bearer ' + token
    }
  };
  
  const response = UrlFetchApp.fetch(url, options);
  const data = JSON.parse(response.getContentText());
  
  // Traitement des données
  const sheet = SpreadsheetApp.getActiveSheet();
  // Remplir la feuille avec les données...
}
```

#### Intégration avec des plateformes de trading externes

Utilisez les webhooks pour synchroniser les données avec d'autres plateformes :

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = "your_webhook_secret"

@app.route('/callback', methods=['POST'])
def callback():
    # Vérifier la signature
    signature = request.headers.get('X-EVIL2ROOT-Signature', '')
    payload = request.get_data()
    
    calculated_signature = 'sha256=' + hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, calculated_signature):
        return jsonify({"error": "Invalid signature"}), 403
    
    # Traiter l'événement
    event = request.json
    if event['event'] == 'trade.new':
        # Synchroniser avec la plateforme externe
        # ...
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(port=5000)
```

#### Intégration avec des applications mobiles personnalisées

Créez des applications mobiles qui interagissent avec l'API :

```swift
// Exemple Swift pour iOS
func fetchPositions() {
    guard let url = URL(string: "https://example.com/api/v1/positions") else { return }
    
    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.addValue("Bearer \(apiToken)", forHTTPHeaderField: "Authorization")
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        guard let data = data else { return }
        
        do {
            let positions = try JSONDecoder().decode(PositionsResponse.self, from: data)
            DispatchQueue.main.async {
                // Mettre à jour l'UI avec les positions
            }
        } catch {
            print("Error decoding positions: \(error)")
        }
    }.resume()
}
``` 
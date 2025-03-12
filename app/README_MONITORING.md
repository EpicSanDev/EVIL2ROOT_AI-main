# Monitoring Amélioré pour EVIL2ROOT Trading Bot

Ce module fournit un système de monitoring complet pour le trading bot EVIL2ROOT, avec des métriques détaillées sur les performances des modèles d'IA, les opérations de trading, l'utilisation des ressources système et les performances des API.

## Fonctionnalités principales

- **Métriques de trading** : signaux générés, trades exécutés, positions ouvertes, performance du portefeuille
- **Métriques de modèles** : précision, rappel, RMSE, calibration, drift detection
- **Métriques système** : CPU, mémoire, disque, réseau, uptime
- **Métriques API** : requêtes, latence, quotas, erreurs
- **Métriques base de données** : durée des requêtes, connexions, opérations
- **Exportation de métriques** : format JSON pour analyses externes
- **Intégration Prometheus** : pour la visualisation avec Grafana

## Installation

Le module requiert les dépendances suivantes :

```bash
pip install prometheus-client psutil numpy pandas
```

## Utilisation rapide

### Initialisation du service

```python
from monitoring_enhanced import init_enhanced_monitoring, get_monitoring_service

# Initialisation du service
monitoring = init_enhanced_monitoring(
    port=8000,  # Port pour le serveur Prometheus
    export_directory="logs/metrics",  # Répertoire pour les exports JSON
    model_metrics_interval=300  # Intervalle pour les métriques des modèles (en secondes)
)
```

### Enregistrement des métriques de trading

```python
# Enregistrer un signal de trading
monitoring.record_trading_signal(
    symbol="BTC/USD",
    direction="buy",
    confidence_level=0.85,
    model_type="price_prediction"
)

# Enregistrer un trade exécuté
monitoring.record_executed_trade(
    symbol="BTC/USD",
    direction="buy",
    success=True
)

# Mettre à jour les métriques du portefeuille
open_positions = {
    "BTC/USD": {"buy": 2, "sell": 0},
    "ETH/USD": {"buy": 1, "sell": 1}
}
monitoring.update_portfolio_metrics(
    portfolio_value=12500.0,
    balance=5000.0,
    open_positions=open_positions
)
```

### Métriques de performance des modèles

```python
# Mettre à jour les métriques de performance d'un modèle
monitoring.update_model_performance(
    model_name="price_prediction",
    timeframe="1h",
    symbol="BTC/USD",
    metrics={
        'rmse': 0.032,
        'mae': 0.025
    }
)

# Mettre à jour les métriques de calibration
monitoring.update_calibration_metrics(
    model_name="price_prediction",
    timeframe="1h",
    symbol="BTC/USD",
    ece=0.04,
    reliability_bins={
        '0.1': 0.12,
        '0.3': 0.28,
        '0.5': 0.52,
        '0.7': 0.73,
        '0.9': 0.87
    }
)

# Mettre à jour les métriques d'apprentissage en ligne
monitoring.update_online_learning_metrics(
    model_name="price_prediction",
    timeframe="1h",
    symbol="BTC/USD",
    drift_score=0.15,
    memory_size=2000,
    loss=0.08
)
```

### Utilisation des contextes pour les API et la base de données

Les contextes permettent de mesurer automatiquement la latence et d'enregistrer les erreurs.

```python
# Contexte pour une requête API
with monitoring.api_request_context('/api/market/data', 'GET') as status_code:
    # Effectuer la requête API
    response = requests.get('https://api.example.com/market/data')
    status_code.value = response.status_code  # Mettre à jour le code de statut

# Contexte pour une requête de base de données
with monitoring.db_query_context('select'):
    # Exécuter la requête
    result = db.execute('SELECT * FROM trades')
```

## Intégration avec Flask

Pour intégrer le monitoring avec une application Flask :

```python
from flask import Flask
from monitoring_enhanced import init_enhanced_monitoring

app = Flask(__name__)

# Initialiser le monitoring avec l'intégration Flask
monitoring = init_enhanced_monitoring(
    port=8000,
    export_directory="logs/metrics",
    with_flask=True,
    flask_app=app
)

# Vos routes Flask normales
@app.route('/')
def index():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)
```

## Visualisation avec Grafana

1. Assurez-vous que Prometheus est configuré pour scraper le endpoint `http://localhost:8000/`
2. Dans Grafana, ajoutez Prometheus comme source de données
3. Créez des dashboards pour visualiser les différentes métriques

## Exemple d'utilisation

Un exemple complet est disponible dans le fichier `monitoring_example.py`. Pour l'exécuter :

```bash
python monitoring_example.py
```

Cela démarrera une démonstration qui simule l'activité de trading, les métriques des modèles, les requêtes API et les opérations de base de données. Les métriques seront disponibles sur `http://localhost:8000/`.

## Structure des métriques

### Métriques de trading
- `trading_signals_total` : Compteur de signaux de trading générés
- `executed_trades_total` : Compteur de trades exécutés
- `open_positions` : Nombre de positions ouvertes par symbole et direction
- `portfolio_value` : Valeur actuelle du portefeuille
- `balance` : Solde disponible
- `win_rate` : Taux de trades gagnants
- `profit_factor` : Facteur de profit (gains/pertes)
- `average_win` : Gain moyen par trade gagnant
- `average_loss` : Perte moyenne par trade perdant
- `max_drawdown` : Drawdown maximum
- `sharpe_ratio` : Ratio de Sharpe
- `sortino_ratio` : Ratio de Sortino
- `calmar_ratio` : Ratio de Calmar

### Métriques de modèles
- `model_accuracy` : Précision du modèle
- `model_precision` : Précision des prédictions positives
- `model_recall` : Rappel des prédictions positives
- `model_f1_score` : Score F1 du modèle
- `model_rmse` : Erreur quadratique moyenne
- `model_mae` : Erreur absolue moyenne
- `calibration_error` : Erreur de calibration (ECE)
- `calibration_reliability` : Fiabilité de la calibration par bin
- `drift_detection` : Score de détection de drift conceptuel
- `memory_buffer_size` : Taille du buffer de mémoire
- `online_learning_loss` : Perte lors de l'apprentissage en ligne

### Métriques système
- `cpu_usage_percent` : Utilisation CPU en pourcentage
- `memory_usage_bytes` : Utilisation de la mémoire en bytes
- `memory_usage_percent` : Utilisation de la mémoire en pourcentage
- `disk_usage_bytes` : Utilisation du disque en bytes
- `disk_usage_percent` : Utilisation du disque en pourcentage
- `network_sent_bytes` : Données réseau envoyées
- `network_received_bytes` : Données réseau reçues
- `process_count` : Nombre de processus en cours d'exécution
- `system_load` : Charge système (1, 5, 15 minutes)
- `uptime_seconds` : Temps d'exécution en secondes

### Métriques API et base de données
- `api_requests_total` : Nombre total de requêtes API
- `api_latency_seconds` : Latence des requêtes API
- `api_quota_remaining` : Quota d'API restant
- `api_errors_total` : Nombre total d'erreurs API
- `db_query_duration_seconds` : Durée des requêtes de base de données
- `db_connection_pool_size` : Taille du pool de connexions
- `db_connections_active` : Nombre de connexions actives
- `db_operations_total` : Nombre total d'opérations de base de données

## Contribution

Pour contribuer à ce module, veuillez respecter les conventions suivantes :
- Ajouter des tests pour les nouvelles fonctionnalités
- Documenter les nouvelles métriques
- Maintenir la cohérence dans le nommage des métriques

## Licence

Ce module est fourni sous licence privée pour EVIL2ROOT Trading. 
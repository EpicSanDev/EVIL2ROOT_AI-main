# Système d'Auto-Apprentissage pour Modèles de Trading

Ce module implémente un système complet d'auto-apprentissage permettant aux modèles de trading d'analyser leurs performances, de détecter leurs erreurs et d'ajuster automatiquement leurs paramètres pour s'améliorer continuellement.

## Fonctionnalités principales

- **Journal de Trading** : Enregistre et gère l'historique complet des transactions
- **Analyse de Performance** : Évalue les performances des modèles selon divers critères et métriques
- **Détection d'Erreurs** : Identifie automatiquement les patterns d'erreurs dans les transactions perdantes
- **Ajustement Automatique** : Modifie les paramètres des modèles en fonction des erreurs détectées
- **Visualisations** : Génère des rapports visuels pour faciliter l'analyse humaine
- **Orchestration** : Coordonne l'ensemble du processus d'auto-apprentissage

## Architecture du système

Le système est composé de plusieurs modules complémentaires :

1. **TradeJournal** (`trade_journal.py`) : Base de données et interface pour l'historique des transactions
2. **PerformanceAnalyzer** (`performance_analyzer.py`) : Analyse statistique des performances
3. **ErrorDetector** (`error_detector.py`) : Détection et classification des erreurs
4. **ModelAdjuster** (`model_adjuster.py`) : Ajustement des paramètres des modèles
5. **LearningOrchestrator** (`learning_orchestrator.py`) : Coordination du flux de travail complet

## Comment utiliser le système

### Initialisation de base

```python
from src.models.auto_learning import (
    TradeJournal, 
    ErrorDetector,
    PerformanceAnalyzer, 
    ModelAdjuster
)
from src.models.auto_learning.learning_orchestrator import LearningOrchestrator

# Initialiser l'orchestrateur qui coordonne tous les composants
orchestrator = LearningOrchestrator(
    config_path="config.json",  # Facultatif
    db_path="data/trade_journal.db",
    models_dir="saved_models",
    reports_dir="data/reports"
)
```

### Enregistrer une transaction

```python
# Données d'une transaction
trade_data = {
    "symbol": "BTC/USD",
    "entry_time": "2023-05-10T14:30:00",
    "exit_time": "2023-05-11T09:15:00",
    "entry_price": 27500.0,
    "exit_price": 28200.0,
    "position_size": 0.5,
    "direction": "BUY",
    "pnl": 350.0,
    "pnl_percent": 2.54,
    "fee": 13.75,
    "strategy_name": "trend_following_v1",
    "model_version": "1.0.0",
    "entry_signals": {...},
    "exit_signals": {...},
    "market_conditions": {...}
}

trade_id = orchestrator.log_trade(trade_data)
```

### Analyser les performances

```python
# Analyser les performances des 30 derniers jours
performance_data = orchestrator.analyze_performance(days=30)

# Génération de visualisations
visualization_paths = orchestrator.performance_analyzer.generate_performance_visualizations()
```

### Détecter et analyser les erreurs

```python
# Analyser les erreurs des 90 derniers jours
error_analysis = orchestrator.detect_errors(days=90)

# Les erreurs détectées sont accessibles via
detected_errors = error_analysis.get('detected_errors', [])
error_patterns = error_analysis.get('error_patterns', {})
```

### Ajuster les modèles automatiquement

```python
# Ajuster les modèles en fonction des erreurs et performances récentes
adjustment_results = orchestrator.adjust_models(days=30)

# Vérifier les ajustements effectués
adjusted_models = adjustment_results.get('adjustments_made', [])
```

### Exécuter un cycle complet d'apprentissage

```python
# Exécute toutes les étapes du cycle d'auto-apprentissage
learning_results = orchestrator.run_learning_cycle()
```

## Types d'erreurs détectés

Le système peut détecter plusieurs types d'erreurs de trading, notamment :

- **TIMING_ERROR** : Entrée ou sortie mal chronométrée
- **SIZE_ERROR** : Taille de position inadaptée au risque
- **TREND_MISREAD** : Mauvaise lecture de la tendance 
- **SIGNAL_CONFLICT** : Signaux contradictoires ignorés
- **OVERTRADING** : Trop de transactions en peu de temps
- **PREMATURE_EXIT** : Sortie prématurée avant objectif
- **DELAYED_EXIT** : Sortie retardée après signal
- **IGNORED_STOP** : Stop-loss ignoré
- **MARKET_CONDITION** : Inadaptation aux conditions de marché
- **HIGH_VOLATILITY** : Sous-estimation de la volatilité

## Types de modèles supportés

Le système d'auto-apprentissage peut ajuster différents types de modèles :

- **RL** : Modèles de reinforcement learning
- **Ensemble** : Modèles combinant plusieurs stratégies
- **Price** : Modèles basés sur les analyses de prix et indicateurs techniques
- **Sentiment** : Modèles basés sur l'analyse de sentiment

## Exemple complet

Consultez le script d'exemple dans `src/examples/auto_learning_example.py` pour voir comment utiliser l'ensemble du système dans un scénario complet.

## Configuration

Le système peut être configuré à l'aide d'un fichier JSON avec les options suivantes :

```json
{
    "learning_frequency_days": 7,
    "analysis_window_days": 30,
    "error_analysis_window_days": 90,
    "min_trades_for_analysis": 10,
    "auto_adjust_enabled": true,
    "visualization_enabled": true,
    "email_reports_enabled": false,
    "email_recipients": []
}
```

## Extension et personnalisation

Le système est conçu pour être facilement extensible. Vous pouvez :

1. Ajouter de nouveaux types d'erreurs dans `ErrorDetector`
2. Implémenter de nouvelles métriques de performance dans `PerformanceAnalyzer`
3. Ajouter des stratégies d'ajustement pour de nouveaux types de modèles dans `ModelAdjuster`
4. Étendre le système de journalisation dans `TradeJournal` 
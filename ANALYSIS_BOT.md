# Bot d'Analyse de Marché EVIL2ROOT

Ce document décrit les fonctionnalités et l'utilisation du bot d'analyse de marché intégré au système EVIL2ROOT Trading Bot.

## Table des matières

- [Aperçu](#aperçu)
- [Fonctionnalités principales](#fonctionnalités-principales)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
  - [Analyse quotidienne](#analyse-quotidienne)
  - [Analyses planifiées](#analyses-planifiées)
  - [Entraînement forcé](#entraînement-forcé)
- [Personnalisation des analyses](#personnalisation-des-analyses)
- [Rapports d'analyse](#rapports-danalyse)
- [Intégration avec le trading](#intégration-avec-le-trading)

## Aperçu

Le bot d'analyse de marché EVIL2ROOT est un composant sophistiqué qui effectue des analyses approfondies des marchés financiers en utilisant une combinaison de techniques d'apprentissage automatique, d'analyse technique traditionnelle et d'analyse de sentiment. Il fournit des rapports détaillés pour aider à la prise de décision de trading, que ce soit pour le trading automatisé ou pour les traders manuels.

Le système comprend deux composants d'analyse principaux:
1. **Daily Analysis Bot** (`app/daily_analysis_bot.py`): Analyse quotidienne approfondie des marchés
2. **Market Analysis Scheduler** (`app/market_analysis_scheduler.py`): Planificateur d'analyses régulières

## Fonctionnalités principales

### Daily Analysis Bot

- Analyse technique complète (indicateurs, patterns, niveaux)
- Analyse fondamentale basique (métriques clés, nouvelles récentes)
- Prédictions de prix basées sur des modèles d'apprentissage profond
- Analyse du sentiment de marché à partir des actualités et médias sociaux
- Identification des tendances et mouvements potentiels
- Génération de rapports détaillés au format texte et visuel
- Recommandations de trading avec niveaux d'entrée/sortie suggérés

### Market Analysis Scheduler

- Planification flexible des analyses à différents intervalles
- Analyses spécifiques en fonction des fuseaux horaires et sessions de marché
- Déclenchement automatique d'analyses lors d'événements de marché importants
- Distribution des rapports via Telegram et interface web
- Archivage et historique des analyses précédentes
- Détection d'anomalies et alertes en temps réel

## Architecture

```
┌─────────────────────────┐
│                         │
│  Market Analysis        │
│  Scheduler              │
│                         │
└────────────┬────────────┘
             │
             │ Déclenche
             ▼
┌─────────────────────────┐
│                         │
│  Daily Analysis Bot     │
│                         │
└────────────┬────────────┘
             │
             │ Utilise
             ▼
┌─────────────────────────┐
│                         │
│  Modèles d'IA           │
│  - PricePrediction      │
│  - SentimentAnalysis    │
│  - TechnicalIndicators  │
│  - ...                  │
│                         │
└────────────┬────────────┘
             │
             │ Produit
             ▼
┌─────────────────────────┐
│                         │
│  Rapports d'Analyse     │
│                         │
└────────────┬────────────┘
             │
             │ Distribués via
             ▼
┌─────────────┬─────────────┐
│             │             │
│  Telegram   │  Web UI     │
│             │             │
└─────────────┴─────────────┘
```

## Configuration

La configuration du bot d'analyse se fait principalement via le fichier `.env`:

```
# Analysis Configuration
ENABLE_DAILY_ANALYSIS=true
DAILY_ANALYSIS_TIME=00:00  # UTC
MARKET_ANALYSIS_INTERVAL=60  # minutes
ANALYSIS_DETAIL_LEVEL=detailed  # basic|detailed|comprehensive

# Symbols to analyze
ANALYSIS_SYMBOLS=AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,BTC-USD,ETH-USD

# Model Configuration for Analysis
USE_TRANSFORMER_MODEL=true
USE_LSTM_MODEL=true
USE_SENTIMENT_ANALYSIS=true
USE_NEWS_ANALYSIS=true

# Telegram Configuration for Reports
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

Options avancées dans le fichier `config/analysis_config.json` (si présent) :

```json
{
  "technical_indicators": {
    "enable_all": true,
    "specific_indicators": ["RSI", "MACD", "Bollinger", "Ichimoku", "Fibonacci"]
  },
  "chart_patterns": {
    "enable_detection": true,
    "confidence_threshold": 0.65
  },
  "report_generation": {
    "include_charts": true,
    "include_predictions": true,
    "prediction_timeframes": ["1d", "3d", "7d"],
    "language": "fr"
  },
  "market_events": {
    "track_economic_calendar": true,
    "track_earnings": true
  }
}
```

## Utilisation

### Analyse quotidienne

Pour lancer une analyse quotidienne manuellement:

```bash
python start_daily_analysis.py
```

Options disponibles:
- `--force-train`: Force l'entraînement des modèles avant l'analyse
- `--symbols`: Liste spécifique de symboles à analyser (ex: `--symbols AAPL,MSFT,TSLA`)
- `--detail`: Niveau de détail de l'analyse (`basic`, `detailed`, `comprehensive`)
- `--no-telegram`: Désactive l'envoi de rapports via Telegram
- `--save-only`: Sauvegarde les rapports sans les envoyer

Exemple:
```bash
python start_daily_analysis.py --symbols AAPL,TSLA --detail comprehensive --force-train
```

### Analyses planifiées

Pour lancer le planificateur d'analyses:

```bash
./start_market_scheduler.sh
```

Le planificateur exécutera les analyses selon les intervalles configurés. Pour arrêter le planificateur:

```bash
./stop_market_scheduler.sh
```

### Entraînement forcé

Pour forcer l'entraînement des modèles avant les analyses:

```bash
python start_daily_analysis.py --force-train
```

Ou avec Docker:
```bash
./start_docker_force_train.sh
```

## Personnalisation des analyses

Le bot d'analyse peut être personnalisé de plusieurs façons:

1. **Plugins d'analyse** (`app/plugins/`):
   - Créez des plugins d'analyse personnalisés
   - Activez/désactivez des plugins spécifiques via l'interface web

2. **Indicateurs techniques**:
   - Configurez les indicateurs à utiliser dans `config/analysis_config.json`
   - Ajustez les paramètres des indicateurs existants

3. **Sources de données**:
   - Configurez des sources de données additionnelles dans `app/models/news_retrieval.py`
   - Ajoutez des sources de sentiment de marché personnalisées

4. **Format des rapports**:
   - Personnalisez les templates de rapport dans `app/templates/reports/`
   - Ajustez le format et le contenu des notifications Telegram

## Rapports d'analyse

Les rapports d'analyse générés incluent:

### 1. Analyse technique
- Tendances actuelles (court, moyen, long terme)
- Niveaux de support et résistance clés
- Indicateurs techniques (RSI, MACD, moyennes mobiles, etc.)
- Patterns graphiques identifiés (têtes et épaules, triangles, etc.)

### 2. Analyse de prix
- Prédictions de prix pour différentes périodes
- Niveaux de volatilité attendus
- Zones de prix importantes à surveiller

### 3. Analyse fondamentale
- Événements récents affectant l'actif
- Données fondamentales clés (pour les actions)
- Métriques on-chain (pour les crypto-monnaies)

### 4. Analyse de sentiment
- Sentiment global du marché
- Analyse des actualités récentes
- Sentiment des médias sociaux
- Changements de sentiment notables

### 5. Recommandations
- Opportunités de trading potentielles
- Niveaux d'entrée suggérés
- Niveaux de take-profit et stop-loss recommandés
- Notation de confiance pour chaque recommandation

## Intégration avec le trading

Le bot d'analyse s'intègre avec le système de trading de plusieurs façons:

1. **Validation des décisions**: Les analyses peuvent être utilisées pour valider les décisions de trading automatiques

2. **Filtrage de marché**: Les conditions de marché identifiées par les analyses peuvent filtrer les actifs à trader

3. **Ajustement des paramètres**: Les résultats d'analyse peuvent ajuster dynamiquement les paramètres de trading

4. **Préparation du trading**: Les analyses nocturnes préparent le système pour la session de trading suivante

Pour configurer l'intégration entre l'analyse et le trading, utilisez les paramètres suivants dans `.env`:

```
# Integration Configuration
USE_ANALYSIS_FOR_TRADING=true
ANALYSIS_CONFIDENCE_THRESHOLD=0.7
ADJUST_RISK_BASED_ON_ANALYSIS=true
```

---

Pour plus d'informations sur l'utilisation avancée du bot d'analyse, consultez la documentation complète ou contactez l'équipe de support. 
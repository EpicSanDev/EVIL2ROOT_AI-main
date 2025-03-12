# Composants Principaux du Système EVIL2ROOT

Ce document détaille les différents composants qui constituent le système de trading EVIL2ROOT, leurs rôles, fonctionnalités et interactions.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Trading Bot](#trading-bot)
3. [AI Validator](#ai-validator)
4. [Web UI](#web-ui)
5. [Daily Analysis Bot](#daily-analysis-bot)
6. [Market Analysis Scheduler](#market-analysis-scheduler)
7. [Model Trainer](#model-trainer)
8. [Telegram Bot](#telegram-bot)
9. [Monitoring System](#monitoring-system)
10. [Interactions entre composants](#interactions-entre-composants)

## Vue d'ensemble

Les composants du système EVIL2ROOT sont conçus selon une architecture microservices, chacun ayant une responsabilité spécifique et communiquant via Redis. Cette approche permet une maintenance plus facile, un développement indépendant et une meilleure résilience.

## Trading Bot

**Fichier principal** : `app/trading.py`

**Rôle** : Composant central qui analyse les marchés, génère des signaux de trading et exécute les transactions.

**Fonctionnalités clés** :
- Collecte des données de marché en temps réel
- Analyse technique via indicateurs traditionnels (RSI, MACD, etc.)
- Génération de signaux à partir de multiples modèles
- Envoi des signaux pour validation à l'AI Validator
- Exécution des transactions validées
- Gestion des positions ouvertes (trailing stops, take-profits)
- Calcul des tailles de position selon le risque

**Méthodes principales** :
```python
def analyze_market(self, symbol, timeframe='1h'):
    """Analyse le marché pour un symbole et timeframe donnés."""
    
def generate_signals(self, symbol, data):
    """Génère des signaux de trading à partir des modèles."""
    
def validate_trade(self, symbol, action, entry_price, stop_loss, take_profit):
    """Envoie une demande de validation à l'AI Validator."""
    
def execute_trade(self, trade_data):
    """Exécute une transaction validée."""
    
def manage_open_positions(self):
    """Gère les positions ouvertes (ajustement des SL/TP, etc.)."""
```

**Configuration** :
- Instruments tradés (via variable d'environnement `SYMBOLS`)
- Taille maximale des positions (`MAX_POSITION_SIZE`)
- Risque par transaction (`RISK_PER_TRADE`)
- Mode simulation ou réel (`ENABLE_LIVE_TRADING`)

## AI Validator

**Fichier principal** : `app/ai_trade_validator.py`

**Rôle** : Système de validation des décisions de trading via une analyse avancée.

**Fonctionnalités clés** :
- Analyse multi-facteur des propositions de trading
- Communication avec Claude 3.7 pour une analyse avancée
- Analyse du sentiment des actualités financières
- Vérification de la compatibilité avec les tendances de marché
- Calcul de scores de confiance pour les décisions
- Validation ou rejet des transactions proposées

**Architecture interne** :
- Écouteur Redis pour les demandes de validation
- Modèles ML internes pour la pré-validation
- Client API OpenRouter pour la communication avec Claude 3.7
- Analyseur de sentiment pour les actualités financières
- Système de décision combinant plusieurs facteurs

**Exemple de flux de validation** :
1. Réception d'une demande de validation via Redis
2. Analyse préliminaire avec modèles ML internes
3. Génération de prompt détaillé pour Claude 3.7
4. Envoi du prompt à Claude 3.7 via OpenRouter
5. Analyse de la réponse de Claude 3.7
6. Combinaison avec l'analyse de sentiment et autres facteurs
7. Calcul du score de confiance final
8. Envoi de la décision (validé/rejeté) au Trading Bot

## Web UI

**Fichier principal** : `app/routes.py`

**Rôle** : Interface utilisateur web pour monitorer et configurer le système.

**Fonctionnalités clés** :
- Tableau de bord en temps réel des performances
- Visualisation des transactions et signaux
- Graphiques interactifs des marchés et indicateurs
- Configuration du système (paramètres, symboles, etc.)
- Statistiques de performance et métriques
- Journaux du système et notifications

**Technologies utilisées** :
- Backend : Flask
- Frontend : Bootstrap, Chart.js, DataTables
- Temps réel : Socket.IO
- Visualisation : Plotly

**Routes principales** :
```python
@app.route('/')
def dashboard():
    """Page principale avec tableau de bord."""
    
@app.route('/trades')
def trades():
    """Historique des transactions."""
    
@app.route('/symbols')
def symbols():
    """Vue détaillée par symbole."""
    
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Configuration du système."""
    
@app.route('/api/v1/performance')
def api_performance():
    """API pour les données de performance."""
```

**Structure des templates** :
- `/templates/layout.html` - Template de base
- `/templates/dashboard.html` - Tableau de bord principal
- `/templates/trades.html` - Historique des transactions
- `/templates/settings.html` - Page de configuration
- `/templates/analytics.html` - Analyses approfondies

## Daily Analysis Bot

**Fichier principal** : `app/daily_analysis_bot.py`

**Rôle** : Réalisation d'analyses quotidiennes approfondies des marchés.

**Fonctionnalités clés** :
- Analyse technique complète sur différents timeframes
- Identification des opportunités de trading à moyen terme
- Détection des patterns chartistes majeurs
- Analyse de la force relative entre actifs
- Génération de rapports détaillés
- Distribution des rapports via Telegram et email

**Processus d'analyse quotidienne** :
1. Récupération des données historiques pour tous les symboles
2. Calcul des indicateurs techniques et patterns
3. Analyse de corrélation entre les actifs
4. Identification des opportunités potentielles
5. Génération de rapports personnalisés
6. Distribution des rapports aux utilisateurs

**Configuration** :
- Heure d'exécution configurable
- Profondeur d'analyse paramétrable
- Format des rapports personnalisable
- Niveau de détail ajustable

## Market Analysis Scheduler

**Fichier principal** : `app/market_analysis_scheduler.py`

**Rôle** : Planification et coordination des analyses de marché à intervalles réguliers.

**Fonctionnalités clés** :
- Planification des tâches d'analyse à intervalles configurables
- Gestion des priorités selon les conditions de marché
- Optimisation des ressources système
- Coordination avec les autres composants
- Adaptation aux heures de marché (sessions de trading)
- Exécution d'analyses spéciales lors d'événements importants

**Fonctionnement** :
- Utilisation de l'APScheduler pour la planification des tâches
- Gestion de files d'attente Redis pour les tâches
- Mécanismes de verrouillage pour éviter les analyses simultanées
- Surveillance des ressources système pour éviter la surcharge

**Types d'analyses planifiées** :
- Analyses horaires rapides
- Analyses de session (ouverture/fermeture de marchés)
- Analyses journalières approfondies
- Analyses hebdomadaires stratégiques
- Analyses événementielles (annonces économiques, etc.)

## Model Trainer

**Fichier principal** : `app/model_trainer.py`

**Rôle** : Entraînement et optimisation des modèles d'IA et d'apprentissage automatique.

**Fonctionnalités clés** :
- Entraînement régulier des modèles sur de nouvelles données
- Optimisation des hyperparamètres via recherche bayésienne
- Évaluation des performances des modèles
- Gestion des versions des modèles
- Fine-tuning des modèles par symbole
- Test A/B des différentes variantes de modèles

**Modèles gérés** :
- Modèles de prédiction de prix (LSTM, GRU)
- Modèles de classification des signaux
- Modèles d'analyse de sentiment
- Agents d'apprentissage par renforcement
- Modèles de détection de patterns

**Cycle d'entraînement** :
1. Collection des données récentes
2. Préparation et prétraitement des données
3. Entraînement des modèles (optimisation si nécessaire)
4. Évaluation des performances
5. Déploiement des nouveaux modèles si supérieurs
6. Archivage des versions précédentes

## Telegram Bot

**Fichier principal** : `app/telegram_bot.py`

**Rôle** : Interface Telegram pour les notifications et le contrôle du système.

**Fonctionnalités clés** :
- Notifications en temps réel des transactions
- Alertes sur les événements importants
- Rapports d'analyse quotidiens
- Commandes pour vérifier l'état du système
- Configuration basique à distance
- Alertes de sécurité et de performance

**Commandes disponibles** :
- `/status` - État actuel du système
- `/performance` - Statistiques de performance
- `/positions` - Positions actuellement ouvertes
- `/trades` - Dernières transactions
- `/report` - Générer un rapport d'analyse rapide
- `/settings` - Afficher/modifier les paramètres
- `/help` - Aide sur les commandes disponibles

## Monitoring System

**Fichier principal** : `app/monitoring.py` et `app/monitoring_enhanced.py`

**Rôle** : Surveillance de la santé et des performances du système.

**Fonctionnalités clés** :
- Surveillance des ressources système (CPU, RAM, GPU)
- Monitoring des performances de trading
- Détection d'anomalies dans le comportement du système
- Alertes en cas de problèmes détectés
- Métriques de latence et de performance
- Journalisation avancée des événements

**Métriques surveillées** :
- Utilisation des ressources système
- Temps de réponse des modèles
- Taux de validation des transactions
- Drawdown et autres métriques de risque
- Stabilité de la connexion aux API externes
- Santé de la base de données

## Interactions entre composants

Le diagramme suivant résume les principales interactions entre les composants du système :

```
┌────────────────┐           ┌────────────────┐
│                │   1. Data │                │
│  Trading Bot   │──────────►│  AI Validator  │
│                │◄──────────│                │
└───────┬────────┘   2. Validation └─────────┬──────┘
        │                                    │
        │                                    │
        │      ┌────────────────┐           │
        │      │                │           │
        ├─────►│    Web UI      │◄──────────┤
        │      │                │           │
        │      └────────┬───────┘           │
        │               │                   │
        │               │                   │
        │      ┌────────▼───────┐           │
        │      │                │           │
        ├─────►│   Daily        │◄──────────┤
        │      │   Analysis Bot │           │
        │      │                │           │
        │      └────────┬───────┘           │
        │               │                   │
        │      ┌────────▼───────┐           │
        │      │                │           │
        └─────►│   Market       │◄──────────┘
               │   Scheduler    │
               │                │
               └────────────────┘
```

**Flux typiques** :

1. **Génération et validation de trade** :
   - Trading Bot détecte une opportunité
   - Envoi à AI Validator pour validation
   - Si validé, exécution de la transaction
   - Notification via Telegram Bot
   - Mise à jour des métriques dans Web UI

2. **Analyse quotidienne** :
   - Market Scheduler déclenche l'analyse
   - Daily Analysis Bot effectue l'analyse
   - Résultats stockés en base de données
   - Rapport envoyé via Telegram Bot
   - Mise à jour des informations dans Web UI

3. **Entraînement de modèle** :
   - Market Scheduler déclenche l'entraînement
   - Model Trainer collecte les données récentes
   - Optimisation et entraînement des modèles
   - Évaluation et déploiement si améliorations
   - Mise à jour des métriques de performance 
# Changelog

Toutes les modifications notables apportées au projet EVIL2ROOT Trading Bot seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2024-03-12

### Ajouté
- Nouveau Market Analysis Scheduler pour des analyses programmées à intervalles réguliers
- Support pour les modèles Transformer dans l'analyse de séries temporelles
- Système avancé de gestion de plugins d'analyse
- Interface de configuration des plugins via l'interface web
- Tableau de bord avancé avec visualisations détaillées
- Nouveau modèle d'IA pour la récupération et l'analyse des actualités financières
- Script de démarrage pour l'analyse de marché planifiée `start_market_scheduler.sh`
- Optimisation bayésienne des hyperparamètres pour les modèles de prédiction

### Amélioré
- Mise à jour du système de validation IA pour utiliser Claude 3.7 Sonnet via OpenRouter
- Architecture de réseau neural améliorée avec couches de normalisation et dropout
- Performances des modèles d'apprentissage par renforcement
- Documentation complète du projet (README, ARCHITECTURE, USER_GUIDE, etc.)
- Interface utilisateur avec tableaux de bord plus détaillés
- Optimisation des performances du système d'analyse en temps réel
- Mécanisme de filtrage intelligent des actifs basé sur l'analyse de marché

### Corrections
- Résolution des problèmes de mémoire avec les grands modèles LSTM
- Correction des erreurs d'affichage dans l'interface web
- Amélioration de la stabilité de la connexion Telegram
- Correction des fuites de mémoire dans le traitement des données de marché
- Optimisation des requêtes à la base de données

## [1.4.0] - 2024-03-06

### Ajouté
- Intégration Docker complète avec docker-compose
- Bot d'analyse quotidienne avec planification flexible
- Support pour l'apprentissage par renforcement dans les décisions de trading
- Interface web améliorée avec visualisations avancées
- Système de notification Telegram pour les signaux et analyses
- Module d'analyse de sentiment basé sur les actualités financières
- Scripts de déploiement pour différents environnements

### Amélioré
- Performance des modèles de prédiction de prix
- Système de gestion des risques avec calcul dynamique
- Documentation et instructions d'installation
- Optimisation des performances pour les environnements à ressources limitées
- Mécanisme de backtesting pour l'évaluation des stratégies

### Corrections
- Problèmes de synchronisation dans le traitement des données en temps réel
- Erreurs dans le calcul des indicateurs techniques
- Problèmes de compatibilité avec différentes versions de Python
- Fuites de mémoire dans le traitement des grandes séries temporelles

## [1.3.0] - 2024-02-15

### Ajouté
- Nouveaux modèles de prédiction basés sur LSTM et GRU
- Fonctionnalités d'analyse technique avancée
- Support pour le trading de crypto-monnaies
- Module de validation des décisions de trading par IA
- Interface web de base pour le monitoring
- API REST pour l'intégration avec d'autres services

### Amélioré
- Architecture du système pour une meilleure modularité
- Performance et précision des modèles d'IA
- Gestion des données de marché avec mise en cache efficace
- Documentation du code et des fonctionnalités

### Corrections
- Bugs dans le calcul des indicateurs techniques
- Problèmes de stabilité dans le traitement des données en temps réel
- Erreurs de connexion aux APIs externes
- Problèmes de performance avec les grands ensembles de données

## [1.2.0] - 2024-01-10

### Ajouté
- Modèles de machine learning pour la prédiction des prix
- Système de gestion des risques
- Support pour plusieurs actifs financiers
- Système de logging amélioré
- Configuration via variables d'environnement

### Amélioré
- Précision des signaux de trading
- Performance du traitement des données
- Interface en ligne de commande
- Structure du code pour une meilleure maintenabilité

### Corrections
- Bugs dans les calculs de rendement
- Problèmes de gestion de la mémoire
- Erreurs dans la gestion des exceptions

## [1.1.0] - 2023-12-05

### Ajouté
- Support pour les indicateurs techniques de base
- Base de données PostgreSQL pour le stockage des données
- Système de backtesting simple
- Documentation initiale

### Amélioré
- Algorithmes de trading
- Performance globale du système
- Structure du projet

### Corrections
- Bugs divers dans la logique de trading
- Problèmes de formatage des données

## [1.0.0] - 2023-11-01

### Ajouté
- Version initiale du bot de trading
- Support pour les actions et ETFs
- Stratégies de trading basiques
- Configuration par fichier
- Logs de base

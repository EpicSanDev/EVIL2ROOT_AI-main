# EVIL2ROOT Trading Bot - Documentation Technique

## Introduction

EVIL2ROOT Trading Bot est un système complet de trading automatisé qui utilise l'intelligence artificielle pour prendre et valider des décisions de trading. Cette documentation technique présente l'ensemble du système, son architecture, ses composants et son fonctionnement.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Composants principaux](#composants-principaux)
4. [Modèles d'IA](#modèles-dia)
5. [Base de données](#base-de-données)
6. [API et interfaces](#api-et-interfaces)
7. [Déploiement](#déploiement)
8. [Guide d'utilisation](#guide-dutilisation)
9. [Guide de développement](#guide-de-développement)
10. [Sécurité](#sécurité)

## Vue d'ensemble

EVIL2ROOT Trading Bot est conçu comme une architecture de microservices communiquant via Redis et utilisant PostgreSQL comme stockage persistant. Le système combine plusieurs approches d'intelligence artificielle pour générer des signaux de trading et dispose d'un système de validation supplémentaire basé sur Claude 3.7 pour améliorer la qualité des décisions.

Le projet est structuré de manière modulaire pour permettre une évolution indépendante des différents composants et fournit une interface web complète pour la surveillance et la configuration.

## Architecture

[Voir la documentation détaillée sur l'architecture](./ARCHITECTURE.md)

Le système est organisé autour des principes suivants :
- Microservices communiquant via Redis
- Base de données PostgreSQL pour la persistance
- Architecture orientée message pour le découplage des composants
- Interface web Flask pour la surveillance et la configuration
- Exécution dans des conteneurs Docker pour la portabilité et l'isolation

## Composants principaux

[Voir la documentation détaillée sur les composants](./COMPONENTS.md)

Les composants majeurs du système sont :

1. **Trading Bot** : Composant principal qui analyse les marchés, génère des signaux et exécute les transactions
2. **AI Validator** : Service qui valide les décisions de trading via une analyse avancée 
3. **Web UI** : Interface utilisateur basée sur Flask pour surveiller et configurer le système
4. **Daily Analysis Bot** : Réalise des analyses quotidiennes approfondies des marchés
5. **Market Analysis Scheduler** : Planifie et coordonne les analyses de marché

## Modèles d'IA

[Voir la documentation détaillée sur les modèles d'IA](./AI_MODELS.md)

Le système utilise plusieurs modèles d'IA complémentaires :

1. **PricePredictionModel** : Modèles LSTM et GRU pour la prédiction des prix
2. **IndicatorManagementModel** : Analyse des indicateurs techniques
3. **RiskManagementModel** : Évaluation et gestion des risques
4. **TpSlManagementModel** : Gestion des take-profit et stop-loss
5. **RLTradingModel** : Agent d'apprentissage par renforcement pour les décisions de trading
6. **SentimentAnalyzer** : Analyse du sentiment des news financières
7. **TransformerModel** : Architecture Transformer pour l'analyse de séries temporelles
8. **NewsRetrieval** : Collecte et analyse d'actualités financières

## Base de données

[Voir la documentation détaillée sur la base de données](./DATABASE.md)

Le système utilise PostgreSQL avec un schéma optimisé pour stocker :
- Historique des transactions
- Signaux de trading
- Données de marché
- Métriques de performance
- Configurations du système
- Résultats d'analyses

## API et interfaces

[Voir la documentation détaillée sur les API](./API.md)

Le système expose plusieurs interfaces :
- API REST pour l'intégration avec d'autres systèmes
- Interface web pour la configuration et la surveillance
- Webhooks pour les notifications
- Bot Telegram pour les alertes mobiles

## Déploiement

[Voir la documentation détaillée sur le déploiement](./DEPLOYMENT.md)

Le système peut être déployé via :
- Docker et Docker Compose (recommandé)
- Installation manuelle (pour le développement)
- Déploiement cloud (AWS, Digital Ocean)

## Guide d'utilisation

[Voir le guide utilisateur complet](./USER_GUIDE.md)

Le guide utilisateur couvre :
- Installation et configuration
- Démarrage des services
- Configuration du trading
- Surveillance des performances
- Résolution des problèmes courants

## Guide de développement

[Voir le guide développeur](./DEVELOPMENT.md)

Le guide de développement explique :
- Structure du code
- Conventions de code
- Ajout de nouveaux modèles
- Extension des fonctionnalités
- Tests et validation

## Sécurité

[Voir la documentation sur la sécurité](./SECURITY.md)

La documentation sur la sécurité couvre :
- Gestion des clés API
- Sécurisation de la base de données
- Mesures de sécurité réseau
- Gestion des accès
- Chiffrement des données sensibles 
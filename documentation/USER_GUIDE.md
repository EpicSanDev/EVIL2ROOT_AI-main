# Guide Utilisateur du Système EVIL2ROOT Trading Bot

Ce guide utilisateur vous aide à comprendre et à utiliser efficacement le système EVIL2ROOT Trading Bot. Il couvre l'installation, la configuration et l'utilisation quotidienne du bot.

## Table des matières

1. [Introduction](#introduction)
2. [Premiers pas](#premiers-pas)
   - [Installation](#installation)
   - [Configuration initiale](#configuration-initiale)
   - [Authentification](#authentification)
3. [Interface utilisateur web](#interface-utilisateur-web)
   - [Tableau de bord](#tableau-de-bord)
   - [Visualisation des données](#visualisation-des-données)
   - [Configuration des stratégies](#configuration-des-stratégies)
4. [Bot Telegram](#bot-telegram)
   - [Configuration](#configuration-du-bot-telegram)
   - [Commandes disponibles](#commandes-disponibles)
   - [Alertes automatiques](#alertes-automatiques)
5. [Configuration des stratégies de trading](#configuration-des-stratégies-de-trading)
   - [Création d'une stratégie](#création-dune-stratégie)
   - [Paramétrage des indicateurs](#paramétrage-des-indicateurs)
   - [Gestion du risque](#gestion-du-risque)
6. [Gestion des API de trading](#gestion-des-api-de-trading)
   - [Ajout d'une nouvelle API](#ajout-dune-nouvelle-api)
   - [Sécurisation des clés API](#sécurisation-des-clés-api)
7. [Suivi des performances](#suivi-des-performances)
   - [Métriques disponibles](#métriques-disponibles)
   - [Rapports](#rapports)
8. [Maintenance](#maintenance)
   - [Sauvegarde des données](#sauvegarde-des-données)
   - [Mise à jour du système](#mise-à-jour-du-système)
9. [Dépannage](#dépannage)
   - [Problèmes courants](#problèmes-courants)
   - [Journaux d'erreurs](#journaux-derreurs)
10. [Ressources supplémentaires](#ressources-supplémentaires)

## Introduction

Le système EVIL2ROOT Trading Bot est une plateforme de trading algorithmique combinant l'intelligence artificielle avancée et des indicateurs techniques traditionnels pour automatiser les stratégies de trading sur différents marchés financiers.

Ce système peut surveiller les marchés 24h/24, analyser les données, générer des signaux et exécuter des transactions automatiquement selon les stratégies définies. Il est conçu pour être convivial tout en offrant une grande flexibilité pour les traders expérimentés.

**Fonctionnalités principales :**
- Analyse de marché multi-actifs (crypto-monnaies, forex, actions, etc.)
- Modèles d'IA pour la prédiction de prix et l'analyse de sentiment
- Exécution automatique des transactions
- Support pour plusieurs échanges et courtiers
- Interface web intuitive avec tableaux de bord personnalisables
- Intégration avec Telegram pour le contrôle à distance et les alertes
- Journalisation complète des transactions et performances

## Premiers pas

### Installation

Pour installer le système EVIL2ROOT Trading Bot, référez-vous à la [documentation de déploiement](DEPLOYMENT.md) qui fournit des instructions détaillées pour différentes méthodes d'installation.

### Configuration initiale

Après l'installation, vous devez effectuer une configuration initiale du système :

1. **Accès à l'interface web** : Ouvrez votre navigateur et accédez à l'URL où le système est hébergé (par exemple `http://localhost:5000` pour une installation locale).

2. **Création d'un compte administrateur** : Lors de la première connexion, vous serez invité à créer un compte administrateur :
   - Définissez un nom d'utilisateur et un mot de passe forts
   - Fournissez une adresse e-mail valide
   - Activez l'authentification à deux facteurs (fortement recommandé)

3. **Configuration de la base de données** : Vérifiez que la connexion à la base de données fonctionne correctement.

4. **Configuration des API** : Ajoutez vos clés API pour les plateformes de trading que vous souhaitez utiliser.

### Authentification

Le système utilise un système d'authentification robuste :

1. **Connexion standard** : Nom d'utilisateur/email et mot de passe.

2. **Authentification à deux facteurs (2FA)** :
   - Scannez le code QR avec une application d'authentification comme Google Authenticator ou Authy
   - Entrez le code à 6 chiffres généré par l'application pour compléter la connexion
   - Conservez les codes de récupération dans un endroit sûr

3. **Gestion des sessions** :
   - Les sessions expirent après 15 minutes d'inactivité
   - Vous pouvez consulter et révoquer les sessions actives depuis votre profil

## Interface utilisateur web

### Tableau de bord

Le tableau de bord principal est entièrement personnalisable et affiche :

1. **Vue d'ensemble du portefeuille** :
   - Valeur totale actuelle
   - Performance globale (quotidienne, hebdomadaire, mensuelle)
   - Répartition des actifs

2. **Statut des bots actifs** :
   - Liste des stratégies actives
   - Performance individuelle
   - Signaux récents générés

3. **Tendances du marché** :
   - Graphiques en temps réel des actifs surveillés
   - Indicateurs techniques superposés
   - Signaux d'achat/vente

Pour personnaliser votre tableau de bord :
1. Cliquez sur l'icône "Modifier" en haut à droite
2. Faites glisser et déposez les widgets selon vos préférences
3. Configurez chaque widget en cliquant sur l'icône d'engrenage
4. Cliquez sur "Enregistrer" pour conserver vos modifications

### Visualisation des données

La section de visualisation des données offre des outils avancés pour analyser le marché :

1. **Graphiques interactifs** :
   - Bougies japonaises, graphiques linéaires, barres, etc.
   - Intervalles de temps personnalisables (1m, 5m, 15m, 1h, 4h, 1j, etc.)
   - Superposition d'indicateurs techniques (MA, RSI, MACD, etc.)
   - Outils de dessin (lignes de tendance, Fibonacci, etc.)

2. **Analyse technique automatisée** :
   - Détection automatique des figures chartistes
   - Points de support et résistance
   - Signaux des indicateurs techniques

3. **Insights IA** :
   - Prédictions de tendance basées sur l'IA
   - Analyse de sentiment des médias sociaux
   - Corrélations entre actifs

### Configuration des stratégies

La section de configuration des stratégies permet de :

1. **Gérer les stratégies existantes** :
   - Activer/désactiver les stratégies
   - Modifier les paramètres
   - Dupliquer ou supprimer des stratégies

2. **Créer de nouvelles stratégies** :
   - À partir de zéro
   - À partir de modèles prédéfinis
   - En important depuis un fichier

Consultez la section [Configuration des stratégies de trading](#configuration-des-stratégies-de-trading) pour plus de détails.

## Bot Telegram

### Configuration du Bot Telegram

Le bot Telegram vous permet de contrôler le système à distance et de recevoir des alertes :

1. **Activation du bot** :
   - Dans l'interface web, accédez à "Paramètres" > "Notifications" > "Telegram"
   - Cliquez sur "Activer l'intégration Telegram"
   - Suivez les instructions pour connecter votre compte Telegram

2. **Sécurisation du bot** :
   - Définissez un code secret pour l'authentification
   - Limitez l'accès à votre ID Telegram uniquement

### Commandes disponibles

Voici les principales commandes disponibles via le bot Telegram :

```
/start - Démarrer le bot et afficher le message d'accueil
/auth [code_secret] - S'authentifier auprès du bot
/status - Afficher le statut du système (stratégies actives, performance)
/portfolio - Afficher l'état du portefeuille
/strategies - Lister les stratégies configurées
/enable [nom_stratégie] - Activer une stratégie
/disable [nom_stratégie] - Désactiver une stratégie
/trade [acheter/vendre] [symbole] [quantité] - Passer un ordre manuel
/performance [jour/semaine/mois/année] - Afficher les performances
/alerts - Gérer les alertes de prix
/subscribe [symbole] [condition] [valeur] - Créer une alerte
/help - Afficher l'aide et les commandes disponibles
```

### Alertes automatiques

Le bot Telegram peut envoyer automatiquement les alertes suivantes :

1. **Alertes de trading** :
   - Signaux d'achat/vente générés
   - Ordres exécutés (réussis ou échoués)
   - Seuils de profit/perte atteints

2. **Alertes de prix** :
   - Prix au-dessus/en-dessous d'un seuil défini
   - Variation de prix importante (%)
   - Franchissement de moyennes mobiles

3. **Alertes système** :
   - Problèmes de connexion aux API
   - Anomalies détectées (comportement inhabituel du marché)
   - Transactions suspectes

Pour configurer vos alertes :
1. Utilisez la commande `/alerts` pour voir les alertes actuelles
2. Utilisez `/subscribe` pour ajouter une nouvelle alerte
3. Ou configurez les alertes via l'interface web dans "Paramètres" > "Notifications"

## Configuration des stratégies de trading

### Création d'une stratégie

Pour créer une nouvelle stratégie :

1. Dans l'interface web, accédez à "Stratégies" > "Nouvelle stratégie"

2. **Informations de base** :
   - Nom de la stratégie
   - Description
   - Marché cible (crypto, forex, actions, etc.)
   - Paire/Symbole (ex: BTC/USDT, EUR/USD)
   - Intervalle de temps (1m, 5m, 15m, 1h, 4h, 1j)

3. **Sélection du type de stratégie** :
   - Basée sur des indicateurs techniques
   - Basée sur l'IA
   - Mixte (indicateurs + IA)
   - Arbitrage
   - Grid trading
   - Market making

4. **Configuration des règles** :
   - Conditions d'entrée en position
   - Conditions de sortie
   - Gestion des risques

### Paramétrage des indicateurs

Pour une stratégie basée sur des indicateurs techniques :

1. **Ajout d'indicateurs** :
   - Cliquez sur "Ajouter un indicateur"
   - Sélectionnez parmi les indicateurs disponibles (MA, EMA, RSI, MACD, Bollinger, etc.)
   - Configurez les paramètres (périodes, multiplicateurs, etc.)

2. **Création de conditions** :
   - Définissez les conditions d'achat/vente basées sur les indicateurs
   - Exemple : "Acheter quand MA(50) croise au-dessus de MA(200)"
   - Combinez plusieurs conditions avec des opérateurs logiques (ET, OU)

3. **Test backtest** :
   - Testez votre stratégie sur des données historiques
   - Analysez les performances et affinez les paramètres

Exemple de configuration d'une stratégie de croisement de moyennes mobiles :

```
Nom : Croisement_MA_BTC
Description : Stratégie basée sur le croisement des moyennes mobiles 50 et 200
Marché : Crypto
Symbole : BTC/USDT
Intervalle : 4h

Indicateurs :
- MA rapide : Période = 50, Type = EMA
- MA lente : Période = 200, Type = SMA

Règles d'entrée :
- ACHETER quand MA rapide croise au-dessus de MA lente

Règles de sortie :
- VENDRE quand MA rapide croise en-dessous de MA lente
- OU StopLoss = -2% du prix d'entrée
- OU TakeProfit = +5% du prix d'entrée

Gestion du risque :
- Taille de position = 10% du capital disponible
- Trailing stop = 1.5%
```

### Gestion du risque

Paramètres de gestion du risque disponibles :

1. **Taille de position** :
   - Montant fixe
   - Pourcentage du capital
   - Calculé selon la volatilité (Kelly, ATR)

2. **Stop Loss** :
   - Fixe (niveau de prix)
   - Relatif (% du prix d'entrée)
   - Basé sur ATR ou volatilité
   - Trailing stop (suit le prix)

3. **Take Profit** :
   - Fixe (niveau de prix)
   - Relatif (% du prix d'entrée)
   - Multiple cibles avec sorties partielles

4. **Filtres supplémentaires** :
   - Heure d'exécution (limiter aux heures spécifiques)
   - Conditions de marché (volatilité, volume)
   - Filtres de tendance

## Gestion des API de trading

### Ajout d'une nouvelle API

Pour ajouter une nouvelle API d'échange/courtier :

1. Dans l'interface web, accédez à "Paramètres" > "API"
2. Cliquez sur "Ajouter une nouvelle API"
3. Sélectionnez l'échange/courtier dans la liste
4. Saisissez vos identifiants API :
   - Clé API
   - Clé secrète
   - Phrase de passe (si nécessaire)
5. Définissez les permissions (lecture seule ou trading)
6. Testez la connexion avant d'enregistrer

Plateformes supportées :
- Binance, Coinbase Pro, Kraken, FTX, Bybit (crypto)
- Interactive Brokers, Alpaca (actions)
- OANDA, FXCM (forex)
- Autres via des plugins

### Sécurisation des clés API

Pour sécuriser vos clés API :

1. **Création des clés API** :
   - Créez des clés API dédiées pour le bot
   - Limitez les IP autorisées à se connecter
   - Accordez uniquement les permissions nécessaires

2. **Bonnes pratiques** :
   - Utilisez différentes clés API pour chaque plateforme
   - Créez des clés en lecture seule pour le monitoring
   - Activez 2FA sur vos comptes d'échange
   - Surveillez régulièrement l'activité des API

3. **Révocation d'urgence** :
   - En cas de suspicion de compromission, accédez à "Paramètres" > "API" > "Révoquer"
   - Ou révoque directement sur le site de l'échange

## Suivi des performances

### Métriques disponibles

Le système offre des métriques complètes pour évaluer les performances :

1. **Métriques de rendement** :
   - Rendement total (absolu et %)
   - Rendement quotidien/hebdomadaire/mensuel/annuel
   - Rendement annualisé
   - Rendement ajusté au risque (Sharpe, Sortino)

2. **Métriques de risque** :
   - Volatilité
   - Drawdown maximal
   - Value at Risk (VaR)
   - Ratio de gain/perte

3. **Métriques de transaction** :
   - Nombre total de transactions
   - Taux de réussite (% de transactions rentables)
   - Profit moyen par transaction
   - Durée moyenne de détention

### Rapports

Différents types de rapports sont disponibles :

1. **Rapport quotidien** :
   - Résumé des activités de trading
   - Transactions exécutées
   - Performance du jour

2. **Rapport de performance** :
   - Analyse détaillée des stratégies
   - Graphiques de performance
   - Comparaison avec des benchmarks

3. **Rapport fiscal** :
   - Résumé des transactions pour la déclaration fiscale
   - Calcul des plus/moins-values
   - Export au format CSV/PDF

Pour générer des rapports :
1. Accédez à "Rapports" dans l'interface web
2. Sélectionnez le type de rapport et la période
3. Cliquez sur "Générer" et attendez le traitement
4. Téléchargez ou partagez le rapport généré

## Maintenance

### Sauvegarde des données

Pour sauvegarder vos données :

1. **Sauvegarde automatique** :
   - Le système effectue des sauvegardes quotidiennes automatiques
   - Configurez la rétention des sauvegardes dans "Paramètres" > "Système" > "Sauvegardes"

2. **Sauvegarde manuelle** :
   - Accédez à "Paramètres" > "Système" > "Sauvegardes"
   - Cliquez sur "Créer une sauvegarde maintenant"
   - Téléchargez une copie locale pour plus de sécurité

3. **Restauration** :
   - Pour restaurer, accédez à "Paramètres" > "Système" > "Sauvegardes"
   - Sélectionnez la sauvegarde à restaurer
   - Cliquez sur "Restaurer" et confirmez

### Mise à jour du système

Pour mettre à jour le système :

1. **Mise à jour automatique** :
   - Le système vérifie automatiquement les mises à jour
   - Vous serez notifié quand une mise à jour est disponible

2. **Procédure de mise à jour** :
   - Sauvegardez vos données avant la mise à jour
   - Accédez à "Paramètres" > "Système" > "Mises à jour"
   - Cliquez sur "Installer la mise à jour"
   - Suivez les instructions à l'écran

3. **Mise à jour manuelle** :
   - Consultez la [documentation de déploiement](DEPLOYMENT.md) pour les instructions de mise à jour manuelle

## Dépannage

### Problèmes courants

Voici les solutions aux problèmes les plus fréquents :

1. **Problèmes de connexion API** :
   - Vérifiez que vos clés API sont correctes et actives
   - Vérifiez les restrictions d'IP sur l'échange
   - Vérifiez que l'échange n'est pas en maintenance

2. **Transactions non exécutées** :
   - Vérifiez le solde disponible
   - Vérifiez les limites de trading (min/max)
   - Vérifiez les journaux pour les messages d'erreur spécifiques

3. **Interface web inaccessible** :
   - Vérifiez que le serveur est en cours d'exécution
   - Vérifiez votre connexion réseau
   - Vérifiez les journaux du serveur

4. **Performance lente du système** :
   - Réduisez le nombre de stratégies actives simultanément
   - Augmentez les ressources serveur (RAM, CPU)
   - Optimisez la base de données (voir Documentation de Maintenance)

### Journaux d'erreurs

Pour consulter les journaux d'erreurs :

1. **Via l'interface web** :
   - Accédez à "Paramètres" > "Système" > "Journaux"
   - Filtrez par niveau (INFO, WARNING, ERROR, CRITICAL)
   - Filtrez par date ou module

2. **Via le système de fichiers** :
   - Les journaux sont stockés dans le répertoire `/logs`
   - Fichiers organisés par date et module
   - Utilisez `tail -f /logs/error.log` pour suivre les erreurs en temps réel

3. **Signalement d'un problème** :
   - Incluez toujours les journaux pertinents
   - Décrivez les étapes pour reproduire le problème
   - Mentionnez votre environnement (OS, version du système)

## Ressources supplémentaires

Pour approfondir vos connaissances :

1. **Documentation technique** :
   - [Architecture du système](ARCHITECTURE.md)
   - [Modèles d'IA](AI_MODELS.md)
   - [API publique](API.md)

2. **Tutoriels vidéo** :
   - [Chaîne YouTube EVIL2ROOT](https://youtube.com/evil2root)
   - Webinaires enregistrés
   - Tutoriels pas à pas

3. **Communauté et support** :
   - Forum d'entraide
   - Canal Discord
   - Support par email : support@evil2root.com

4. **Formation avancée** :
   - Cours sur la création de stratégies avancées
   - Certification d'expert EVIL2ROOT
   - Sessions de coaching individuelles 
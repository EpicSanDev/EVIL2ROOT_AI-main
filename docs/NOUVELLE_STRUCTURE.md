# Nouvelle Structure du Projet EVIL2ROOT Trading

Ce document explique la nouvelle structure du projet après la restructuration. Cette organisation vise à améliorer la lisibilité, la maintenabilité et le respect des bonnes pratiques de développement Python.

## Architecture Générale

```
EVIL2ROOT_AI-main/
│
├── src/                    # Code source principal
│   ├── core/               # Fonctionnalités principales et moteur de trading
│   │   ├── database/       # Gestion de la base de données
│   │   ├── market_data/    # Récupération et traitement des données de marché
│   │   ├── strategies/     # Stratégies de trading
│   │   └── risk_management/ # Gestion des risques et des positions
│   │
│   ├── models/             # Modèles d'IA et de prédiction
│   │   ├── price/          # Modèles de prédiction de prix
│   │   ├── sentiment/      # Analyse de sentiment
│   │   ├── rl/             # Apprentissage par renforcement
│   │   └── ensemble/       # Modèles d'ensemble et intégration
│   │
│   ├── services/           # Services divers (notifications, paiements)
│   ├── api/                # API et endpoints
│   ├── utils/              # Utilitaires communs
│   ├── traders/            # Implémentations spécifiques de trading
│   ├── validators/         # Validation des décisions de trading
│   ├── notification/       # Services de notification (Telegram, etc.)
│   └── ui/                 # Interface utilisateur
│       ├── templates/      # Templates HTML
│       ├── static/         # Fichiers statiques (CSS, JS, images)
│       └── controllers/    # Contrôleurs de l'interface
│
├── tests/                  # Tests unitaires et d'intégration
│   ├── unit/               # Tests unitaires
│   ├── integration/        # Tests d'intégration
│   └── e2e/                # Tests end-to-end
│
├── config/                 # Fichiers de configuration
│   └── environments/       # Configurations par environnement (.env.*)
│
├── scripts/                # Scripts utilitaires
│   ├── deployment/         # Scripts de déploiement
│   ├── monitoring/         # Scripts de surveillance
│   ├── development/        # Scripts de développement
│   └── shell/              # Scripts shell
│
├── docs/                   # Documentation
│   ├── api/                # Documentation API
│   ├── architecture/       # Documentation architecture
│   ├── user_guide/         # Guide utilisateur
│   └── development/        # Guide de développement
│
├── docker/                 # Fichiers Docker
│   ├── Dockerfile          # Dockerfile principal
│   └── docker-compose.yml  # Configuration docker-compose
│
├── kubernetes/             # Configuration Kubernetes
│   ├── base/               # Configurations de base
│   └── overlays/           # Overlays par environnement
│
├── web_app/                # Application web spécifique (si nécessaire)
│
├── data/                   # Données (ignorées par git)
│   ├── historical/         # Données historiques
│   ├── processed/          # Données traitées
│   └── models/             # Modèles entraînés sauvegardés
│
├── logs/                   # Logs (ignorés par git)
│
├── setup.py                # Configuration du package Python
├── requirements.txt        # Dépendances principales
├── requirements-dev.txt    # Dépendances de développement
├── install.sh              # Script d'installation
└── README.md               # Documentation principale
```

## Points Clés de la Restructuration

1. **Organisation modulaire**
   - Code organisé par fonctionnalité et type de composant
   - Séparation claire entre interfaces, logique métier et accès aux données

2. **Importations simplifiées**
   - Structure en package Python avec fichiers `__init__.py`
   - Importations plus claires et cohérentes

3. **Configuration centralisée**
   - Fichiers de configuration regroupés dans `/config`
   - Variables d'environnement dans `.env` avec exemples dans `.env.example`

4. **Tests organisés**
   - Tests séparés par niveau (unitaires, intégration, end-to-end)
   - Structure correspondant à celle du code source

5. **Déploiement standardisé**
   - Configuration Docker mise à jour
   - Support Kubernetes amélioré

6. **Documentation enrichie**
   - Documents structurés par thème
   - Meilleure accessibilité pour les nouveaux contributeurs

## Migration vers la Nouvelle Structure

La migration a été réalisée en conservant tous les fichiers originaux. Les étapes suivantes sont recommandées pour finaliser la transition :

1. Mettre à jour les imports dans les fichiers Python
2. Adapter les chemins de fichiers dans les scripts
3. Vérifier et adapter les configurations Docker
4. Mettre à jour les documentation avec les nouveaux chemins
5. Exécuter les tests pour vérifier que tout fonctionne correctement

## Prochaines Étapes

1. Mettre à jour les imports et références internes
2. Corriger les éventuels problèmes de chemins
3. Ajouter des tests pour les nouveaux composants
4. Améliorer la documentation API
5. Standardiser les logs à travers l'application 
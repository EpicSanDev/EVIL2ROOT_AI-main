# Guide de Configuration EVIL2ROOT

Ce document détaille les bonnes pratiques de configuration pour l'application EVIL2ROOT Trading Bot, en mettant l'accent sur la sécurité et la portabilité.

## Changements Récents

Deux problèmes majeurs ont été corrigés dans l'application :

1. **Variables d'environnement sans valeurs par défaut sécurisées**
2. **Inconsistances dans les chemins de fichiers**

## Variables d'Environnement et Sécurité

### Nouveau Système de Gestion des Variables d'Environnement

Un module utilitaire a été créé pour gérer les variables d'environnement de manière sécurisée :

- `src/utils/env_config.py` : Fournit des fonctions pour récupérer des variables d'environnement avec des valeurs par défaut sécurisées

### Principales Améliorations

- **Mode Production vs Développement** : Le système détecte automatiquement si l'application est en mode production ou développement basé sur la variable `FLASK_ENV`
- **Valeurs Requises en Production** : En production, certaines variables critiques sont requises et doivent être explicitement définies
- **Génération Sécurisée de Mots de Passe** : En développement, des mots de passe sécurisés sont générés aléatoirement si non définis
- **Valeurs Par Défaut Sécurisées** : Remplacement des valeurs par défaut hardcodées par des valeurs générées dynamiquement ou alternatives sécurisées

### Utilisation de l'Utilitaire

```python
from src.utils.env_config import get_env_var, get_db_params, get_redis_params

# Récupérer une variable d'environnement avec valeur par défaut
debug_mode = get_env_var('DEBUG', False)

# Récupérer des variables requises en production
api_key = get_env_var('API_KEY', required_in_production=True)

# Récupérer tous les paramètres de base de données
db_params = get_db_params()

# Récupérer tous les paramètres Redis
redis_params = get_redis_params()
```

## Gestion des Chemins de Fichiers

### Nouveau Système de Configuration des Logs

Un module utilitaire a été créé pour configurer les logs avec des chemins cohérents :

- `src/utils/log_config.py` : Fournit une fonction pour configurer les logs avec des chemins relatifs au répertoire du projet

### Principales Améliorations

- **Chemins Relatifs** : Utilisation de `pathlib.Path` pour créer des chemins relatifs au répertoire du projet
- **Création Automatique des Répertoires** : Les répertoires de logs sont créés automatiquement s'ils n'existent pas
- **Configuration Cohérente** : Configuration standardisée des logs dans tous les fichiers
- **Niveaux de Log Configurables** : Possibilité de définir le niveau de log via des variables d'environnement

### Utilisation de l'Utilitaire

```python
from src.utils.log_config import setup_logging

# Configurer un logger avec le nom du fichier par défaut (nom_du_logger.log)
logger = setup_logging('nom_du_logger')

# Configurer un logger avec un nom de fichier spécifique
logger = setup_logging('api', 'api_requests.log')

# Le logger peut être utilisé normalement
logger.info("Application démarrée")
logger.error("Une erreur s'est produite")
```

## Configuration pour la Production

Pour déployer l'application en production, assurez-vous de :

1. **Configurer explicitement les variables d'environnement critiques**
   - Utiliser un fichier `.env` en production avec toutes les variables requises
   - Ne jamais utiliser les valeurs par défaut du développement

2. **Définir `FLASK_ENV=production`**
   - Cela activera les vérifications strictes des variables requises

3. **Utiliser des mots de passe forts et uniques**
   - Générer des mots de passe forts pour la base de données, l'admin et autres services
   - Exemple : `openssl rand -hex 32` pour générer une clé secrète

4. **Surveiller les journaux d'application**
   - Les logs sont désormais stockés dans des chemins cohérents
   - Les erreurs liées aux variables manquantes seront clairement indiquées

## Fichier .env.example

Le fichier `.env.example` a été mis à jour pour indiquer clairement quelles variables sont requises en production, avec le préfixe `PRODUCTION REQUIRED` dans les commentaires. Utilisez ce fichier comme base pour créer votre propre fichier `.env` pour la production.

## Bonnes Pratiques de Sécurité

1. **Ne jamais stocker de secrets dans le code source**
   - Tous les secrets doivent être dans des variables d'environnement ou des services de secrets

2. **Utiliser des valeurs différentes entre développement et production**
   - Ne jamais réutiliser les mêmes secrets entre environnements

3. **Rotation régulière des secrets**
   - Changer régulièrement les clés API, mots de passe et autres secrets

4. **Privilèges minimaux**
   - Les utilisateurs de base de données et services ne devraient avoir que les permissions nécessaires

5. **Audit et journalisation**
   - Surveiller les accès aux ressources sensibles via les logs 
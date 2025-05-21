# Configuration Kubernetes Locale pour Evil2Root Trading AI

Ce guide explique comment configurer et utiliser un environnement Kubernetes local pour tester l'application Evil2Root Trading AI.

## Prérequis

- Docker Desktop installé
- Kubernetes activé dans Docker Desktop
- kubectl configuré pour utiliser le contexte Docker Desktop

## Installation

1. **Activer Kubernetes dans Docker Desktop**

   - Ouvrez Docker Desktop
   - Allez dans Paramètres (icône d'engrenage)
   - Sélectionnez l'onglet "Kubernetes"
   - Cochez "Enable Kubernetes"
   - Cliquez sur "Apply & Restart"
   - Attendez que Kubernetes démarre (le point vert s'affiche)

2. **Déployer l'environnement minimal**

   ```bash
   ./setup-minimal-k8s.sh
   ```

   Ce script va :
   - Vérifier que Docker Desktop et Kubernetes sont bien configurés
   - Créer un namespace pour votre application
   - Créer les secrets nécessaires
   - Configurer les variables d'environnement via ConfigMap
   - Déployer les services de base (PostgreSQL, Redis, Adminer)

## Utilisation

### Services déployés

L'environnement minimal déploie les services suivants :

- **PostgreSQL** : Base de données principale
- **Redis** : Cache et file de messages
- **Adminer** : Interface d'administration pour PostgreSQL

### Accès aux services

Utilisez le script `k8s-port-forward.sh` pour accéder aux services :

```bash
# Accès à PostgreSQL
./k8s-port-forward.sh postgres

# Accès à Redis
./k8s-port-forward.sh redis

# Accès à Adminer (interface web pour PostgreSQL)
./k8s-port-forward.sh adminer

# Accès à tous les services (ouvre plusieurs terminaux)
./k8s-port-forward.sh all
```

### Accès à Adminer

Adminer est accessible via http://localhost:8080 quand le port-forward est actif.

Informations de connexion :
- Système : PostgreSQL
- Serveur : postgres
- Utilisateur : postgres
- Mot de passe : postgres_password
- Base de données : trading_bot

### Gestion de l'environnement

Utilisez le script `k8s-control.sh` pour gérer l'environnement :

```bash
# Vérifier l'état des services
./k8s-control.sh status

# Afficher les logs d'un service
./k8s-control.sh logs

# Redémarrer tous les services
./k8s-control.sh restart

# Arrêter tous les services
./k8s-control.sh stop

# Démarrer tous les services
./k8s-control.sh start
```

## Connexion depuis votre application

Votre application peut se connecter aux services déployés via les ports en port-forward :

### PostgreSQL
- Hôte : localhost
- Port : 5432
- Utilisateur : postgres
- Mot de passe : postgres_password
- Base de données : trading_bot

### Redis
- Hôte : localhost
- Port : 6379

## Nettoyage

Pour supprimer l'environnement Kubernetes local :

```bash
kubectl --context docker-desktop delete namespace evil2root-trading
```

## Désactiver Kubernetes

Si vous souhaitez désactiver Kubernetes dans Docker Desktop :

1. Ouvrez Docker Desktop
2. Allez dans Paramètres (icône d'engrenage)
3. Sélectionnez l'onglet "Kubernetes"
4. Décochez "Enable Kubernetes"
5. Cliquez sur "Apply & Restart"

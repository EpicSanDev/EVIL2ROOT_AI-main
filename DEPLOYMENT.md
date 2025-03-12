# Déploiement sur DigitalOcean avec GitHub Actions

Ce document explique comment déployer notre application sur DigitalOcean à l'aide de GitHub Actions.

## Préparation

Assurez-vous d'avoir:

1. Un compte DigitalOcean
2. Un compte GitHub avec ce dépôt configuré
3. Un Personal Access Token (PAT) DigitalOcean avec les droits d'écriture

## Configuration initiale

### 1. Créer un Container Registry sur DigitalOcean

Suivez les instructions dans [docs/digitalocean-setup.md](docs/digitalocean-setup.md).

### 2. Configurer les secrets GitHub

Suivez les instructions dans [docs/github-secrets-setup.md](docs/github-secrets-setup.md).

## Processus de déploiement

### 1. Workflow GitHub Actions

Le workflow est défini dans `.github/workflows/docker-build-push.yml`. Ce workflow:

- Est déclenché à chaque push sur les branches main/master
- Peut également être déclenché manuellement
- Construit l'image Docker à l'aide de Docker Buildx
- Pousse l'image vers le Container Registry DigitalOcean
- Utilise la mise en cache pour accélérer les builds

### 2. Utilisation de l'image

Une fois l'image construite et poussée, vous pouvez l'utiliser sur:

- **App Platform**: Déployez l'application directement depuis le Container Registry
- **Droplets**: Tirez l'image et exécutez-la sur n'importe quel serveur DigitalOcean
- **Kubernetes**: Utilisez l'image dans un cluster DOKS (DigitalOcean Kubernetes)

### 3. Déploiement sur App Platform

Pour déployer sur App Platform:

1. Accédez à votre tableau de bord DigitalOcean
2. Sélectionnez "App Platform"
3. Cliquez sur "Create App"
4. Sélectionnez "Container Registry" comme source
5. Choisissez votre image
6. Configurez les paramètres (environnement, stockage, etc.)
7. Déployez l'application

### 4. Déploiement sur Droplet

Pour déployer sur un Droplet:

```bash
# Se connecter au Droplet
ssh root@your-droplet-ip

# Se connecter au Container Registry
doctl registry login

# Tirer l'image
docker pull registry.digitalocean.com/your-registry-name/evil2root-ai:latest

# Exécuter le conteneur
docker run -d -p 5000:5000 \
  -e SECRET_KEY=your_secure_key \
  -e DATABASE_URI=your_database_uri \
  --name evil2root-ai \
  registry.digitalocean.com/your-registry-name/evil2root-ai:latest
```

## Monitoring et santé

Cette application inclut:

- Un endpoint `/health` pour vérifier l'état de l'application
- Des healthchecks Docker configurés dans le Dockerfile
- Des métriques Prometheus exposées sur le port 9090

## Troubleshooting

Si vous rencontrez des problèmes avec le déploiement:

1. Vérifiez les logs de GitHub Actions
2. Assurez-vous que les secrets sont correctement configurés
3. Vérifiez que le Container Registry existe et est accessible
4. Consultez les logs du conteneur pour des messages d'erreur 
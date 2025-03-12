# Configuration des secrets GitHub

Pour que le workflow GitHub Actions fonctionne correctement, vous devez ajouter les secrets suivants à votre dépôt GitHub:

## Étapes pour ajouter des secrets

1. Accédez à votre dépôt GitHub
2. Cliquez sur "Settings" (onglet)
3. Dans le menu de gauche, cliquez sur "Secrets and variables" puis "Actions"
4. Cliquez sur "New repository secret" pour chaque secret à ajouter

## Secrets nécessaires

Ajoutez les secrets suivants:

1. **DIGITALOCEAN_ACCESS_TOKEN**
   - Valeur: Le token d'accès personnel généré sur DigitalOcean
   - Ce token permet d'authentifier GitHub Actions auprès de DigitalOcean

2. **DIGITALOCEAN_REGISTRY_NAME**
   - Valeur: Le nom de votre registry DigitalOcean (ex: "evil2root-registry")
   - Ce nom est utilisé pour construire l'URL complète de votre registry

## Vérification

Une fois les secrets configurés, vous pouvez déclencher manuellement le workflow pour vérifier qu'il fonctionne:

1. Accédez à l'onglet "Actions" de votre dépôt
2. Sélectionnez le workflow "Build and Push Docker Image to DigitalOcean"
3. Cliquez sur "Run workflow"
4. Sélectionnez la branche à utiliser (généralement "main" ou "master")
5. Cliquez sur "Run workflow" 
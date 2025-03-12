# Résoudre les erreurs d'authentification avec DigitalOcean

## Erreur "401 Unauthorized"

Si vous rencontrez cette erreur:
```
failed to authorize: failed to fetch oauth token: unexpected status from GET request to https://api.digitalocean.com/v2/registry/auth?scope=repository: 401 Unauthorized
```

Voici les étapes à suivre:

### 1. Vérifier le token d'accès

Le token d'accès personnel DigitalOcean peut être invalide ou avoir expiré.

1. Générez un nouveau token d'accès sur DigitalOcean:
   - Allez dans le dashboard DigitalOcean → API
   - Générez un nouveau token avec des permissions suffisantes (read/write)
   - Copiez le nouveau token

2. Mettez à jour le secret GitHub:
   - Accédez à votre dépôt GitHub → Settings → Secrets → Actions
   - Modifiez `DIGITALOCEAN_ACCESS_TOKEN` avec le nouveau token

### 2. Vérifier le nom du registre

Assurez-vous que le nom de votre registre est correctement configuré:

1. Vérifiez le nom exact dans le dashboard DigitalOcean → Container Registry
2. Mettez à jour le secret `DIGITALOCEAN_REGISTRY_NAME` dans GitHub avec la valeur exacte
   (dans votre cas, la valeur devrait être `epicsandev`)

### 3. Vérifier les permissions du registre

1. Assurez-vous que le token a les permissions nécessaires pour accéder au registre
2. Dans DigitalOcean, vérifiez les paramètres du registre pour confirmer qu'il est accessible

### 4. Vérifier la configuration doctl dans le workflow

Assurez-vous que l'étape d'authentification au registre est correcte:

```yaml
- name: Log in to DigitalOcean Container Registry
  run: doctl registry login --expiry-seconds 3600
```

### 5. Forcer une authentification avec informations complètes

Si les méthodes ci-dessus ne fonctionnent pas, essayez de vous connecter explicitement:

```yaml
- name: Log in to DigitalOcean Container Registry
  run: |
    echo "${{ secrets.DIGITALOCEAN_ACCESS_TOKEN }}" | docker login registry.digitalocean.com -u "${{ secrets.DIGITALOCEAN_ACCESS_TOKEN }}" --password-stdin
``` 
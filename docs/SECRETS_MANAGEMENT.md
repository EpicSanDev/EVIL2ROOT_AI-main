# Gestion des Secrets

Ce document explique comment gérer les secrets de manière sécurisée dans notre application de trading.

## Principe important

**NE JAMAIS stocker de secrets (mots de passe, clés API, tokens) en clair dans les fichiers versionnés** comme `.env` ou tout autre fichier du repository.

## Configuration des secrets

### Développement local

Pour le développement local, créez un fichier `.env.local` (qui est ignoré par Git) contenant vos secrets :

```bash
cp .env .env.local
```

Puis modifiez ce fichier pour y ajouter vos secrets.

### Environnement de production avec Docker

Pour l'environnement Docker, utilisez Docker Secrets :

```bash
./init-secrets.sh
```

Ce script lit les variables sensibles de votre fichier `.env.local` et génère des fichiers de secrets dans le répertoire `./secrets/`.

### Environnement Kubernetes

Pour Kubernetes, utilisez les Secrets Kubernetes :

1. Créez un fichier `secrets.yaml` :

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-bot-secrets
  namespace: evil2root-trading
type: Opaque
stringData:
  DB_USER: votre_utilisateur_db
  DB_PASSWORD: votre_mot_de_passe_db_securise
  SECRET_KEY: votre_cle_secrete_tres_longue_et_aleatoire
  ADMIN_PASSWORD: votre_mot_de_passe_admin_securise
  TELEGRAM_TOKEN: votre_token_telegram
  OPENROUTER_API_KEY: votre_cle_api_openrouter
  FINNHUB_API_KEY: votre_cle_api_finnhub
  COINBASE_API_KEY: votre_cle_api_coinbase
  COINBASE_WEBHOOK_SECRET: votre_secret_webhook_coinbase
```

2. Appliquez le fichier de secrets :

```bash
kubectl apply -f secrets.yaml
```

3. Supprimez le fichier `secrets.yaml` après application.

## Bonnes pratiques

1. Utilisez un gestionnaire de secrets comme HashiCorp Vault en production
2. Changez régulièrement les mots de passe et tokens
3. Utilisez des politiques de restriction des accès à vos secrets
4. Auditez régulièrement l'accès aux secrets
5. Utilisez des tokens avec privilèges limités quand c'est possible

## Rotation des secrets

Il est recommandé de faire une rotation régulière des secrets :

1. Générez de nouveaux tokens/clés API
2. Mettez à jour les secrets dans votre environnement
3. Vérifiez que tout fonctionne correctement
4. Révoquemz les anciens tokens/clés 
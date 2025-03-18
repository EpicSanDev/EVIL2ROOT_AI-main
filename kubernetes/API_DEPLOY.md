# Déploiement de l'API EVIL2ROOT Trading Bot sur Kubernetes

Ce document explique comment déployer l'API FastAPI du bot de trading EVIL2ROOT sur un cluster Kubernetes.

## Architecture

L'API est une application FastAPI qui fournit les endpoints pour interagir avec le bot de trading. Elle communique avec la base de données PostgreSQL et Redis pour la mise en cache et les files d'attente.

Architecture du déploiement :
- **Deployment**: Gère les pods de l'API
- **Service**: Expose l'API à l'intérieur du cluster
- **Ingress**: Expose l'API à l'extérieur du cluster avec HTTPS

## Prérequis

- Un cluster Kubernetes fonctionnel
- `kubectl` configuré pour se connecter au cluster
- Un registre d'images Docker (nous utilisons DigitalOcean Container Registry)
- Un contrôleur Ingress installé sur le cluster (nous recommandons NGINX Ingress Controller)
- Un émetteur de certificats SSL (comme cert-manager avec Let's Encrypt)

## Configuration DNS

Pour accéder à l'API depuis l'extérieur, vous devez configurer un enregistrement DNS :

```
api.trading.example.com   →   <ADRESSE_IP_DE_VOTRE_INGRESS>
```

Remplacez `example.com` par votre domaine réel.

## Fichiers de déploiement

- `api-deployment.yaml` : Contient la définition du Deployment, Service et Ingress
- `deploy-api.sh` : Script facilitant le déploiement

## Déploiement

### Méthode 1 : Script automatisé

Utilisez le script de déploiement fourni :

```bash
# Rendez le script exécutable si nécessaire
chmod +x kubernetes/deploy-api.sh

# Exécutez le script
./kubernetes/deploy-api.sh
```

Le script va :
1. Construire l'image Docker de l'API
2. Pousser l'image vers le registre
3. Déployer sur Kubernetes
4. Vérifier l'état du déploiement

### Méthode 2 : Déploiement manuel

Si vous préférez déployer manuellement :

1. Construisez l'image Docker :
   ```bash
   docker build -t registry.digitalocean.com/evil2root-registry/evil2root-api:latest -f Dockerfile.api .
   ```

2. Poussez l'image vers le registre :
   ```bash
   docker push registry.digitalocean.com/evil2root-registry/evil2root-api:latest
   ```

3. Déployez sur Kubernetes :
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   kubectl apply -f kubernetes/api-deployment.yaml
   ```

## Personnalisation

### Domaine

Modifiez le nom de domaine dans `api-deployment.yaml` :

```yaml
spec:
  tls:
  - hosts:
    - api.trading.example.com    # Remplacez par votre domaine
    secretName: api-tls
  rules:
  - host: api.trading.example.com    # Remplacez par votre domaine
```

### Ressources

Ajustez les ressources allouées aux pods dans `api-deployment.yaml` selon vos besoins :

```yaml
resources:
  requests:
    memory: "512Mi"   # Mémoire minimale
    cpu: "500m"       # CPU minimal (0.5 core)
  limits:
    memory: "1Gi"     # Limite de mémoire
    cpu: "1000m"      # Limite de CPU (1 core)
```

### Nombre de réplicas

Modifiez le nombre de pods dans `api-deployment.yaml` :

```yaml
spec:
  replicas: 3   # Nombre de pods souhaité
```

### Variables d'environnement

Vous pouvez ajouter ou modifier les variables d'environnement dans la section `env` du fichier `api-deployment.yaml`. Par exemple, pour activer le mode debug :

```yaml
- name: API_DEBUG
  value: "True"
```

## Communication avec la base de données et Redis

L'API est configurée pour communiquer avec :
- Base de données PostgreSQL via le service `postgres`
- Redis via le service `redis`

Assurez-vous que ces services sont correctement déployés et fonctionnels.

## CORS (Cross-Origin Resource Sharing)

L'API est configurée pour accepter les requêtes CORS depuis le frontend. Les paramètres sont configurés dans l'Ingress :

```yaml
# CORS settings
nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
nginx.ingress.kubernetes.io/cors-allow-headers: "DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization"
nginx.ingress.kubernetes.io/cors-allow-origin: "https://ui.trading.example.com"
nginx.ingress.kubernetes.io/cors-allow-credentials: "true"
```

Si vous modifiez l'URL du frontend, assurez-vous de mettre à jour la valeur de `cors-allow-origin`.

## Vérification du déploiement

Pour vérifier l'état du déploiement :

```bash
# Vérifier les pods
kubectl get pods -n evil2root-trading -l app=trading-bot,component=api

# Vérifier le service
kubectl get svc -n evil2root-trading -l app=trading-bot,component=api

# Vérifier l'ingress
kubectl get ingress -n evil2root-trading trading-bot-api-ingress
```

## Documentation de l'API

Une fois déployée, la documentation Swagger de l'API est accessible à :
- https://api.trading.example.com/docs

Vous pouvez utiliser cette interface pour tester les endpoints et comprendre la structure de l'API.

## Dépannage

### Problèmes courants

1. **Image non trouvée** : Vérifiez que l'image est correctement poussée vers le registre et que le secret d'accès au registre est configuré.

2. **Problèmes de certificat SSL** : Vérifiez que cert-manager est correctement installé et que l'émetteur de certificats fonctionne.

3. **Erreurs de connexion à la base de données** : Vérifiez que les services PostgreSQL et Redis sont accessibles et que les secrets sont correctement configurés.

### Logs

Pour consulter les logs des pods API :

```bash
kubectl logs -n evil2root-trading -l app=trading-bot,component=api
```

### Redémarrage

Pour redémarrer le déploiement :

```bash
kubectl rollout restart deployment/trading-bot-api -n evil2root-trading
```

## Mise à jour

Pour mettre à jour l'API vers une nouvelle version :

1. Mettez à jour le code source
2. Construisez une nouvelle image Docker avec un tag spécifique
3. Mettez à jour le tag d'image dans `api-deployment.yaml`
4. Appliquez les modifications avec `kubectl apply -f kubernetes/api-deployment.yaml`

Ou utilisez simplement le script `deploy-api.sh` qui gère tout le processus.

## Sécurité

Le déploiement inclut diverses mesures de sécurité :
- Exécution en tant qu'utilisateur non-root
- Restriction des capacités
- En-têtes de sécurité HTTP (CSP, XSS Protection, etc.)
- HTTPS obligatoire avec TLS 1.2+ uniquement
- Isolation réseau (via les NetworkPolicies existantes) 
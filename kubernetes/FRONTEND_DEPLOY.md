# Déploiement du Frontend EVIL2ROOT Trading Bot sur Kubernetes

Ce document explique comment déployer l'interface utilisateur (frontend) du bot de trading EVIL2ROOT sur un cluster Kubernetes.

## Architecture

Le frontend est une application React servie par Nginx. Elle communique avec l'API backend via des requêtes HTTP/REST.

Architecture du déploiement :
- **Deployment**: Gère les pods du frontend
- **Service**: Expose le frontend à l'intérieur du cluster
- **Ingress**: Expose le frontend à l'extérieur du cluster avec HTTPS

## Prérequis

- Un cluster Kubernetes fonctionnel
- `kubectl` configuré pour se connecter au cluster
- Un registre d'images Docker (nous utilisons DigitalOcean Container Registry)
- Un contrôleur Ingress installé sur le cluster (nous recommandons NGINX Ingress Controller)
- Un émetteur de certificats SSL (comme cert-manager avec Let's Encrypt)

## Configuration DNS

Pour accéder au frontend depuis l'extérieur, vous devez configurer un enregistrement DNS :

```
ui.trading.example.com   →   <ADRESSE_IP_DE_VOTRE_INGRESS>
```

Remplacez `example.com` par votre domaine réel.

## Fichiers de déploiement

- `frontend-deployment.yaml` : Contient la définition du Deployment, Service et Ingress
- `deploy-frontend.sh` : Script facilitant le déploiement

## Déploiement

### Méthode 1 : Script automatisé

Utilisez le script de déploiement fourni :

```bash
# Rendez le script exécutable si nécessaire
chmod +x kubernetes/deploy-frontend.sh

# Exécutez le script
./kubernetes/deploy-frontend.sh
```

Le script va :
1. Construire l'image Docker du frontend
2. Pousser l'image vers le registre
3. Déployer sur Kubernetes
4. Vérifier l'état du déploiement

### Méthode 2 : Déploiement manuel

Si vous préférez déployer manuellement :

1. Construisez l'image Docker :
   ```bash
   docker build -t registry.digitalocean.com/evil2root-registry/evil2root-frontend:latest -f frontend/Dockerfile frontend/
   ```

2. Poussez l'image vers le registre :
   ```bash
   docker push registry.digitalocean.com/evil2root-registry/evil2root-frontend:latest
   ```

3. Déployez sur Kubernetes :
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   kubectl apply -f kubernetes/frontend-deployment.yaml
   ```

## Personnalisation

### Domaine

Modifiez le nom de domaine dans `frontend-deployment.yaml` :

```yaml
spec:
  tls:
  - hosts:
    - ui.trading.example.com    # Remplacez par votre domaine
    secretName: frontend-tls
  rules:
  - host: ui.trading.example.com    # Remplacez par votre domaine
```

### Ressources

Ajustez les ressources allouées aux pods dans `frontend-deployment.yaml` selon vos besoins :

```yaml
resources:
  requests:
    memory: "128Mi"   # Mémoire minimale
    cpu: "100m"       # CPU minimal (0.1 core)
  limits:
    memory: "256Mi"   # Limite de mémoire
    cpu: "200m"       # Limite de CPU (0.2 core)
```

### Nombre de réplicas

Modifiez le nombre de pods dans `frontend-deployment.yaml` :

```yaml
spec:
  replicas: 2   # Nombre de pods souhaité
```

## Communication avec le Backend

Le frontend est configuré pour communiquer avec l'API backend via le service `trading-bot-web`. 

Dans la configuration Nginx du frontend (`frontend/nginx/default.conf`), les requêtes `/api/` sont redirigées vers `http://api:8000/api/`. Dans Kubernetes, cette redirection doit être gérée par le frontend via le service approprié.

## Vérification du déploiement

Pour vérifier l'état du déploiement :

```bash
# Vérifier les pods
kubectl get pods -n evil2root-trading -l app=trading-bot,component=frontend

# Vérifier le service
kubectl get svc -n evil2root-trading -l app=trading-bot,component=frontend

# Vérifier l'ingress
kubectl get ingress -n evil2root-trading trading-bot-frontend-ingress
```

## Dépannage

### Problèmes courants

1. **Image non trouvée** : Vérifiez que l'image est correctement poussée vers le registre et que le secret d'accès au registre est configuré.

2. **Problèmes de certificat SSL** : Vérifiez que cert-manager est correctement installé et que l'émetteur de certificats fonctionne.

3. **Erreurs 502/504** : Vérifiez la communication entre le frontend et le backend, ainsi que les timeouts configurés dans l'Ingress.

### Logs

Pour consulter les logs des pods frontend :

```bash
kubectl logs -n evil2root-trading -l app=trading-bot,component=frontend
```

### Redémarrage

Pour redémarrer le déploiement :

```bash
kubectl rollout restart deployment/trading-bot-frontend -n evil2root-trading
```

## Mise à jour

Pour mettre à jour le frontend vers une nouvelle version :

1. Construisez une nouvelle image Docker avec un tag spécifique
2. Mettez à jour le tag d'image dans `frontend-deployment.yaml`
3. Appliquez les modifications avec `kubectl apply -f kubernetes/frontend-deployment.yaml`

Ou utilisez simplement le script `deploy-frontend.sh` qui gère tout le processus.

## Sécurité

Le déploiement inclut diverses mesures de sécurité :
- Exécution en tant qu'utilisateur non-root
- Restriction des capacités
- En-têtes de sécurité HTTP (CSP, XSS Protection, etc.)
- HTTPS obligatoire avec TLS 1.2+ uniquement 
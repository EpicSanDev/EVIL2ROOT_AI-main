# Déploiement EVIL2ROOT Trading Bot sur Kubernetes

Ce répertoire contient tous les fichiers nécessaires pour déployer le Trading Bot sur un cluster Kubernetes.

## Prérequis

- Un cluster Kubernetes fonctionnel (version 1.19+)
- `kubectl` installé et configuré pour accéder à votre cluster
- `kustomize` installé (facultatif, mais recommandé)
- Un contrôleur Ingress (comme NGINX Ingress Controller) installé sur le cluster
- cert-manager (optionnel, pour les certificats SSL automatiques)
- Accès au registre Docker contenant l'image evil2root/trading-bot:latest

### Support GPU (optionnel)

Pour utiliser les fonctionnalités d'entraînement accéléré par GPU, vous devez:
- Avoir des nœuds avec GPU dans votre cluster
- Installer l'opérateur NVIDIA sur votre cluster:
  ```bash
  kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
  ```

## Structure des fichiers

- `namespace.yaml` - Définit le namespace du projet
- `configmap.yaml` - Contient les configurations partagées
- `secrets.yaml` - Définit les secrets (à personnaliser avant déploiement)
- `storage.yaml` - Crée les PersistentVolumeClaims pour le stockage persistant
- `db-deployment.yaml` - Déploie PostgreSQL
- `redis-deployment.yaml` - Déploie Redis
- `web-deployment.yaml` - Déploie l'interface web principale
- `analysis-bot-deployment.yaml` - Déploie le bot d'analyse
- `market-scheduler-deployment.yaml` - Déploie le planificateur du marché
- `train-analyze-job.yaml` - Déploie le job d'entraînement et d'analyse
- `monitoring-deployment.yaml` - Déploie Prometheus et Grafana
- `adminer-deployment.yaml` - Déploie Adminer pour la gestion de la base de données
- `kustomization.yaml` - Configuration Kustomize pour un déploiement simple
- `deploy.sh` - Script de déploiement automatisé

## Procédure de déploiement rapide

1. Personnalisez `secrets.yaml` avec vos propres mots de passe (ne laissez pas les valeurs par défaut en production!)
2. Modifiez les noms d'hôtes dans les fichiers d'Ingress pour qu'ils correspondent à votre domaine
3. Exécutez le script de déploiement:
   ```bash
   ./deploy.sh
   ```

## Déploiement manuel

Si vous préférez déployer manuellement, suivez ces étapes:

1. Créez le namespace:
   ```bash
   kubectl apply -f namespace.yaml
   ```

2. Créez les secrets et configmaps:
   ```bash
   kubectl apply -f configmap.yaml -f secrets.yaml
   ```

3. Créez les volumes persistants:
   ```bash
   kubectl apply -f storage.yaml
   ```

4. Déployez la base de données et Redis:
   ```bash
   kubectl apply -f db-deployment.yaml -f redis-deployment.yaml
   ```
   
5. Attendez que la base de données et Redis soient prêts:
   ```bash
   kubectl -n evil2root-trading wait --for=condition=available --timeout=300s deployment/postgres deployment/redis
   ```

6. Déployez les autres composants:
   ```bash
   kubectl apply -f web-deployment.yaml -f analysis-bot-deployment.yaml -f market-scheduler-deployment.yaml -f monitoring-deployment.yaml -f adminer-deployment.yaml
   ```

7. Créez le job d'entraînement (uniquement si nécessaire):
   ```bash
   kubectl apply -f train-analyze-job.yaml
   ```

## Vérification du déploiement

Pour vérifier que tout fonctionne correctement:

```bash
kubectl -n evil2root-trading get pods
kubectl -n evil2root-trading get svc
kubectl -n evil2root-trading get ingress
```

## Accès aux services

Une fois déployé, accédez aux services via les URLs suivants:

- Interface Web: https://trading.example.com (à remplacer par votre domaine)
- Grafana: https://grafana.trading.example.com
- Adminer: https://adminer.trading.example.com

## Personnalisation avancée

### Ajustement des ressources

Les ressources (CPU/mémoire) sont configurées avec des valeurs par défaut dans chaque fichier de déploiement. Vous pouvez les ajuster en fonction de votre environnement.

### Mise à l'échelle

Pour augmenter le nombre de replicas de l'interface web:

```bash
kubectl -n evil2root-trading scale deployment/trading-bot-web --replicas=3
```

### Stockage

Le déploiement utilise par défaut des PersistentVolumeClaims avec la classe de stockage "standard". Modifiez `storage.yaml` pour utiliser une autre classe de stockage disponible dans votre cluster.

## Dépannage

### Vérification des journaux

```bash
kubectl -n evil2root-trading logs deployment/trading-bot-web
kubectl -n evil2root-trading logs deployment/analysis-bot
kubectl -n evil2root-trading logs deployment/market-scheduler
```

### Redémarrage d'un pod

```bash
kubectl -n evil2root-trading delete pod <nom-du-pod>
```

### Problèmes de persistance

Si vous rencontrez des problèmes de persistance, vérifiez l'état des PVCs:

```bash
kubectl -n evil2root-trading get pvc
```

## Mises à jour et maintenance

### Mise à jour de l'image

Pour mettre à jour l'image du trading bot:

```bash
kubectl -n evil2root-trading set image deployment/trading-bot-web web=evil2root/trading-bot:nouvelle-version
kubectl -n evil2root-trading set image deployment/analysis-bot analysis-bot=evil2root/trading-bot:nouvelle-version
kubectl -n evil2root-trading set image deployment/market-scheduler market-scheduler=evil2root/trading-bot:nouvelle-version
```

### Sauvegarde de la base de données

Pour sauvegarder la base de données:

```bash
kubectl -n evil2root-trading exec -it deployment/postgres -- pg_dump -U postgres tradingbot > backup.sql
```
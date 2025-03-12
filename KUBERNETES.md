# Déploiement du Trading Bot sur Kubernetes

Ce document fournit des instructions détaillées pour déployer EVIL2ROOT Trading Bot sur un cluster Kubernetes.

## Table des matières

- [Introduction](#introduction)
- [Avantages de Kubernetes](#avantages-de-kubernetes)
- [Architecture du déploiement Kubernetes](#architecture-du-déploiement-kubernetes)
- [Prérequis](#prérequis)
- [Instructions de déploiement](#instructions-de-déploiement)
- [Déploiement automatisé sur DigitalOcean](#déploiement-automatisé-sur-digitalocean)
- [Configuration avancée](#configuration-avancée)
- [Surveillance et maintenance](#surveillance-et-maintenance)
- [Dépannage](#dépannage)
- [Meilleures pratiques](#meilleures-pratiques)

## Introduction

Le Trading Bot EVIL2ROOT peut maintenant être déployé sur Kubernetes, ce qui offre de nombreux avantages en termes de scalabilité, de fiabilité et de facilité de gestion. Ce document explique comment configurer et déployer l'application sur un cluster Kubernetes.

## Avantages de Kubernetes

Le déploiement sur Kubernetes offre plusieurs avantages par rapport au déploiement traditionnel avec Docker Compose:

- **Haute disponibilité**: Kubernetes gère automatiquement les redémarrages en cas de défaillance
- **Scalabilité dynamique**: Possibilité d'augmenter ou de diminuer le nombre de replicas en fonction de la charge
- **Mises à jour sans interruption**: Les déploiements progressifs permettent des mises à jour sans temps d'arrêt
- **Auto-guérison**: Kubernetes remplace automatiquement les conteneurs défaillants
- **Orchestration de ressources**: Gestion optimisée des ressources CPU et mémoire
- **Isolation des environnements**: Séparation claire entre les environnements de développement, de test et de production

## Architecture du déploiement Kubernetes

Le déploiement Kubernetes du Trading Bot est structuré comme suit:

```
evil2root-trading namespace
|
├── Deployments
|   ├── trading-bot-web (2 replicas)
|   ├── analysis-bot (1 replica)
|   ├── market-scheduler (1 replica)
|   ├── postgres (1 replica)
|   ├── redis (1 replica)
|   ├── prometheus (1 replica)
|   ├── grafana (1 replica)
|   └── adminer (1 replica)
|
├── Services
|   ├── trading-bot-web
|   ├── postgres
|   ├── redis
|   ├── prometheus
|   ├── grafana
|   └── adminer
|
├── Jobs/CronJobs
|   ├── train-and-analyze (job à la demande)
|   └── scheduled-train-analyze (cronjob quotidien)
|
├── Ingress
|   ├── trading-bot-web-ingress
|   ├── grafana-ingress
|   └── adminer-ingress
|
├── PersistentVolumeClaims
|   ├── postgres-data
|   ├── redis-data
|   ├── app-data
|   ├── app-logs
|   ├── saved-models
|   ├── prometheus-data
|   └── grafana-data
|
├── ConfigMaps
|   ├── trading-bot-config
|   ├── prometheus-config
|   ├── grafana-provisioning
|   └── grafana-dashboards
|
└── Secrets
    └── trading-bot-secrets
```

## Prérequis

Pour déployer le Trading Bot sur Kubernetes, vous aurez besoin de:

1. Un cluster Kubernetes fonctionnel (version 1.19+)
2. `kubectl` installé et configuré pour accéder à votre cluster
3. `kustomize` installé (recommandé)
4. Un contrôleur Ingress installé sur le cluster (par exemple, NGINX Ingress Controller)
5. cert-manager pour gérer les certificats SSL (optionnel mais recommandé)
6. Pour les fonctionnalités GPU: support NVIDIA sur les nœuds et NVIDIA Device Plugin installé

### Options de cluster Kubernetes

Vous pouvez déployer le Trading Bot sur différentes plateformes Kubernetes:

- **Solutions managées**:
  - Google Kubernetes Engine (GKE)
  - Amazon Elastic Kubernetes Service (EKS)
  - Azure Kubernetes Service (AKS)
  - DigitalOcean Kubernetes

- **Installation on-premise**:
  - k3s (pour les clusters légers)
  - kubeadm (pour les installations personnalisées)
  - Rancher (pour la gestion multi-cluster)

## Instructions de déploiement

### 1. Préparation

Clonez le dépôt et accédez au répertoire Kubernetes:

```bash
git clone https://github.com/votre-compte/EVIL2ROOT_AI.git
cd EVIL2ROOT_AI/kubernetes
```

### 2. Configuration

Personnalisez les fichiers de configuration selon vos besoins:

- Modifiez `secrets.yaml` pour définir des mots de passe sécurisés
- Adaptez les ressources (CPU/mémoire) dans les fichiers de déploiement selon les capacités de votre cluster
- Modifiez les noms d'hôte dans les fichiers Ingress pour qu'ils correspondent à votre domaine

### 3. Déploiement automatisé

```bash
chmod +x deploy.sh
./deploy.sh
```

Le script vérifie les prérequis, crée les ressources nécessaires et attend que les déploiements soient prêts.

### 4. Vérification

Vérifiez que tous les pods sont en état "Running":

```bash
kubectl -n evil2root-trading get pods
```

Vérifiez que les services sont disponibles:

```bash
kubectl -n evil2root-trading get svc
```

Vérifiez les Ingress:

```bash
kubectl -n evil2root-trading get ingress
```

### 5. Accès à l'application

Accédez aux services via les URLs configurés:

- Interface web: https://trading.example.com (à adapter à votre domaine)
- Grafana: https://grafana.trading.example.com
- Adminer: https://adminer.trading.example.com

## Déploiement automatisé sur DigitalOcean

Pour simplifier davantage le déploiement, nous fournissons un script automatisé qui crée un cluster Kubernetes sur DigitalOcean et y déploie l'application en une seule commande.

### Prérequis pour le déploiement DigitalOcean

- Un compte DigitalOcean
- Un token API DigitalOcean
- Les outils `doctl`, `kubectl`, `curl` et `jq` installés

### Déploiement en une commande

```bash
cd kubernetes
chmod +x deploy_to_digitalocean.sh
./deploy_to_digitalocean.sh -t votre_token_api_digitalocean
```

### Options de personnalisation

Le script offre de nombreuses options pour personnaliser votre déploiement:

```
Options:
  -h, --help              Affiche l'aide
  -t, --token TOKEN       Token API DigitalOcean (obligatoire)
  -n, --name NAME         Nom du cluster (défaut: evil2root-trading)
  -r, --region REGION     Région DigitalOcean (défaut: fra1)
  -s, --size SIZE         Taille des nœuds (défaut: s-4vcpu-8gb)
  -c, --count COUNT       Nombre de nœuds (défaut: 3)
  -g, --gpu               Ajouter un nœud GPU (si disponible)
  -d, --domain DOMAIN     Domaine personnalisé pour les ingress
```

### Exemples

```bash
# Déploiement dans la région de New York, avec des nœuds plus puissants
./deploy_to_digitalocean.sh --token votre_token --region nyc1 --size s-8vcpu-16gb --count 2

# Déploiement avec un nœud GPU et un domaine personnalisé
./deploy_to_digitalocean.sh -t votre_token -g -d votredomaine.com
```

Pour plus de détails sur le déploiement DigitalOcean, consultez le fichier [kubernetes/DIGITALOCEAN.md](kubernetes/DIGITALOCEAN.md).

## Configuration avancée

### Utilisation de GPU

Les jobs d'entraînement sont configurés pour utiliser des GPU NVIDIA. Pour adapter cette configuration:

1. Si vous n'avez pas de GPU, modifiez les fichiers `train-analyze-job.yaml`:
   - Supprimez les lignes `nvidia.com/gpu` dans les sections `resources`
   - Ajustez les ressources CPU et mémoire en conséquence

2. Si vous avez un type de GPU différent (AMD, etc.), modifiez les références de `nvidia.com/gpu` pour correspondre à votre type de GPU.

### Configuration de l'autoscaling

Pour configurer l'autoscaling horizontal pour l'interface web:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trading-bot-web-hpa
  namespace: evil2root-trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trading-bot-web
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Configuration de stockage avancée

Pour utiliser des classes de stockage spécifiques (par exemple, pour utiliser des disques SSD sur cloud):

1. Modifiez le fichier `storage.yaml` en remplaçant `storageClassName: standard` par la classe de stockage appropriée pour votre environnement.

2. Pour les clusters sur cloud publics:
   - GKE: `storageClassName: premium-rwo`
   - EKS: `storageClassName: gp2`
   - AKS: `storageClassName: managed-premium`

### Configuration de la haute disponibilité

Pour améliorer la haute disponibilité:

1. Augmentez le nombre de replicas pour les composants critiques:
   ```bash
   kubectl -n evil2root-trading scale deployment/trading-bot-web --replicas=3
   ```

2. Utilisez des nodeSelectors ou affinités pour répartir les pods sur différents nœuds:
   ```yaml
   spec:
     affinity:
       podAntiAffinity:
         preferredDuringSchedulingIgnoredDuringExecution:
         - weight: 100
           podAffinityTerm:
             labelSelector:
               matchExpressions:
               - key: app
                 operator: In
                 values:
                 - trading-bot
             topologyKey: kubernetes.io/hostname
   ```

## Surveillance et maintenance

### Surveillance

Le déploiement inclut Prometheus et Grafana pour la surveillance:

1. Accédez à Grafana via https://grafana.trading.example.com
2. Connectez-vous avec les identifiants configurés
3. Explorez les tableaux de bord prédéfinis pour surveiller les performances

### Mise à jour de l'application

Pour mettre à jour l'application vers une nouvelle version:

```bash
# Mettre à jour l'image pour tous les déploiements
kubectl -n evil2root-trading set image deployment/trading-bot-web web=evil2root/trading-bot:nouvelle-version
kubectl -n evil2root-trading set image deployment/analysis-bot analysis-bot=evil2root/trading-bot:nouvelle-version
kubectl -n evil2root-trading set image deployment/market-scheduler market-scheduler=evil2root/trading-bot:nouvelle-version
```

### Sauvegarde et restauration

#### Sauvegarde de la base de données

```bash
# Créer une sauvegarde
kubectl -n evil2root-trading exec -it $(kubectl -n evil2root-trading get pod -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- pg_dump -U postgres tradingbot > backup.sql

# Restaurer une sauvegarde
cat backup.sql | kubectl -n evil2root-trading exec -i $(kubectl -n evil2root-trading get pod -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- psql -U postgres tradingbot
```

#### Sauvegarde des volumes

Pour les environnements cloud, utilisez les fonctionnalités de snapshot de votre fournisseur:

- GKE: Snapshots de disque persistant
- EKS: Snapshots EBS
- AKS: Snapshots de disque managé

## Dépannage

### Problèmes courants et solutions

1. **Les pods restent en état "Pending"**:
   - Vérifiez les ressources disponibles dans le cluster
   - Vérifiez si les PVCs sont en attente de liaisons

2. **Les pods crashent au démarrage**:
   - Vérifiez les logs: `kubectl -n evil2root-trading logs <pod-name>`
   - Vérifiez les événements: `kubectl -n evil2root-trading describe pod <pod-name>`

3. **Problèmes d'accès aux services**:
   - Vérifiez la configuration Ingress
   - Vérifiez que le DNS est correctement configuré
   - Vérifiez l'état du contrôleur Ingress

### Récupération après un désastre

1. Assurez-vous d'avoir des sauvegardes régulières de la base de données et des volumes
2. Documentez toutes les modifications non versionné de configuration
3. Testez régulièrement les procédures de restauration

## Meilleures pratiques

1. **Sécurité**:
   - Utilisez des secrets Kubernetes pour toutes les informations sensibles
   - Limitez les privilèges des conteneurs
   - Mettez régulièrement à jour les images pour corriger les vulnérabilités

2. **Persistance des données**:
   - Utilisez des PVCs pour toutes les données qui doivent persister
   - Mettez en place des sauvegardes régulières
   - Vérifiez périodiquement l'intégrité des données

3. **Surveillance**:
   - Configurez des alertes pour les métriques critiques
   - Surveillez l'utilisation des ressources pour optimiser les demandes/limites
   - Gardez un œil sur les logs pour détecter les problèmes émergents

4. **Mise à l'échelle**:
   - Utilisez HPA pour adapter automatiquement les ressources
   - Planifiez la capacité en fonction des modèles d'utilisation
   - Testez la scalabilité avec des charges réalistes

---

Pour toute question ou assistance sur le déploiement Kubernetes, veuillez ouvrir une issue sur le dépôt GitHub du projet. 
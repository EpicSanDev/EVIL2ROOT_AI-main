# Configuration Kubernetes pour EVIL2ROOT Trading Bot

Ce répertoire contient l'ensemble des fichiers de configuration Kubernetes pour déployer le Trading Bot EVIL2ROOT sur un cluster Kubernetes.

## Améliorations récentes apportées

La configuration Kubernetes a été mise à jour avec les améliorations suivantes:

1. **Sécurité renforcée**:
   - Contextes de sécurité pour tous les conteneurs (`securityContext`)
   - Configuration `runAsNonRoot` et `readOnlyRootFilesystem`
   - Suppression des capacités superflues
   - Prévention de l'escalade de privilèges
   - En-têtes de sécurité HTTP améliorés pour les Ingress
   - Isolation réseau via les NetworkPolicies

2. **Haute disponibilité**:
   - PodDisruptionBudgets configurés pour tous les services critiques
   - Stratégies de déploiement RollingUpdate avec zero downtime
   - Affinités de nœuds et topologie pour une meilleure répartition
   - Sondes de healthcheck (liveness, readiness, startup) améliorées

3. **Autoscaling intelligent**:
   - Métriques personnalisées pour le scaling basé sur les charges réelles
   - Comportements de scaling améliorés avec périodes de stabilisation
   - Politiques de montée et descente en charge plus précises

4. **Optimisation de ressources**:
   - Limites et requêtes ajustées pour optimiser les performances
   - Volume temporaire pour les écritures en lecture seule
   - Répartition sur les nœuds worker via nodeAffinity

5. **Déploiement amélioré**:
   - Script de déploiement progressif et sécurisé
   - Génération de checksums pour la détection des changements de configuration
   - Prise en compte des timestamps pour forcer les redéploiements si nécessaire

6. **Interface utilisateur moderne**:
   - Frontend React déployé séparément pour une meilleure scalabilité
   - Livraison optimisée via Nginx avec compression et cache
   - Protection HTTPS avec headers de sécurité renforcés

7. **Architecture microservices**:
   - API FastAPI séparée du backend pour une meilleure maintenabilité
   - Communication inter-services optimisée
   - Gestion CORS sécurisée pour l'interaction frontend-API

## Structure des fichiers

- `namespace.yaml` - Définition du namespace dédié
- `configmap.yaml` - Configuration partagée
- `secrets.yaml` - Informations sensibles (encodées en base64)
- `storage.yaml` - Volumes persistants
- `db-deployment.yaml` - Déploiement PostgreSQL
- `redis-deployment.yaml` - Déploiement Redis
- `web-deployment.yaml` - API Backend du trading bot
- `frontend-deployment.yaml` - Interface utilisateur React
- `api-deployment.yaml` - API FastAPI pour l'interface utilisateur
- `analysis-bot-deployment.yaml` - Service d'analyse en continu
- `market-scheduler-deployment.yaml` - Planificateur d'ordres de marché
- `train-analyze-job.yaml` - Jobs d'entraînement et d'analyse
- `monitoring-deployment.yaml` - Prometheus et Grafana
- `hpa.yaml` - Configuration d'autoscaling
- `network-policies.yaml` - Sécurité réseau
- `pod-disruption-budgets.yaml` - Garanties de disponibilité
- `database-backup.yaml` - Sauvegarde automatique de la base de données
- `kustomization.yaml` - Configuration Kustomize
- `deploy.sh` - Script de déploiement amélioré

## Prérequis

- Kubernetes 1.20+
- kubectl 1.20+
- Un cluster avec les ressources suivantes:
  - Au moins 3 nœuds worker
  - Au moins 16GB de RAM disponible
  - Au moins 8 vCPUs disponibles
  - Au moins 100GB de stockage persistant
  - (Optionnel) Support GPU pour les tâches d'entraînement

## Déploiement

Pour déployer l'application, exécutez le script de déploiement amélioré:

```bash
cd kubernetes
chmod +x deploy.sh
./deploy.sh
```

Le script vérifiera les prérequis, préparera les fichiers, appliquera les ressources dans le bon ordre et attendra que tous les services soient disponibles.

## Personnalisation

Vous pouvez personnaliser le déploiement en modifiant les fichiers de configuration YAML avant d'exécuter le script de déploiement.

### Pour modifier les ressources (CPU/mémoire)

Modifiez les sections `resources` dans les fichiers de déploiement:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "1000m"
```

### Pour modifier le nombre de réplicas

Modifiez la propriété `replicas` dans les fichiers de déploiement, ou utilisez:

```bash
kubectl -n evil2root-trading scale deployment/trading-bot-web --replicas=3
```

### Pour les environnements sans GPU

Dans `train-analyze-job.yaml`, supprimez ou commentez les sections relatives aux GPU.

## Surveillance

Une fois déployé, vous pouvez accéder à Grafana pour surveiller les performances du système:

- URL: https://grafana.trading.example.com
- Identifiant par défaut: admin
- Mot de passe: voir les secrets ou récupérer avec la commande:
  ```bash
  kubectl -n evil2root-trading get secret trading-bot-secrets -o jsonpath='{.data.GRAFANA_ADMIN_PASSWORD}' | base64 --decode
  ```

## Dépannage

Si vous rencontrez des problèmes lors du déploiement:

1. Vérifiez les logs des pods:
   ```bash
   kubectl -n evil2root-trading logs -f deployment/trading-bot-web
   ```

2. Vérifiez le statut des pods:
   ```bash
   kubectl -n evil2root-trading get pods
   ```

3. Détails sur un pod spécifique:
   ```bash
   kubectl -n evil2root-trading describe pod pod-name
   ```

Pour des informations plus détaillées, consultez le fichier KUBERNETES.md à la racine du projet.
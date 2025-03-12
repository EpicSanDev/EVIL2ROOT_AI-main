# Déploiement automatisé sur DigitalOcean Kubernetes

Ce document explique comment déployer automatiquement EVIL2ROOT Trading Bot sur DigitalOcean Kubernetes (DOKS) en utilisant le script `deploy_to_digitalocean.sh`.

## Prérequis

Avant de commencer, assurez-vous d'avoir :

1. Un compte DigitalOcean (créez-en un sur [digitalocean.com](https://digitalocean.com) si vous n'en avez pas)
2. Un token API DigitalOcean (généré depuis le panneau de contrôle DigitalOcean)
3. Les outils suivants installés sur votre machine locale :
   - `kubectl` - [Guide d'installation](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
   - `doctl` - [Guide d'installation](https://docs.digitalocean.com/reference/doctl/how-to/install/)
   - `curl`
   - `jq`

## Génération d'un token API DigitalOcean

1. Connectez-vous à votre compte DigitalOcean
2. Allez dans API → Tokens/Keys
3. Cliquez sur "Generate New Token"
4. Donnez un nom au token (par exemple, "EVIL2ROOT Trading Bot Deployment")
5. Sélectionnez les permissions d'écriture (Write)
6. Cliquez sur "Generate Token"
7. Copiez le token généré et conservez-le en lieu sûr (vous en aurez besoin pour le déploiement)

## Options du script

Le script `deploy_to_digitalocean.sh` offre plusieurs options pour personnaliser votre déploiement :

```
Options:
  -h, --help              Affiche cette aide
  -t, --token TOKEN       Token API DigitalOcean (obligatoire)
  -n, --name NAME         Nom du cluster (défaut: evil2root-trading)
  -r, --region REGION     Région DigitalOcean (défaut: fra1)
  -s, --size SIZE         Taille des nœuds (défaut: s-4vcpu-8gb)
  -c, --count COUNT       Nombre de nœuds (défaut: 3)
  -g, --gpu               Ajouter un nœud GPU (si disponible)
  -d, --domain DOMAIN     Domaine personnalisé pour les ingress
```

## Exemples de déploiement

### Déploiement de base

Pour un déploiement avec les paramètres par défaut, exécutez :

```bash
cd kubernetes
./deploy_to_digitalocean.sh -t votre_token_api_digitalocean
```

### Déploiement personnalisé

Vous pouvez personnaliser votre déploiement selon vos besoins :

```bash
# Déploiement dans la région de New York, avec des nœuds plus puissants
./deploy_to_digitalocean.sh --token votre_token --region nyc1 --size s-8vcpu-16gb --count 2

# Déploiement avec un nœud GPU et un domaine personnalisé
./deploy_to_digitalocean.sh -t votre_token -g -d votredomaine.com
```

## Régions DigitalOcean disponibles

Les régions couramment utilisées sont :

- `nyc1`, `nyc3` : New York, USA
- `sfo3` : San Francisco, USA
- `ams3` : Amsterdam, Pays-Bas
- `sgp1` : Singapour
- `lon1` : Londres, Royaume-Uni
- `fra1` : Francfort, Allemagne
- `tor1` : Toronto, Canada
- `blr1` : Bangalore, Inde

Pour obtenir la liste complète des régions disponibles, utilisez la commande :

```bash
doctl compute region list
```

## Tailles de nœuds disponibles

Les tailles de nœuds couramment utilisées sont :

- `s-2vcpu-4gb` : 2 vCPU, 4GB RAM (entrée de gamme)
- `s-4vcpu-8gb` : 4 vCPU, 8GB RAM (recommandé pour la plupart des cas)
- `s-8vcpu-16gb` : 8 vCPU, 16GB RAM (performances élevées)
- `g-8vcpu-32gb` : 8 vCPU, 32GB RAM avec GPU (pour l'entraînement de modèles)

Pour obtenir la liste complète des tailles disponibles, utilisez la commande :

```bash
doctl compute size list
```

## Utilisation d'un domaine personnalisé

Si vous possédez un domaine, vous pouvez l'utiliser pour accéder à vos services :

1. Exécutez le script avec l'option `-d` ou `--domain` :
   ```bash
   ./deploy_to_digitalocean.sh -t votre_token -d votredomaine.com
   ```

2. Après le déploiement, le script vous indiquera l'adresse IP à configurer dans vos DNS :
   ```
   Pour utiliser vos domaines, créez des enregistrements DNS A pointant vers: X.X.X.X
   - trading.votredomaine.com -> X.X.X.X
   - grafana.trading.votredomaine.com -> X.X.X.X
   - adminer.trading.votredomaine.com -> X.X.X.X
   ```

3. Ajoutez ces enregistrements DNS via votre fournisseur de nom de domaine

## Accès à l'interface utilisateur

Une fois le déploiement terminé, vous pourrez accéder à :

- Interface Web : `https://trading.votredomaine.com` (ou via l'IP si pas de domaine)
- Grafana : `https://grafana.trading.votredomaine.com`
- Adminer : `https://adminer.trading.votredomaine.com`

Les certificats SSL seront générés automatiquement via Let's Encrypt.

## Gestion du cluster

### Afficher les informations du cluster

```bash
doctl kubernetes cluster get evil2root-trading
```

### Afficher les nœuds du cluster

```bash
kubectl get nodes
```

### Mettre à l'échelle le cluster

Pour modifier le nombre de nœuds :

```bash
# Obtenez l'ID du cluster
CLUSTER_ID=$(doctl kubernetes cluster list --format ID --no-header --name "evil2root-trading")

# Obtenez l'ID du pool de nœuds
POOL_ID=$(doctl kubernetes cluster node-pool list $CLUSTER_ID --format ID --no-header)

# Modifiez le nombre de nœuds (par exemple, pour passer à 5 nœuds)
doctl kubernetes cluster node-pool update $CLUSTER_ID $POOL_ID --count 5
```

### Supprimer le cluster

Si vous n'avez plus besoin du cluster, vous pouvez le supprimer pour éviter des coûts inutiles :

```bash
doctl kubernetes cluster delete evil2root-trading
```

## Dépannage

### Le script échoue à trouver une région ou une taille de nœud

Utilisez `doctl compute region list` et `doctl compute size list` pour voir les options disponibles, puis spécifiez une région et une taille valides.

### Les nœuds GPU ne sont pas créés

Les nœuds GPU ne sont pas disponibles dans toutes les régions. Essayez une autre région comme `nyc1` ou `sfo3`.

### Les services ne sont pas accessibles via l'Ingress

1. Vérifiez que l'Ingress est correctement déployé : `kubectl get ingress -n evil2root-trading`
2. Vérifiez que le Load Balancer est correctement configuré : `kubectl get svc -n ingress-nginx`
3. Vérifiez vos DNS si vous utilisez un domaine personnalisé

### Les certificats SSL ne sont pas générés

1. Vérifiez l'état de cert-manager : `kubectl get pods -n cert-manager`
2. Vérifiez les émetteurs de certificats : `kubectl get clusterissuers`
3. Vérifiez les certificats : `kubectl get certificates -n evil2root-trading`

## Estimation des coûts

Avec la configuration par défaut (3 nœuds `s-4vcpu-8gb`), le coût mensuel estimé est d'environ 144$ (48$ par nœud).

Avec un nœud GPU supplémentaire, le coût total peut augmenter d'environ 500$ à 1000$ par mois selon le type de GPU.

Pour une estimation précise, consultez la [calculatrice de prix DigitalOcean](https://www.digitalocean.com/pricing/calculator). 
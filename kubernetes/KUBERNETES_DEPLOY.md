# Déploiement sur Kubernetes avec GitHub Actions

Ce document explique comment configurer et utiliser le déploiement automatisé sur Kubernetes via GitHub Actions.

## Prérequis

Pour que le déploiement fonctionne correctement, vous devez configurer les secrets GitHub suivants:

1. `DIGITALOCEAN_ACCESS_TOKEN` - Token d'accès DigitalOcean avec les autorisations pour créer/gérer des clusters Kubernetes
2. `DB_USER` - Nom d'utilisateur de la base de données
3. `DB_PASSWORD` - Mot de passe de la base de données
4. `DB_NAME` - Nom de la base de données
5. `REDIS_PASSWORD` - Mot de passe Redis
6. `GRAFANA_ADMIN_USER` - Nom d'utilisateur administrateur Grafana
7. `GRAFANA_ADMIN_PASSWORD` - Mot de passe administrateur Grafana

## Fonctionnement du déploiement

Le processus de déploiement se déroule comme suit:

1. Lorsque vous poussez des modifications sur la branche `main`, le workflow GitHub Action `Build and Deploy to Kubernetes` est déclenché.
2. Ce workflow:
   - Construit l'image Docker de l'application
   - La pousse vers le registre DigitalOcean
   - Vérifie si un cluster Kubernetes existe déjà, sinon en crée un nouveau
   - Déploie l'application et toutes ses dépendances sur le cluster
   - Expose l'application via un service LoadBalancer
   - Récupère l'IP externe attribuée et la sauvegarde dans un fichier `ACCESS.md`

## Accès à l'application

Une fois le déploiement terminé, vous pourrez accéder à l'application directement via l'IP externe, sans avoir besoin d'un nom de domaine. L'IP sera:
- Affichée dans les logs de l'action GitHub
- Sauvegardée dans le fichier `ACCESS.md` à la racine du projet

Exemple d'accès:
- Application web: `http://<EXTERNAL_IP>`
- API: `http://<EXTERNAL_IP>/api`
- Métriques: `http://<EXTERNAL_IP>/metrics`

## Mise à jour des secrets

Pour mettre à jour les secrets de l'application:

1. Mettez à jour les secrets dans les paramètres GitHub du projet
2. Déclenchez manuellement le workflow `Update Kubernetes Secrets` depuis l'interface GitHub

## Dépannage

Si vous rencontrez des problèmes lors du déploiement:

1. Vérifiez les logs des GitHub Actions pour identifier les erreurs
2. Assurez-vous que tous les secrets nécessaires sont correctement configurés
3. Vérifiez l'état du cluster avec les commandes suivantes:
   ```bash
   # Obtenir les informations d'identification du cluster
   doctl kubernetes cluster kubeconfig save evil2root-trading
   
   # Vérifier l'état des pods
   kubectl get pods -n evil2root-trading
   
   # Vérifier l'état des services
   kubectl get services -n evil2root-trading
   
   # Vérifier les logs d'un pod spécifique
   kubectl logs -n evil2root-trading <nom-du-pod>
   ```

4. Si nécessaire, vous pouvez supprimer le cluster et relancer le déploiement:
   ```bash
   doctl kubernetes cluster delete evil2root-trading
   ```

## Personnalisation

Pour modifier la configuration du cluster (taille, région, nombre de nœuds), modifiez le fichier `.github/workflows/deploy-to-kubernetes.yml`. 
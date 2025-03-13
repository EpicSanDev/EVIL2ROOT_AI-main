#!/bin/bash

set -e

# Couleurs pour les messages
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
RESET="\033[0m"

# Répertoire du projet
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_DIR="${PROJECT_DIR}/kubernetes"

# Fonction pour afficher les messages
log() {
  local type=$1
  local message=$2
  
  case $type in
    "info")
      echo -e "${GREEN}[INFO]${RESET} $message"
      ;;
    "warn")
      echo -e "${YELLOW}[WARN]${RESET} $message"
      ;;
    "error")
      echo -e "${RED}[ERROR]${RESET} $message"
      ;;
    "azure")
      echo -e "${BLUE}[AZURE]${RESET} $message"
      ;;
  esac
}

# Vérification des prérequis
check_prerequisites() {
  log "info" "Vérification des prérequis..."
  
  # Vérifier kubectl
  if ! command -v kubectl &> /dev/null; then
    log "error" "kubectl n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
  fi
  
  # Vérifier kustomize
  if ! command -v kustomize &> /dev/null; then
    log "warn" "kustomize n'est pas installé. Installation..."
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
    sudo mv kustomize /usr/local/bin/
  fi
  
  # Vérifier az CLI si on utilise Azure
  if [[ "$USE_AZURE" == "true" ]] && ! command -v az &> /dev/null; then
    log "warn" "Azure CLI n'est pas installé. Installation..."
    
    # Détection du système d'exploitation
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    
    case "$OS" in
      darwin)
        log "info" "Installation d'Azure CLI via Homebrew..."
        brew update && brew install azure-cli
        ;;
      linux)
        log "info" "Installation d'Azure CLI pour Linux..."
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
        ;;
      *)
        log "error" "Système d'exploitation non pris en charge pour l'installation automatique d'Azure CLI. Veuillez l'installer manuellement."
        exit 1
        ;;
    esac
  fi
  
  log "info" "Prérequis vérifiés avec succès."
}

# Configuration d'Azure
configure_azure() {
  log "azure" "Configuration de l'accès à Azure..."

  # Vérifier si déjà connecté
  if az account show &> /dev/null; then
    CURRENT_ACCOUNT=$(az account show --query name -o tsv)
    log "azure" "Déjà connecté à Azure avec le compte: $CURRENT_ACCOUNT"
    
    # Demander si l'utilisateur veut changer de compte
    read -p "Voulez-vous utiliser un autre compte Azure? (o/n): " CHANGE_ACCOUNT
    if [[ "$CHANGE_ACCOUNT" == "o" ]]; then
      az logout
      az login --use-device-code
    fi
  else
    # Se connecter à Azure
    log "azure" "Connexion à Azure..."
    az login --use-device-code
    
    if [ $? -ne 0 ]; then
      log "error" "Échec de l'authentification avec Azure."
      exit 1
    fi
    
    log "azure" "Authentification réussie avec Azure."
  fi
  
  # Sélectionner une souscription si l'utilisateur en a plusieurs
  SUBSCRIPTION_COUNT=$(az account list --query "length([])" -o tsv)
  if [ "$SUBSCRIPTION_COUNT" -gt 1 ]; then
    log "azure" "Plusieurs souscriptions disponibles. Veuillez en choisir une:"
    az account list --output table
    
    read -p "Entrez l'ID de la souscription à utiliser: " SUBSCRIPTION_ID
    az account set --subscription "$SUBSCRIPTION_ID"
    
    log "azure" "Souscription définie à: $SUBSCRIPTION_ID"
  fi
}

# Création d'un cluster Kubernetes sur Azure
create_aks_cluster() {
  log "azure" "Création d'un nouveau cluster AKS sur Azure..."
  
  # Demander les informations du groupe de ressources
  read -p "Nom du groupe de ressources (ex: evil2root-rg): " RESOURCE_GROUP
  read -p "Région (ex: westeurope, eastus): " LOCATION
  
  # Vérifier si le groupe de ressources existe
  if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    log "azure" "Création du groupe de ressources $RESOURCE_GROUP dans la région $LOCATION..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
  else
    log "azure" "Le groupe de ressources $RESOURCE_GROUP existe déjà."
  fi
  
  # Demander les informations du cluster
  read -p "Nom du cluster AKS (ex: trading-cluster): " CLUSTER_NAME
  read -p "Taille des nœuds (ex: Standard_DS2_v2): " NODE_SIZE
  read -p "Nombre de nœuds: " NODE_COUNT
  read -p "Version de Kubernetes (laissez vide pour la version par défaut): " K8S_VERSION
  
  # Créer le cluster AKS
  log "azure" "Création du cluster $CLUSTER_NAME dans le groupe de ressources $RESOURCE_GROUP..."
  
  VERSION_PARAM=""
  if [[ -n "$K8S_VERSION" ]]; then
    VERSION_PARAM="--kubernetes-version $K8S_VERSION"
  fi
  
  az aks create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CLUSTER_NAME" \
    --node-count "$NODE_COUNT" \
    --node-vm-size "$NODE_SIZE" \
    --enable-managed-identity \
    --generate-ssh-keys \
    $VERSION_PARAM
  
  if [ $? -ne 0 ]; then
    log "error" "Échec de la création du cluster AKS."
    exit 1
  fi
  
  # Configuration de kubectl avec le nouveau cluster
  log "azure" "Configuration de kubectl pour utiliser le nouveau cluster..."
  az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME"
  
  log "azure" "Cluster AKS créé avec succès. Attendez quelques minutes que tous les nœuds soient prêts."
  
  # Vérifier l'état des nœuds
  kubectl get nodes
}

# Connexion à un cluster existant
connect_aks_cluster() {
  log "azure" "Connexion à un cluster AKS existant sur Azure..."
  
  # Lister les groupes de ressources disponibles
  log "azure" "Groupes de ressources disponibles:"
  az group list --output table
  
  # Demander le groupe de ressources
  read -p "Entrez le nom du groupe de ressources contenant le cluster: " RESOURCE_GROUP
  
  # Lister les clusters AKS dans ce groupe de ressources
  log "azure" "Clusters AKS disponibles dans ce groupe de ressources:"
  az aks list --resource-group "$RESOURCE_GROUP" --output table
  
  # Demander le nom du cluster
  read -p "Entrez le nom du cluster AKS à utiliser: " CLUSTER_NAME
  
  # Configuration de kubectl avec le cluster existant
  log "azure" "Configuration de kubectl pour utiliser le cluster $CLUSTER_NAME..."
  az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME"
  
  if [ $? -ne 0 ]; then
    log "error" "Échec de la configuration de kubectl avec le cluster AKS."
    exit 1
  fi
  
  log "azure" "Connexion au cluster réussie."
  
  # Vérifier l'état des nœuds
  kubectl get nodes
}

# Installer cert-manager pour les certificats TLS
install_cert_manager() {
  log "azure" "Installation de cert-manager pour la gestion des certificats TLS..."
  
  # Vérifier si cert-manager est déjà installé
  if kubectl get namespace cert-manager &> /dev/null; then
    log "azure" "cert-manager est déjà installé."
    return
  fi
  
  # Vérifier si Helm est installé
  if ! command -v helm &> /dev/null; then
    log "warn" "Helm n'est pas installé. Installation..."
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    chmod 700 get_helm.sh
    ./get_helm.sh
    rm get_helm.sh
  fi
  
  # Ajouter le dépôt Helm Jetstack
  helm repo add jetstack https://charts.jetstack.io
  helm repo update
  
  # Installer cert-manager avec les CRDs
  kubectl create namespace cert-manager
  helm install cert-manager jetstack/cert-manager \
    --namespace cert-manager \
    --version v1.12.0 \
    --set installCRDs=true \
    --set prometheus.enabled=false
  
  # Attendre que cert-manager soit prêt
  kubectl -n cert-manager wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager --timeout=120s
  
  # Créer l'émetteur de certificats LetsEncrypt
  cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: $(read -p "Entrez votre adresse e-mail pour Let's Encrypt: " EMAIL && echo $EMAIL)
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
  
  log "azure" "cert-manager a été installé avec succès."
}

# Installer Nginx Ingress Controller
install_nginx_ingress() {
  log "azure" "Installation du contrôleur Nginx Ingress..."
  
  # Vérifier si nginx ingress est déjà installé
  if kubectl get namespace ingress-nginx &> /dev/null; then
    log "azure" "Nginx Ingress est déjà installé."
    return
  fi
  
  # Ajouter le dépôt Helm pour Nginx Ingress
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
  helm repo update
  
  # Installer Nginx Ingress Controller
  kubectl create namespace ingress-nginx
  helm install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-dns-label-name"="evil2root-trading" \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-health-probe-request-path"="/healthz"
  
  # Attendre que l'ingress controller soit prêt
  kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=120s
  
  # Obtenir l'IP externe de l'ingress controller
  log "azure" "En attente de l'attribution d'une IP externe au contrôleur d'ingress..."
  sleep 30  # Attendre que le load balancer soit provisionné
  
  INGRESS_IP=$(kubectl -n ingress-nginx get service ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  INGRESS_HOST=$(kubectl -n ingress-nginx get service ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
  
  if [[ -n "$INGRESS_IP" ]]; then
    log "azure" "Contrôleur d'ingress installé avec succès. IP externe: $INGRESS_IP"
  elif [[ -n "$INGRESS_HOST" ]]; then
    log "azure" "Contrôleur d'ingress installé avec succès. Hostname externe: $INGRESS_HOST"
  fi
  
  log "azure" "Veuillez configurer vos enregistrements DNS pour pointer vers cette IP ou ce hostname."
}

# Fonction pour configurer le cluster avec les prérequis
setup_aks_cluster() {
  log "azure" "Configuration du cluster AKS avec les prérequis..."
  
  # Installer Nginx Ingress Controller
  install_nginx_ingress
  
  # Installer cert-manager
  install_cert_manager
  
  # Installer Metrics Server pour HPA (inclus par défaut dans AKS)
  log "azure" "Vérification que Metrics Server est activé..."
  if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
    log "warn" "Metrics Server ne semble pas être disponible dans votre cluster AKS."
    log "warn" "Cela est inhabituel car il est normalement déployé par défaut."
    log "warn" "Vérifiez la configuration de votre cluster AKS."
  else
    log "azure" "Metrics Server est correctement installé."
  fi
  
  log "azure" "Configuration de base du cluster terminée avec succès."
}

# Fonction pour déployer sur Kubernetes
deploy() {
  log "info" "Début du déploiement sur Kubernetes..."
  
  # Vérifier que le contexte est bien configuré
  CONTEXT=$(kubectl config current-context)
  log "info" "Contexte Kubernetes actuel: $CONTEXT"
  read -p "Êtes-vous sûr de vouloir déployer sur ce contexte? (o/n): " CONFIRM
  if [[ "$CONFIRM" != "o" ]]; then
    log "info" "Déploiement annulé par l'utilisateur."
    exit 0
  fi
  
  # Création du namespace s'il n'existe pas
  if ! kubectl get namespace evil2root-trading &> /dev/null; then
    log "info" "Création du namespace evil2root-trading..."
    kubectl apply -f "${K8S_DIR}/namespace.yaml"
  fi
  
  # Vérifier et mettre à jour les secrets pour production
  if [[ "$ENV" == "production" ]]; then
    log "info" "Configuration pour l'environnement de production..."
    
    # Créer un fichier temporaire de secrets
    TMP_SECRETS_FILE=$(mktemp)
    
    # Extraire le modèle de secrets
    kubectl -n evil2root-trading create secret generic trading-bot-secrets --dry-run=client -o yaml > "$TMP_SECRETS_FILE"
    
    # Demander de nouveaux mots de passe
    read -sp "Mot de passe pour la base de données PostgreSQL: " DB_PASSWORD
    echo
    read -sp "Mot de passe pour Redis: " REDIS_PASSWORD
    echo
    read -sp "Mot de passe admin pour Grafana: " GRAFANA_PASSWORD
    echo
    
    # Mettre à jour le fichier temporaire
    yq eval ".stringData.DB_PASSWORD = \"$DB_PASSWORD\"" -i "$TMP_SECRETS_FILE"
    yq eval ".stringData.REDIS_PASSWORD = \"$REDIS_PASSWORD\"" -i "$TMP_SECRETS_FILE"
    yq eval ".stringData.GRAFANA_ADMIN_PASSWORD = \"$GRAFANA_PASSWORD\"" -i "$TMP_SECRETS_FILE"
    
    # Appliquer les secrets
    kubectl apply -f "$TMP_SECRETS_FILE"
    
    # Supprimer le fichier temporaire
    rm "$TMP_SECRETS_FILE"
    
    log "info" "Secrets mis à jour pour la production."
  fi
  
  # Déploiement avec kustomize
  log "info" "Déploiement des ressources avec kustomize..."
  kubectl apply -k "${K8S_DIR}"
  
  log "info" "Déploiement terminé avec succès."
  
  # Attendre que tous les pods soient prêts
  log "info" "Attente que tous les pods soient prêts..."
  kubectl wait --for=condition=ready pod --all -n evil2root-trading --timeout=300s
  
  # Vérifier l'état des services
  log "info" "Services disponibles:"
  kubectl get services -n evil2root-trading
  
  # Vérifier l'état des ingress
  log "info" "Règles d'ingress configurées:"
  kubectl get ingress -n evil2root-trading
  
  if [[ "$USE_AZURE" == "true" ]]; then
    log "azure" "Déploiement terminé. Assurez-vous de configurer vos DNS pour pointer vers l'IP du load balancer."
    INGRESS_IP=$(kubectl -n ingress-nginx get service ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    log "azure" "IP du load balancer Nginx: $INGRESS_IP"
  fi
}

# Fonction pour mettre à jour l'image du trading bot
update_image() {
  VERSION=$1
  if [[ -z "$VERSION" ]]; then
    log "error" "Version non spécifiée. Utilisation: ./k8s-deploy-azure.sh update_image <version>"
    exit 1
  fi
  
  log "info" "Mise à jour de l'image vers evil2root/trading-bot:$VERSION..."
  
  # Mise à jour de l'image pour les déploiements
  kubectl set image deployment/trading-bot-web web=evil2root/trading-bot:$VERSION -n evil2root-trading
  kubectl set image deployment/analysis-bot analysis-bot=evil2root/trading-bot:$VERSION -n evil2root-trading
  kubectl set image deployment/market-scheduler market-scheduler=evil2root/trading-bot:$VERSION -n evil2root-trading
  
  log "info" "Mise à jour des images terminée. Surveillance du rollout..."
  
  # Surveillance du rollout
  kubectl rollout status deployment/trading-bot-web -n evil2root-trading
  kubectl rollout status deployment/analysis-bot -n evil2root-trading
  kubectl rollout status deployment/market-scheduler -n evil2root-trading
  
  log "info" "Mise à jour des images terminée avec succès."
}

# Fonction pour afficher les pods en cours d'exécution
show_pods() {
  log "info" "Affichage des pods en cours d'exécution dans le namespace evil2root-trading..."
  kubectl get pods -n evil2root-trading -o wide
}

# Fonction pour afficher les logs d'un pod
show_logs() {
  POD_NAME=$1
  if [[ -z "$POD_NAME" ]]; then
    log "error" "Nom du pod non spécifié. Utilisation: ./k8s-deploy-azure.sh logs <nom_pod>"
    
    # Afficher la liste des pods disponibles
    log "info" "Pods disponibles:"
    kubectl get pods -n evil2root-trading -o name | sed 's|pod/||'
    exit 1
  fi
  
  # Vérifier si le pod existe
  if ! kubectl get pod "$POD_NAME" -n evil2root-trading &> /dev/null; then
    log "error" "Le pod $POD_NAME n'existe pas dans le namespace evil2root-trading."
    
    # Suggérer des pods existants
    log "info" "Pods disponibles:"
    kubectl get pods -n evil2root-trading -o name | sed 's|pod/||'
    exit 1
  fi
  
  log "info" "Affichage des logs pour le pod $POD_NAME..."
  kubectl logs -f "$POD_NAME" -n evil2root-trading
}

# Fonction pour restaurer une sauvegarde de la base de données
restore_backup() {
  BACKUP_NAME=$1
  if [[ -z "$BACKUP_NAME" ]]; then
    log "error" "Nom de la sauvegarde non spécifié. Utilisation: ./k8s-deploy-azure.sh restore_backup <nom_fichier_backup>"
    
    # Afficher les sauvegardes disponibles
    log "info" "Lancement d'un pod temporaire pour lister les sauvegardes disponibles..."
    kubectl run -it --rm temp-pod --restart=Never --image=postgres:13 -n evil2root-trading -- \
      bash -c "ls -lh /backups" --overrides='{"spec":{"volumes":[{"name":"backup-volume","persistentVolumeClaim":{"claimName":"postgres-backups"}}],"containers":[{"name":"temp-pod","volumeMounts":[{"name":"backup-volume","mountPath":"/backups"}]}]}}'
    exit 1
  fi
  
  log "info" "Restauration de la sauvegarde $BACKUP_NAME en cours..."
  
  # Créer un job pour restaurer la sauvegarde
  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: postgres-restore
  namespace: evil2root-trading
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      containers:
      - name: postgres-restore
        image: postgres:13
        command:
        - /bin/bash
        - -c
        - |
          set -e
          echo "Décompression de la sauvegarde $BACKUP_NAME..."
          gunzip -c /backups/$BACKUP_NAME > /tmp/backup.sql
          echo "Restauration de la base de données..."
          PGPASSWORD=\$POSTGRES_PASSWORD psql -h postgres -U \$POSTGRES_USER -d \$POSTGRES_DB -f /tmp/backup.sql
          echo "Restauration terminée avec succès."
        env:
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: DB_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: DB_PASSWORD
        - name: POSTGRES_DB
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: DB_NAME
        volumeMounts:
        - name: backup-volume
          mountPath: /backups
      volumes:
      - name: backup-volume
        persistentVolumeClaim:
          claimName: postgres-backups
      restartPolicy: Never
EOF
  
  log "info" "Job de restauration créé. Surveillance du job..."
  kubectl wait --for=condition=complete job/postgres-restore --timeout=300s -n evil2root-trading
  
  log "info" "Restauration terminée. Logs du job:"
  kubectl logs job/postgres-restore -n evil2root-trading
}

# Fonction pour configurer Azure Storage pour les backups
setup_azure_storage() {
  log "azure" "Configuration d'Azure Storage pour les backups..."
  
  # Demander les informations du groupe de ressources
  CLUSTER_CONTEXT=$(kubectl config current-context)
  log "azure" "Contexte Kubernetes actuel: $CLUSTER_CONTEXT"
  
  read -p "Nom du groupe de ressources (ex: evil2root-rg): " RESOURCE_GROUP
  read -p "Nom du compte de stockage (doit être unique et en minuscules): " STORAGE_ACCOUNT
  read -p "Nom de la ressource de partage de fichiers: " FILE_SHARE
  
  # Vérifier si le compte de stockage existe
  if ! az storage account show --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    log "azure" "Création du compte de stockage $STORAGE_ACCOUNT..."
    az storage account create \
      --resource-group "$RESOURCE_GROUP" \
      --name "$STORAGE_ACCOUNT" \
      --sku Standard_LRS \
      --kind StorageV2
  else
    log "azure" "Le compte de stockage $STORAGE_ACCOUNT existe déjà."
  fi
  
  # Récupérer la clé du compte de stockage
  STORAGE_KEY=$(az storage account keys list --resource-group "$RESOURCE_GROUP" --account-name "$STORAGE_ACCOUNT" --query "[0].value" -o tsv)
  
  # Vérifier si le partage de fichiers existe
  if ! az storage share exists --name "$FILE_SHARE" --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" | grep -q "true"; then
    log "azure" "Création du partage de fichiers $FILE_SHARE..."
    az storage share create \
      --name "$FILE_SHARE" \
      --account-name "$STORAGE_ACCOUNT" \
      --account-key "$STORAGE_KEY" \
      --quota 100
  else
    log "azure" "Le partage de fichiers $FILE_SHARE existe déjà."
  fi
  
  # Créer un secret Kubernetes pour le compte de stockage
  kubectl create secret generic azure-storage-secret \
    --from-literal=azurestorageaccountname="$STORAGE_ACCOUNT" \
    --from-literal=azurestorageaccountkey="$STORAGE_KEY" \
    -n evil2root-trading
  
  # Créer un PV et PVC pour le partage de fichiers
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: postgres-backups-pv
  labels:
    usage: postgres-backups
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  azureFile:
    secretName: azure-storage-secret
    shareName: $FILE_SHARE
    readOnly: false
  mountOptions:
  - dir_mode=0777
  - file_mode=0777
  - uid=1000
  - gid=1000
  - mfsymlinks
  - nobrl
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-backups
  namespace: evil2root-trading
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  selector:
    matchLabels:
      usage: postgres-backups
EOF
  
  log "azure" "Configuration d'Azure Storage terminée avec succès."
}

# Fonction pour obtenir les métriques et l'état des ressources
show_metrics() {
  log "info" "Affichage des métriques du cluster..."
  
  # Métriques des nœuds
  log "info" "Métriques des nœuds:"
  kubectl top nodes
  
  # Métriques des pods
  log "info" "Métriques des pods:"
  kubectl top pods -n evil2root-trading
  
  # État des HPAs
  log "info" "État des Horizontal Pod Autoscalers:"
  kubectl get hpa -n evil2root-trading
  
  # État des PVCs
  log "info" "État des Persistent Volume Claims:"
  kubectl get pvc -n evil2root-trading
  
  # État des services
  log "info" "État des services:"
  kubectl get services -n evil2root-trading
}

# Fonction principale
main() {
  # Variables globales
  export USE_AZURE=false
  export ENV="development"
  
  # Analyser les options
  while [[ $# -gt 0 ]]; do
    case $1 in
      --azure)
        export USE_AZURE=true
        shift
        ;;
      --production)
        export ENV="production"
        shift
        ;;
      *)
        break
        ;;
    esac
  done
  
  COMMAND=$1
  shift
  
  case $COMMAND in
    "deploy")
      check_prerequisites
      if [[ "$USE_AZURE" == "true" ]]; then
        configure_azure
      fi
      deploy
      ;;
    "update_image")
      update_image "$@"
      ;;
    "pods")
      show_pods
      ;;
    "logs")
      show_logs "$@"
      ;;
    "restore_backup")
      restore_backup "$@"
      ;;
    "metrics")
      show_metrics
      ;;
    "setup_aks")
      export USE_AZURE=true
      check_prerequisites
      configure_azure
      setup_aks_cluster
      ;;
    "create_aks_cluster")
      export USE_AZURE=true
      check_prerequisites
      configure_azure
      create_aks_cluster
      ;;
    "connect_aks_cluster")
      export USE_AZURE=true
      check_prerequisites
      configure_azure
      connect_aks_cluster
      ;;
    "setup_azure_storage")
      export USE_AZURE=true
      check_prerequisites
      configure_azure
      setup_azure_storage
      ;;
    *)
      echo "Evil2Root Trading Bot - Outil de déploiement Kubernetes pour Azure"
      echo
      echo "Utilisation: $0 [options] <commande> [arguments]"
      echo
      echo "Options:"
      echo "  --azure         Utiliser les fonctionnalités spécifiques à Azure"
      echo "  --production    Déployer en mode production (plus sécurisé)"
      echo
      echo "Commandes:"
      echo "  deploy             Déploie l'ensemble de l'application sur Kubernetes"
      echo "  update_image <tag> Met à jour l'image du trading bot vers la version spécifiée"
      echo "  pods               Affiche les pods en cours d'exécution"
      echo "  logs <pod_name>    Affiche les logs d'un pod spécifique"
      echo "  restore_backup <backup_file> Restaure une sauvegarde de la base de données"
      echo "  metrics            Affiche les métriques et l'état des ressources"
      echo
      echo "Commandes Azure:"
      echo "  setup_aks           Configure le cluster AKS avec les prérequis (nginx, cert-manager)"
      echo "  create_aks_cluster  Crée un nouveau cluster Kubernetes sur Azure"
      echo "  connect_aks_cluster Se connecte à un cluster Kubernetes existant sur Azure"
      echo "  setup_azure_storage Configure un partage de fichiers Azure pour les backups"
      echo
      echo "Exemples:"
      echo "  $0 --azure create_aks_cluster"
      echo "  $0 --azure --production deploy"
      echo "  $0 update_image v1.2.3"
      exit 1
      ;;
  esac
}

# Exécution du script
main "$@" 
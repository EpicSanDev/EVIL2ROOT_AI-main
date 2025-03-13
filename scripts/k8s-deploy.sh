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
    "digitalocean")
      echo -e "${BLUE}[DIGITALOCEAN]${RESET} $message"
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
  
  # Vérifier doctl si on utilise DigitalOcean
  if [[ "$USE_DIGITALOCEAN" == "true" ]] && ! command -v doctl &> /dev/null; then
    log "warn" "doctl n'est pas installé. Installation..."
    
    # Détection du système d'exploitation
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    
    case "$OS" in
      darwin)
        log "info" "Installation de doctl via Homebrew..."
        brew install doctl
        ;;
      linux)
        log "info" "Installation de doctl pour Linux..."
        LATEST_VERSION=$(curl -s https://api.github.com/repos/digitalocean/doctl/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")')
        curl -sL https://github.com/digitalocean/doctl/releases/download/$LATEST_VERSION/doctl-$LATEST_VERSION-linux-amd64.tar.gz | tar -xzv
        sudo mv doctl /usr/local/bin/
        ;;
      *)
        log "error" "Système d'exploitation non pris en charge pour l'installation automatique de doctl. Veuillez l'installer manuellement."
        exit 1
        ;;
    esac
  fi
  
  log "info" "Prérequis vérifiés avec succès."
}

# Configuration de DigitalOcean
configure_digitalocean() {
  log "digitalocean" "Configuration de l'accès à DigitalOcean..."

  # Vérifier si le token est déjà configuré
  if doctl account get &> /dev/null; then
    log "digitalocean" "Token DigitalOcean déjà configuré."
  else
    # Demander le token API
    read -p "Entrez votre token API DigitalOcean: " DO_API_TOKEN
    
    # Configurer doctl avec le token
    doctl auth init -t "$DO_API_TOKEN"
    
    if [ $? -ne 0 ]; then
      log "error" "Échec de l'authentification avec DigitalOcean. Vérifiez votre token API."
      exit 1
    fi
    
    log "digitalocean" "Authentification réussie avec DigitalOcean."
  fi
}

# Création d'un cluster Kubernetes sur DigitalOcean
create_do_cluster() {
  log "digitalocean" "Création d'un nouveau cluster Kubernetes sur DigitalOcean..."
  
  # Paramètres par défaut optimisés pour Evil2Root Trading
  DEFAULT_CLUSTER_NAME="evil2root-trading"
  DEFAULT_REGION="fra1"  # Frankfurt (Europe)
  DEFAULT_NODE_SIZE="s-4vcpu-8gb"  # 4 vCPUs, 8GB RAM (optimal pour le trading)
  DEFAULT_NODE_COUNT=3  # 3 nœuds pour la haute disponibilité
  
  # Demander confirmation ou modification des valeurs par défaut
  read -p "Nom du cluster [$DEFAULT_CLUSTER_NAME]: " CLUSTER_NAME
  CLUSTER_NAME=${CLUSTER_NAME:-$DEFAULT_CLUSTER_NAME}
  
  read -p "Région [$DEFAULT_REGION]: " REGION
  REGION=${REGION:-$DEFAULT_REGION}
  
  # Afficher les tailles de droplet disponibles
  log "digitalocean" "Récupération des tailles de droplet disponibles..."
  echo "Tailles disponibles:"
  doctl compute size list --format Slug,Memory,VCPUs,Disk,PriceMonthly | head -20
  
  log "digitalocean" "Taille recommandée pour Evil2Root Trading: $DEFAULT_NODE_SIZE (4 vCPUs, 8GB RAM)"
  read -p "Taille des nœuds [$DEFAULT_NODE_SIZE]: " NODE_SIZE
  NODE_SIZE=${NODE_SIZE:-$DEFAULT_NODE_SIZE}
  
  log "digitalocean" "Nombre de nœuds recommandé: $DEFAULT_NODE_COUNT (pour la haute disponibilité)"
  log "digitalocean" "Utilisation automatique de $DEFAULT_NODE_COUNT nœuds pour assurer la haute disponibilité de l'application de trading"
  NODE_COUNT=$DEFAULT_NODE_COUNT
  
  # Vérifier si la taille est valide
  if ! doctl compute size list --format Slug --no-header | grep -q "^${NODE_SIZE}$"; then
    log "error" "Taille de droplet invalide: ${NODE_SIZE}"
    log "digitalocean" "Veuillez choisir une taille valide parmi la liste ci-dessus."
    return 1
  fi
  
  # Confirmation finale
  log "digitalocean" "Récapitulatif de la configuration du cluster:"
  log "digitalocean" "- Nom: $CLUSTER_NAME"
  log "digitalocean" "- Région: $REGION"
  log "digitalocean" "- Taille des nœuds: $NODE_SIZE"
  log "digitalocean" "- Nombre de nœuds: $NODE_COUNT"
  
  read -p "Confirmer la création du cluster avec ces paramètres? (o/n): " CONFIRM
  if [[ "$CONFIRM" != "o" ]]; then
    log "digitalocean" "Création du cluster annulée."
    return 1
  fi
  
  # Créer le cluster
  log "digitalocean" "Création du cluster $CLUSTER_NAME dans la région $REGION..."
  doctl kubernetes cluster create "$CLUSTER_NAME" \
    --region "$REGION" \
    --size "$NODE_SIZE" \
    --count "$NODE_COUNT" \
    --auto-upgrade=true \
    --ha=true \
    --wait
  
  if [ $? -ne 0 ]; then
    log "error" "Échec de la création du cluster Kubernetes."
    exit 1
  fi
  
  # Configuration de kubectl avec le nouveau cluster
  log "digitalocean" "Configuration de kubectl pour utiliser le nouveau cluster..."
  doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"
  
  log "digitalocean" "Cluster Kubernetes créé avec succès. Attendez quelques minutes que tous les nœuds soient prêts."
  
  # Configuration automatique des resources limits pour le projet
  log "digitalocean" "Configuration automatique des limites de ressources pour Evil2Root Trading..."
  
  # Vérifier l'état des nœuds
  kubectl get nodes
}

# Connexion à un cluster existant
connect_do_cluster() {
  log "digitalocean" "Connexion à un cluster Kubernetes existant sur DigitalOcean..."
  
  # Lister les clusters disponibles
  log "digitalocean" "Clusters disponibles:"
  doctl kubernetes cluster list
  
  # Demander l'ID ou le nom du cluster
  read -p "Entrez l'ID ou le nom du cluster à utiliser: " CLUSTER_ID
  
  # Configuration de kubectl avec le cluster existant
  log "digitalocean" "Configuration de kubectl pour utiliser le cluster $CLUSTER_ID..."
  doctl kubernetes cluster kubeconfig save "$CLUSTER_ID"
  
  if [ $? -ne 0 ]; then
    log "error" "Échec de la configuration de kubectl avec le cluster DigitalOcean."
    exit 1
  fi
  
  log "digitalocean" "Connexion au cluster réussie."
  
  # Vérifier l'état des nœuds
  kubectl get nodes
}

# Installer cert-manager pour les certificats TLS
install_cert_manager() {
  log "digitalocean" "Installation de cert-manager pour la gestion des certificats TLS..."
  
  # Vérifier si cert-manager est déjà installé
  if kubectl get namespace cert-manager &> /dev/null; then
    log "digitalocean" "cert-manager est déjà installé."
    return
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
  
  log "digitalocean" "cert-manager a été installé avec succès."
}

# Installer Nginx Ingress Controller
install_nginx_ingress() {
  log "digitalocean" "Installation du contrôleur Nginx Ingress..."
  
  # Vérifier si nginx ingress est déjà installé
  if kubectl get namespace ingress-nginx &> /dev/null; then
    log "digitalocean" "Nginx Ingress est déjà installé."
    return
  fi
  
  # Ajouter le dépôt Helm pour Nginx Ingress
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
  helm repo update
  
  # Installer Nginx Ingress Controller
  kubectl create namespace ingress-nginx
  helm install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx \
    --set controller.publishService.enabled=true
  
  # Attendre que l'ingress controller soit prêt
  kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=120s
  
  # Obtenir l'IP externe de l'ingress controller
  log "digitalocean" "En attente de l'attribution d'une IP externe au contrôleur d'ingress..."
  sleep 30  # Attendre que le load balancer soit provisionné
  
  INGRESS_IP=$(kubectl -n ingress-nginx get service ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  log "digitalocean" "Contrôleur d'ingress installé avec succès. IP externe: $INGRESS_IP"
  log "digitalocean" "Veuillez configurer vos enregistrements DNS pour pointer vers cette IP."
}

# Fonction pour configurer le cluster avec les prérequis
setup_do_cluster() {
  log "digitalocean" "Configuration du cluster DigitalOcean avec les prérequis..."
  
  # Installer Helm si nécessaire
  if ! command -v helm &> /dev/null; then
    log "warn" "Helm n'est pas installé. Installation..."
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    chmod 700 get_helm.sh
    ./get_helm.sh
    rm get_helm.sh
  fi
  
  # Installer Nginx Ingress Controller
  install_nginx_ingress
  
  # Installer cert-manager
  install_cert_manager
  
  # Installer Metrics Server pour HPA
  if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
    log "digitalocean" "Installation de Metrics Server..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    kubectl -n kube-system wait --for=condition=ready pod -l k8s-app=metrics-server --timeout=60s
  fi
  
  log "digitalocean" "Configuration de base du cluster terminée avec succès."
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
  
  if [[ "$USE_DIGITALOCEAN" == "true" ]]; then
    log "digitalocean" "Déploiement terminé. Assurez-vous de configurer vos DNS pour pointer vers l'IP du load balancer."
    INGRESS_IP=$(kubectl -n ingress-nginx get service ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    log "digitalocean" "IP du load balancer Nginx: $INGRESS_IP"
  fi
}

# Fonction pour mettre à jour l'image du trading bot
update_image() {
  VERSION=$1
  if [[ -z "$VERSION" ]]; then
    log "error" "Version non spécifiée. Utilisation: ./k8s-deploy.sh update_image <version>"
    exit 1
  fi
  
  log "info" "Mise à jour de l'image vers registry.digitalocean.com/epicsandev:$VERSION..."
  
  # Mise à jour de l'image pour les déploiements
  kubectl set image deployment/trading-bot-web web=registry.digitalocean.com/epicsandev:$VERSION -n evil2root-trading
  kubectl set image deployment/analysis-bot analysis-bot=registry.digitalocean.com/epicsandev:$VERSION -n evil2root-trading
  kubectl set image deployment/market-scheduler market-scheduler=registry.digitalocean.com/epicsandev:$VERSION -n evil2root-trading
  
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
    log "error" "Nom du pod non spécifié. Utilisation: ./k8s-deploy.sh logs <nom_pod>"
    
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
    log "error" "Nom de la sauvegarde non spécifié. Utilisation: ./k8s-deploy.sh restore_backup <nom_fichier_backup>"
    
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
  export USE_DIGITALOCEAN=false
  export ENV="development"
  
  # Analyser les options
  while [[ $# -gt 0 ]]; do
    case $1 in
      --digitalocean)
        export USE_DIGITALOCEAN=true
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
      if [[ "$USE_DIGITALOCEAN" == "true" ]]; then
        configure_digitalocean
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
    "setup_do")
      export USE_DIGITALOCEAN=true
      check_prerequisites
      configure_digitalocean
      setup_do_cluster
      ;;
    "create_do_cluster")
      export USE_DIGITALOCEAN=true
      check_prerequisites
      configure_digitalocean
      create_do_cluster
      ;;
    "connect_do_cluster")
      export USE_DIGITALOCEAN=true
      check_prerequisites
      configure_digitalocean
      connect_do_cluster
      ;;
    *)
      echo "Evil2Root Trading Bot - Outil de déploiement Kubernetes"
      echo
      echo "Utilisation: $0 [options] <commande> [arguments]"
      echo
      echo "Options:"
      echo "  --digitalocean    Utiliser les fonctionnalités spécifiques à DigitalOcean"
      echo "  --production      Déployer en mode production (plus sécurisé)"
      echo
      echo "Commandes:"
      echo "  deploy             Déploie l'ensemble de l'application sur Kubernetes"
      echo "  update_image <tag> Met à jour l'image du trading bot vers la version spécifiée"
      echo "  pods               Affiche les pods en cours d'exécution"
      echo "  logs <pod_name>    Affiche les logs d'un pod spécifique"
      echo "  restore_backup <backup_file> Restaure une sauvegarde de la base de données"
      echo "  metrics            Affiche les métriques et l'état des ressources"
      echo
      echo "Commandes DigitalOcean:"
      echo "  setup_do           Configure le cluster DigitalOcean avec les prérequis (nginx, cert-manager)"
      echo "  create_do_cluster  Crée un nouveau cluster Kubernetes sur DigitalOcean"
      echo "  connect_do_cluster Se connecte à un cluster Kubernetes existant sur DigitalOcean"
      echo
      echo "Exemples:"
      echo "  $0 --digitalocean create_do_cluster"
      echo "  $0 --digitalocean --production deploy"
      echo "  $0 update_image v1.2.3"
      exit 1
      ;;
  esac
}

# Exécution du script
main "$@" 
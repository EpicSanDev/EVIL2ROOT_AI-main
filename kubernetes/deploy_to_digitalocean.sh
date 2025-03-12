#!/bin/bash

# Script de déploiement automatisé sur DigitalOcean pour EVIL2ROOT Trading Bot
# Ce script crée un cluster Kubernetes sur DigitalOcean et y déploie l'application

# Couleurs pour le formatage
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables par défaut (peuvent être modifiées en passant des arguments)
CLUSTER_NAME="evil2root-trading"
REGION="fra1" # Frankfurt par défaut
NODE_SIZE="s-4vcpu-8gb" # 4 CPUs, 8GB RAM (bon équilibre coût/performance)
NODE_COUNT=3
GPU_NODE_SIZE="g-8vcpu-32gb" # Nœud GPU (si disponible)
GPU_NODE_COUNT=0 # Par défaut, pas de nœuds GPU
DO_TOKEN=""
CREATE_GPU_NODES=false
CUSTOM_DOMAIN=""

# Fonctions utilitaires
print_header() {
    echo -e "\n${GREEN}==== $1 ====${NC}\n"
}

print_info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}AVERTISSEMENT: $1${NC}"
}

print_error() {
    echo -e "${RED}ERREUR: $1${NC}"
}

# Affichage de l'aide
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help              Affiche cette aide"
    echo "  -t, --token TOKEN       Token API DigitalOcean (obligatoire)"
    echo "  -n, --name NAME         Nom du cluster (défaut: evil2root-trading)"
    echo "  -r, --region REGION     Région DigitalOcean (défaut: fra1)"
    echo "  -s, --size SIZE         Taille des nœuds (défaut: s-4vcpu-8gb)"
    echo "  -c, --count COUNT       Nombre de nœuds (défaut: 3)"
    echo "  -g, --gpu               Ajouter un nœud GPU (si disponible)"
    echo "  -d, --domain DOMAIN     Domaine personnalisé pour les ingress"
    echo
    echo "Exemples:"
    echo "  $0 --token your_token --name prod-cluster --region nyc1 --size s-8vcpu-16gb --count 5"
    echo "  $0 -t your_token -g -d trading.votredomaine.com"
    exit 0
}

# Vérification des prérequis
check_prerequisites() {
    print_header "Vérification des prérequis"
    
    # Vérifier si curl est installé
    if ! command -v curl &> /dev/null; then
        print_error "curl n'est pas installé. Veuillez l'installer et réessayer."
        exit 1
    fi
    
    # Vérifier si jq est installé
    if ! command -v jq &> /dev/null; then
        print_error "jq n'est pas installé. Veuillez l'installer et réessayer."
        exit 1
    fi
    
    # Vérifier si doctl est installé
    if ! command -v doctl &> /dev/null; then
        print_error "doctl n'est pas installé. Veuillez l'installer et réessayer."
        print_info "Installation: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        exit 1
    fi
    
    # Vérifier si kubectl est installé
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl n'est pas installé. Veuillez l'installer et réessayer."
        exit 1
    fi
    
    # Vérifier le token API DigitalOcean
    if [[ -z "$DO_TOKEN" ]]; then
        print_error "Token API DigitalOcean non fourni. Utilisez l'option -t ou --token."
        exit 1
    fi

    print_info "Tous les prérequis sont satisfaits."
}

# Configuration de doctl avec le token
configure_doctl() {
    print_header "Configuration de doctl"
    
    # Tentative d'authentification avec doctl
    doctl auth init -t "$DO_TOKEN"
    
    # Vérification que l'authentification a fonctionné
    if ! doctl account get &> /dev/null; then
        print_error "Échec de l'authentification avec doctl. Vérifiez votre token."
        exit 1
    fi
    
    print_info "doctl configuré avec succès."
}

# Vérification des régions disponibles
check_regions() {
    print_header "Vérification de la région"
    
    # Obtenir la liste des régions disponibles
    REGIONS=$(doctl compute region list --format Slug --no-header)
    
    # Vérifier si la région spécifiée est disponible
    if ! echo "$REGIONS" | grep -q "$REGION"; then
        print_error "La région '$REGION' n'est pas disponible."
        print_info "Régions disponibles: $REGIONS"
        exit 1
    fi
    
    print_info "La région '$REGION' est disponible."
}

# Vérification des tailles de nœuds disponibles
check_node_sizes() {
    print_header "Vérification des tailles de nœuds"
    
    # Obtenir la liste des tailles disponibles dans cette région
    SIZES=$(doctl compute size list --format Slug --no-header)
    
    # Vérifier si la taille spécifiée est disponible
    if ! echo "$SIZES" | grep -q "$NODE_SIZE"; then
        print_error "La taille de nœud '$NODE_SIZE' n'est pas disponible."
        print_info "Tailles disponibles: $SIZES"
        exit 1
    fi
    
    print_info "La taille de nœud '$NODE_SIZE' est disponible."
    
    # Vérifier si les nœuds GPU sont demandés et disponibles
    if [[ "$CREATE_GPU_NODES" == true ]]; then
        if ! echo "$SIZES" | grep -q "$GPU_NODE_SIZE"; then
            print_warning "La taille de nœud GPU '$GPU_NODE_SIZE' n'est pas disponible. Aucun nœud GPU ne sera créé."
            CREATE_GPU_NODES=false
        else
            print_info "La taille de nœud GPU '$GPU_NODE_SIZE' est disponible."
            GPU_NODE_COUNT=1
        fi
    fi
}

# Création du cluster Kubernetes
create_cluster() {
    print_header "Création du cluster Kubernetes"
    
    print_info "Création du cluster '$CLUSTER_NAME' dans la région '$REGION' avec $NODE_COUNT nœuds de taille '$NODE_SIZE'..."
    
    # Construire la commande de création du cluster
    CMD="doctl kubernetes cluster create $CLUSTER_NAME --region $REGION --size $NODE_SIZE --count $NODE_COUNT --wait"
    
    # Exécuter la commande
    print_info "Exécution: $CMD"
    eval "$CMD"
    
    if [ $? -ne 0 ]; then
        print_error "Échec de la création du cluster. Vérifiez les logs ci-dessus."
        exit 1
    fi
    
    print_info "Cluster Kubernetes '$CLUSTER_NAME' créé avec succès."
}

# Ajout de nœuds GPU (si demandé)
add_gpu_nodes() {
    if [[ "$CREATE_GPU_NODES" == true ]]; then
        print_header "Ajout de nœuds GPU"
        
        print_info "Ajout de $GPU_NODE_COUNT nœuds GPU de taille '$GPU_NODE_SIZE'..."
        
        # Obtenir l'ID du cluster
        CLUSTER_ID=$(doctl kubernetes cluster list --format ID --no-header --name "$CLUSTER_NAME")
        
        # Ajouter un pool de nœuds GPU
        doctl kubernetes cluster node-pool create "$CLUSTER_ID" \
            --name "gpu-pool" \
            --size "$GPU_NODE_SIZE" \
            --count "$GPU_NODE_COUNT" \
            --label "gpu=true" \
            --taint "nvidia.com/gpu=present:NoSchedule"
        
        if [ $? -ne 0 ]; then
            print_warning "Échec de l'ajout des nœuds GPU. Le déploiement continuera sans GPU."
        else
            print_info "Nœuds GPU ajoutés avec succès."
        fi
    fi
}

# Configuration de kubectl pour utiliser le nouveau cluster
configure_kubectl() {
    print_header "Configuration de kubectl"
    
    print_info "Configuration de kubectl pour utiliser le cluster '$CLUSTER_NAME'..."
    
    # Sauvegarder la configuration originale
    if [ -f ~/.kube/config ]; then
        cp ~/.kube/config ~/.kube/config.bak
        print_info "Configuration kubectl originale sauvegardée dans ~/.kube/config.bak"
    fi
    
    # Mise à jour de la configuration kubectl
    doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"
    
    if [ $? -ne 0 ]; then
        print_error "Échec de la configuration de kubectl. Restauration de la configuration originale."
        if [ -f ~/.kube/config.bak ]; then
            cp ~/.kube/config.bak ~/.kube/config
        fi
        exit 1
    fi
    
    print_info "kubectl configuré pour utiliser le cluster '$CLUSTER_NAME'."
}

# Installation de l'NGINX Ingress Controller
install_nginx_ingress() {
    print_header "Installation de l'NGINX Ingress Controller"
    
    print_info "Installation de l'NGINX Ingress Controller..."
    
    # Appliquer le manifeste de l'Ingress Controller
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
    
    # Attendre que l'Ingress Controller soit prêt
    print_info "Attente du déploiement de l'Ingress Controller..."
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=300s
    
    if [ $? -ne 0 ]; then
        print_warning "L'Ingress Controller n'est pas prêt. Le déploiement continuera, mais les Ingress pourraient ne pas fonctionner."
    else
        print_info "NGINX Ingress Controller installé avec succès."
    fi
}

# Installation de cert-manager pour les certificats SSL
install_cert_manager() {
    print_header "Installation de cert-manager"
    
    print_info "Installation de cert-manager pour gérer les certificats SSL..."
    
    # Appliquer le manifeste de cert-manager
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml
    
    # Attendre que cert-manager soit prêt
    print_info "Attente du déploiement de cert-manager..."
    kubectl wait --namespace cert-manager \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=300s
    
    if [ $? -ne 0 ]; then
        print_warning "cert-manager n'est pas prêt. Le déploiement continuera, mais les certificats SSL pourraient ne pas fonctionner."
    else
        print_info "cert-manager installé avec succès."
    fi
    
    # Créer l'émetteur de certificats Let's Encrypt
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
    
    print_info "Émetteur de certificats Let's Encrypt configuré."
}

# Personnalisation des fichiers d'Ingress avec le domaine spécifié
customize_ingress_domains() {
    if [[ -n "$CUSTOM_DOMAIN" ]]; then
        print_header "Personnalisation des domaines dans les Ingress"
        
        print_info "Mise à jour des domaines des Ingress pour utiliser: $CUSTOM_DOMAIN"
        
        # Remplacer les domaines d'exemple dans les fichiers d'Ingress
        sed -i.bak "s/trading.example.com/trading.$CUSTOM_DOMAIN/g" web-deployment.yaml
        sed -i.bak "s/grafana.trading.example.com/grafana.trading.$CUSTOM_DOMAIN/g" monitoring-deployment.yaml
        sed -i.bak "s/adminer.trading.example.com/adminer.trading.$CUSTOM_DOMAIN/g" adminer-deployment.yaml
        
        print_info "Domaines des Ingress mis à jour."
    else
        print_warning "Aucun domaine personnalisé spécifié. Les domaines d'exemple seront utilisés."
        print_info "Pour accéder aux services, vous devrez configurer votre DNS ou modifier les fichiers manuellement."
    fi
}

# Déploiement de l'application
deploy_application() {
    print_header "Déploiement de l'application EVIL2ROOT Trading Bot"
    
    # Adapter les fichiers pour les nœuds GPU (si disponibles)
    if [[ "$CREATE_GPU_NODES" == true ]]; then
        print_info "Configuration des jobs pour utiliser les nœuds GPU..."
        
        # Ajouter un nodeSelector aux jobs d'entraînement
        sed -i.bak '/restartPolicy: Never/i\      nodeSelector:\n        gpu: "true"' train-analyze-job.yaml
    else
        print_info "Adaptation des jobs pour fonctionner sans GPU..."
        
        # Supprimer les demandes de ressources GPU dans les jobs
        sed -i.bak '/nvidia.com\/gpu:/d' train-analyze-job.yaml
    fi
    
    # Exécuter le script de déploiement
    print_info "Exécution du script de déploiement..."
    ./deploy.sh
    
    if [ $? -ne 0 ]; then
        print_error "Échec du déploiement de l'application. Vérifiez les logs ci-dessus."
        exit 1
    fi
    
    print_info "EVIL2ROOT Trading Bot déployé avec succès sur DigitalOcean Kubernetes."
}

# Affichage des informations finales
print_final_info() {
    print_header "Informations de déploiement"
    
    # Obtenir l'adresse IP de l'équilibreur de charge de l'Ingress
    print_info "Récupération de l'adresse IP du load balancer..."
    LB_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    echo -e "\n${GREEN}Déploiement terminé avec succès!${NC}\n"
    echo -e "Cluster Kubernetes: ${BLUE}$CLUSTER_NAME${NC}"
    echo -e "Région: ${BLUE}$REGION${NC}"
    echo -e "Nombre de nœuds: ${BLUE}$NODE_COUNT${NC} (taille: $NODE_SIZE)"
    
    if [[ "$CREATE_GPU_NODES" == true ]]; then
        echo -e "Nœuds GPU: ${BLUE}$GPU_NODE_COUNT${NC} (taille: $GPU_NODE_SIZE)"
    fi
    
    echo -e "\n${YELLOW}Configuration DNS:${NC}"
    
    if [[ -n "$CUSTOM_DOMAIN" ]]; then
        echo -e "Pour utiliser vos domaines, créez des enregistrements DNS A pointant vers: ${BLUE}$LB_IP${NC}"
        echo -e "  - trading.$CUSTOM_DOMAIN -> $LB_IP"
        echo -e "  - grafana.trading.$CUSTOM_DOMAIN -> $LB_IP"
        echo -e "  - adminer.trading.$CUSTOM_DOMAIN -> $LB_IP"
    else
        echo -e "Adresse IP du load balancer: ${BLUE}$LB_IP${NC}"
        echo -e "Modifiez vos fichiers d'Ingress ou votre fichier /etc/hosts pour utiliser cette adresse."
    fi
    
    echo -e "\n${YELLOW}Accès aux services:${NC}"
    
    if [[ -n "$CUSTOM_DOMAIN" ]]; then
        echo -e "Interface Web: ${BLUE}https://trading.$CUSTOM_DOMAIN${NC}"
        echo -e "Grafana: ${BLUE}https://grafana.trading.$CUSTOM_DOMAIN${NC}"
        echo -e "Adminer: ${BLUE}https://adminer.trading.$CUSTOM_DOMAIN${NC}"
    else
        echo -e "Une fois la configuration DNS effectuée, accédez aux services via les URLs configurés."
    fi
    
    echo -e "\n${YELLOW}Gestion du cluster:${NC}"
    echo -e "Pour gérer votre cluster, utilisez la commande doctl ou le panneau de contrôle DigitalOcean."
    echo -e "Pour supprimer le cluster: ${BLUE}doctl kubernetes cluster delete $CLUSTER_NAME${NC}"
}

# Traitement des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -t|--token)
            DO_TOKEN="$2"
            shift 2
            ;;
        -n|--name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -s|--size)
            NODE_SIZE="$2"
            shift 2
            ;;
        -c|--count)
            NODE_COUNT="$2"
            shift 2
            ;;
        -g|--gpu)
            CREATE_GPU_NODES=true
            shift
            ;;
        -d|--domain)
            CUSTOM_DOMAIN="$2"
            shift 2
            ;;
        *)
            print_error "Option inconnue: $1"
            echo "Utilisez --help pour afficher les options disponibles."
            exit 1
            ;;
    esac
done

# Exécution principale
check_prerequisites
configure_doctl
check_regions
check_node_sizes
create_cluster
add_gpu_nodes
configure_kubectl
install_nginx_ingress
install_cert_manager
customize_ingress_domains
deploy_application
print_final_info

exit 0 
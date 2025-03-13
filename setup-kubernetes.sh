#!/bin/bash

# Variables
CLUSTER_NAME="evil2root-trading"
NAMESPACE="evil2root-trading"

# Couleurs pour le formatage
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonctions
print_header() {
    echo -e "\n${GREEN}==== $1 ====${NC}\n"
}

print_warning() {
    echo -e "${YELLOW}AVERTISSEMENT: $1${NC}"
}

print_error() {
    echo -e "${RED}ERREUR: $1${NC}"
}

check_prerequisites() {
    print_header "Vérification des prérequis"
    
    # Vérifier si Docker est installé
    if ! command -v docker &> /dev/null; then
        print_error "Docker n'est pas installé. Veuillez l'installer et réessayer."
        echo "https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Vérifier si Kind est installé
    if ! command -v kind &> /dev/null; then
        print_warning "Kind n'est pas installé. Installation en cours..."
        
        # Détection du système d'exploitation
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
            chmod +x ./kind
            sudo mv ./kind /usr/local/bin/kind
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # MacOS
            if [[ $(uname -m) == "arm64" ]]; then
                # M1/M2
                brew install kind
            else
                # Intel
                brew install kind
            fi
        else
            print_error "Système d'exploitation non supporté. Veuillez installer Kind manuellement:"
            echo "https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
            exit 1
        fi
        
        echo "Kind installé avec succès."
    fi
    
    # Vérifier si kubectl est installé
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl n'est pas installé. Installation en cours..."
        
        # Détection du système d'exploitation
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
            chmod +x kubectl
            sudo mv kubectl /usr/local/bin/
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # MacOS
            brew install kubectl
        else
            print_error "Système d'exploitation non supporté. Veuillez installer kubectl manuellement:"
            echo "https://kubernetes.io/docs/tasks/tools/install-kubectl/"
            exit 1
        fi
        
        echo "kubectl installé avec succès."
    fi
    
    # Vérifier si Helm est installé (utile pour certaines installations)
    if ! command -v helm &> /dev/null; then
        print_warning "Helm n'est pas installé. Installation en cours..."
        
        # Détection du système d'exploitation
        if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
            curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        else
            print_warning "Système d'exploitation non supporté pour l'installation automatique de Helm."
            echo "Veuillez installer Helm manuellement: https://helm.sh/docs/intro/install/"
        fi
    fi
}

check_existing_cluster() {
    print_header "Vérification des clusters existants"
    
    if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
        print_warning "Un cluster Kind nommé '${CLUSTER_NAME}' existe déjà."
        read -p "Voulez-vous le supprimer et en créer un nouveau? (o/n) " response
        if [[ "$response" =~ ^[Oo]$ ]]; then
            echo "Suppression du cluster existant..."
            kind delete cluster --name $CLUSTER_NAME
        else
            echo "Utilisation du cluster existant..."
            return 1
        fi
    fi
    
    return 0
}

create_cluster() {
    print_header "Création du cluster Kind"
    
    # Créer un fichier de configuration pour le cluster Kind avec ingress
    cat <<EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: $CLUSTER_NAME
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
EOF
    
    # Créer le cluster
    kind create cluster --config=kind-config.yaml
    
    # Vérifier que le cluster a été créé avec succès
    if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
        print_error "La création du cluster a échoué."
        exit 1
    fi
    
    echo "Cluster Kind '$CLUSTER_NAME' créé avec succès."
    
    # Nettoyer le fichier de configuration
    rm kind-config.yaml
}

setup_ingress() {
    print_header "Installation de l'Ingress NGINX"
    
    echo "Installation de l'ingress NGINX pour Kind..."
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
    
    echo "Attente du démarrage de l'Ingress NGINX..."
    kubectl wait --namespace ingress-nginx \
      --for=condition=ready pod \
      --selector=app.kubernetes.io/component=controller \
      --timeout=90s
      
    echo "Ingress NGINX installé avec succès."
}

create_namespace() {
    print_header "Création du namespace"
    
    # Vérifier si le namespace existe déjà
    if kubectl get namespace | grep -q $NAMESPACE; then
        echo "Le namespace '$NAMESPACE' existe déjà."
    else
        # Créer le namespace
        kubectl create namespace $NAMESPACE
        echo "Namespace '$NAMESPACE' créé avec succès."
    fi
}

setup_storage_class() {
    print_header "Configuration de la classe de stockage"
    
    # Vérifier que la classe de stockage standard existe
    if ! kubectl get storageclass standard &> /dev/null; then
        echo "Création d'une classe de stockage standard..."
        
        # Créer une classe de stockage standard pour Kind
        cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
provisioner: rancher.io/local-path
volumeBindingMode: WaitForFirstConsumer
EOF
    fi
    
    # Définir la classe de stockage standard comme classe par défaut
    kubectl patch storageclass standard -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    
    echo "Classe de stockage configurée avec succès."
}

print_next_steps() {
    print_header "Prochaines étapes"
    
    echo "Votre cluster Kubernetes a été configuré avec succès!"
    echo ""
    echo "Pour déployer votre application, exécutez:"
    echo "  ./deploy-local.sh"
    echo ""
    echo "Pour vérifier l'état du cluster:"
    echo "  kubectl get nodes"
    echo "  kubectl get pods -n $NAMESPACE"
    echo ""
    echo "Pour supprimer le cluster quand vous avez terminé:"
    echo "  kind delete cluster --name $CLUSTER_NAME"
}

# Exécution principale
check_prerequisites

if check_existing_cluster; then
    create_cluster
    setup_ingress
fi

create_namespace
setup_storage_class
print_next_steps 
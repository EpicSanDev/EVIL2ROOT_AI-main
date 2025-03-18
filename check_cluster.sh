#!/bin/bash

# Script de diagnostic pour cluster Kubernetes sur DigitalOcean
# Auteur: Claude AI
# Version: 1.0

# Couleurs pour le formatage
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Variables
CLUSTER_NAME="evil2root-trading"
DO_TOKEN=""
NAMESPACE="evil2root-trading"

# Affichage de l'aide
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help              Affiche cette aide"
    echo "  -t, --token TOKEN       Token API DigitalOcean (obligatoire si non défini dans DIGITALOCEAN_TOKEN)"
    echo "  -n, --name NAME         Nom du cluster (défaut: evil2root-trading)"
    echo "  -s, --namespace NS      Namespace Kubernetes à vérifier (défaut: evil2root-trading)"
    echo "  -a, --all               Exécute tous les diagnostics disponibles"
    echo
    echo "Exemples:"
    echo "  $0 --token your_token"
    echo "  $0 -t your_token -n prod-cluster -s prod-namespace"
    exit 0
}

# Vérification des prérequis
check_prerequisites() {
    print_header "Vérification des prérequis"
    
    # Vérifier si python3 est installé
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 n'est pas installé. Veuillez l'installer et réessayer."
        exit 1
    fi
    
    # Vérifier si le package requests est installé
    if ! python3 -c "import requests" &> /dev/null; then
        print_warning "Le package Python 'requests' n'est pas installé. Installation en cours..."
        pip3 install requests
    fi
    
    # Vérifier si kubectl est installé
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl n'est pas installé. Certaines fonctionnalités ne seront pas disponibles."
        KUBECTL_AVAILABLE=false
    else
        KUBECTL_AVAILABLE=true
    fi
    
    # Vérifier si le token est défini
    if [[ -z "$DO_TOKEN" ]]; then
        if [[ -n "$DIGITALOCEAN_TOKEN" ]]; then
            DO_TOKEN="$DIGITALOCEAN_TOKEN"
        else
            print_error "Token API DigitalOcean non fourni. Utilisez l'option -t ou --token, ou définissez la variable d'environnement DIGITALOCEAN_TOKEN."
            exit 1
        fi
    fi
}

# Vérification du cluster via l'API DigitalOcean
check_cluster_api() {
    print_header "Vérification du cluster via l'API DigitalOcean"
    
    python3 check_do_k8s_status.py -t "$DO_TOKEN" -n "$CLUSTER_NAME" -v
}

# Vérification des ressources Kubernetes
check_kubernetes_resources() {
    if [[ "$KUBECTL_AVAILABLE" == "false" ]]; then
        print_warning "kubectl n'est pas disponible, impossible de vérifier les ressources Kubernetes."
        return
    fi
    
    print_header "Vérification des ressources Kubernetes"
    
    # Vérifier si le contexte Kubernetes est configuré correctement
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Impossible de se connecter au cluster Kubernetes."
        print_info "Assurez-vous que votre fichier kubeconfig est correctement configuré."
        print_info "Exécutez: doctl kubernetes cluster kubeconfig save $CLUSTER_NAME"
        return
    fi
    
    print_info "Nœuds Kubernetes:"
    kubectl get nodes -o wide
    
    echo ""
    print_info "Pods dans le namespace $NAMESPACE:"
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl get pods -n "$NAMESPACE" -o wide
    else
        print_warning "Le namespace $NAMESPACE n'existe pas."
        print_info "Namespaces disponibles:"
        kubectl get namespaces
    fi
    
    echo ""
    print_info "Services dans le namespace $NAMESPACE:"
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl get services -n "$NAMESPACE"
    else
        print_warning "Le namespace $NAMESPACE n'existe pas."
    fi
    
    echo ""
    print_info "Ingress dans le namespace $NAMESPACE:"
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl get ingress -n "$NAMESPACE" 2>/dev/null || print_info "Aucun ingress trouvé."
    fi
    
    echo ""
    print_info "Événements récents dans le namespace $NAMESPACE:"
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -n 20
    fi
}

# Vérification des load balancers Kubernetes
check_load_balancers() {
    if [[ "$KUBECTL_AVAILABLE" == "false" ]]; then
        return
    fi
    
    print_header "Vérification des Load Balancers Kubernetes"
    
    # Vérifier les services de type LoadBalancer
    print_info "Services de type LoadBalancer:"
    kubectl get services --all-namespaces -o wide --field-selector type=LoadBalancer
    
    # Vérifier l'état de l'Ingress NGINX
    print_info "État de l'Ingress NGINX:"
    if kubectl get namespace ingress-nginx &> /dev/null; then
        kubectl get pods -n ingress-nginx
        kubectl get services -n ingress-nginx
    else
        print_warning "Le namespace ingress-nginx n'existe pas. L'Ingress NGINX n'est peut-être pas installé."
    fi
}

# Vérification de l'état des déploiements
check_deployments() {
    if [[ "$KUBECTL_AVAILABLE" == "false" ]]; then
        return
    fi
    
    print_header "Vérification des déploiements"
    
    print_info "Déploiements dans le namespace $NAMESPACE:"
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl get deployments -n "$NAMESPACE" -o wide
        
        # Vérifier les replicas pour chaque déploiement
        echo ""
        print_info "État des replicas:"
        kubectl get deployments -n "$NAMESPACE" -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,AVAILABLE:.status.availableReplicas,READY:.status.readyReplicas
    else
        print_warning "Le namespace $NAMESPACE n'existe pas."
    fi
}

# Afficher les suggestions
show_suggestions() {
    print_header "Suggestions de dépannage"
    
    print_info "Si votre cluster est créé mais que rien ne se passe, vérifiez:"
    
    echo "1. État du cluster: Le cluster peut être encore en cours de provisionnement"
    echo "   - Le provisionnement peut prendre de 5 à 10 minutes"
    echo "   - Vérifiez l'état via le panneau de contrôle DigitalOcean ou avec ce script"
    
    echo ""
    echo "2. Configuration de kubectl:"
    echo "   - Assurez-vous que kubectl est configuré pour utiliser le bon cluster:"
    echo "     doctl kubernetes cluster kubeconfig save $CLUSTER_NAME"
    
    echo ""
    echo "3. Déploiement des applications:"
    echo "   - Vérifiez si vos applications ont été déployées:"
    echo "     kubectl get deployments -n $NAMESPACE"
    echo "   - Vérifiez les logs des pods pour identifier les erreurs:"
    echo "     kubectl logs -n $NAMESPACE [nom-du-pod]"
    
    echo ""
    echo "4. Services et Ingress:"
    echo "   - Vérifiez si les services sont exposés correctement:"
    echo "     kubectl get services -n $NAMESPACE"
    echo "   - Vérifiez si l'Ingress est configuré correctement:"
    echo "     kubectl get ingress -n $NAMESPACE"
    
    echo ""
    echo "5. Problèmes courants:"
    echo "   - Problèmes de DNS: Vérifiez que vos enregistrements DNS pointent vers la bonne IP"
    echo "   - Problèmes de certificats SSL: Vérifiez l'état des certificats"
    echo "   - Problèmes de ressources: Vérifiez l'utilisation des ressources des nœuds"
    
    echo ""
    echo "6. Pour déployer vos applications, exécutez:"
    echo "   cd kubernetes && ./deploy.sh"
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
        -s|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        *)
            print_error "Option inconnue: $1"
            show_help
            ;;
    esac
done

# Exécution principale
check_prerequisites
check_cluster_api

if [[ "$KUBECTL_AVAILABLE" == "true" ]]; then
    check_kubernetes_resources
    check_load_balancers
    check_deployments
fi

show_suggestions

print_header "Diagnostic terminé"
echo "Utilisez ces informations pour résoudre les problèmes avec votre cluster Kubernetes sur DigitalOcean." 
#!/bin/bash

# Variables
NAMESPACE="evil2root-trading"
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

usage() {
    echo "Usage: $0 [start|stop|status|logs|restart]"
    echo ""
    echo "Commandes:"
    echo "  start      Démarre tous les services de l'application"
    echo "  stop       Arrête tous les services de l'application"
    echo "  status     Affiche le statut des pods"
    echo "  logs       Affiche les logs d'un service spécifique"
    echo "  restart    Redémarre tous les services de l'application"
    echo ""
    exit 1
}

check_kubernetes() {
    if ! kubectl --context docker-desktop get nodes &> /dev/null; then
        print_error "Kubernetes n'est pas disponible. Veuillez activer Kubernetes dans Docker Desktop."
        exit 1
    fi
}

check_namespace() {
    if ! kubectl --context docker-desktop get namespace | grep -q $NAMESPACE; then
        print_error "Le namespace '$NAMESPACE' n'existe pas. Veuillez d'abord exécuter ./setup-k8s-local.sh"
        exit 1
    fi
}

start_services() {
    print_header "Démarrage des services"
    
    # Vérifier si les déploiements existent déjà
    if kubectl --context docker-desktop get deployment -n $NAMESPACE 2>/dev/null | grep -q "trading-bot-web"; then
        print_warning "Les services sont déjà déployés. Utilisation de 'kubectl scale' pour les reprendre."
        
        # Reprendre les déploiements (scale up)
        kubectl --context docker-desktop scale deployment --all --replicas=1 -n $NAMESPACE
    else
        print_warning "Les services ne sont pas déployés. Veuillez d'abord exécuter ./setup-k8s-local.sh"
        exit 1
    fi
    
    # Vérifier le statut
    echo "Statut actuel des pods:"
    kubectl --context docker-desktop get pods -n $NAMESPACE
}

stop_services() {
    print_header "Arrêt des services"
    
    # Scale à 0 tous les déploiements pour conserver les configurations
    kubectl --context docker-desktop scale deployment --all --replicas=0 -n $NAMESPACE
    
    echo "Tous les services ont été arrêtés. Les configurations et données persistent."
    echo "Pour redémarrer les services, utilisez: $0 start"
}

show_status() {
    print_header "Statut des services"
    
    echo "Pods:"
    kubectl --context docker-desktop get pods -n $NAMESPACE
    
    echo ""
    echo "Services:"
    kubectl --context docker-desktop get svc -n $NAMESPACE
    
    echo ""
    echo "Déploiements:"
    kubectl --context docker-desktop get deployments -n $NAMESPACE
}

show_logs() {
    print_header "Affichage des logs"
    
    # Liste des services disponibles
    echo "Services disponibles:"
    kubectl --context docker-desktop get pods -n $NAMESPACE -o custom-columns=NAME:.metadata.name
    
    echo ""
    read -p "Entrez le nom du pod pour voir ses logs (ou 'q' pour quitter): " pod_name
    
    if [ "$pod_name" = "q" ]; then
        return
    fi
    
    # Vérifier si le pod existe
    if ! kubectl --context docker-desktop get pod -n $NAMESPACE $pod_name &> /dev/null; then
        print_error "Le pod '$pod_name' n'existe pas."
        return
    fi
    
    # Afficher les logs
    kubectl --context docker-desktop logs -n $NAMESPACE $pod_name -f
}

restart_services() {
    print_header "Redémarrage des services"
    
    # Redémarrer tous les déploiements
    kubectl --context docker-desktop rollout restart deployment -n $NAMESPACE
    
    echo "Tous les services sont en cours de redémarrage."
    echo "Statut actuel:"
    kubectl --context docker-desktop get pods -n $NAMESPACE
}

# Vérifier les arguments
if [ $# -eq 0 ]; then
    usage
fi

# Vérifier que Kubernetes est disponible
check_kubernetes

# Vérifier que le namespace existe
check_namespace

# Traiter la commande
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    restart)
        restart_services
        ;;
    *)
        usage
        ;;
esac

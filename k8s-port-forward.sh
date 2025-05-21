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

check_kubernetes() {
    if ! kubectl --context docker-desktop get nodes &> /dev/null; then
        print_error "Kubernetes n'est pas disponible. Veuillez activer Kubernetes dans Docker Desktop."
        exit 1
    fi
}

print_usage() {
    echo "Usage: $0 [postgres|redis|adminer|all]"
    echo ""
    echo "Options:"
    echo "  postgres    Forward du port PostgreSQL (5432)"
    echo "  redis       Forward du port Redis (6379)"
    echo "  adminer     Forward du port Adminer (8080)"
    echo "  all         Forward de tous les ports"
    echo ""
    exit 1
}

# Vérifier les arguments
if [ $# -eq 0 ]; then
    print_usage
fi

# Vérifier que Kubernetes est disponible
check_kubernetes

case "$1" in
    postgres)
        print_header "Port-forward pour PostgreSQL"
        kubectl --context docker-desktop port-forward -n $NAMESPACE svc/postgres 5432:5432
        ;;
    redis)
        print_header "Port-forward pour Redis"
        kubectl --context docker-desktop port-forward -n $NAMESPACE svc/redis 6379:6379
        ;;
    adminer)
        print_header "Port-forward pour Adminer"
        kubectl --context docker-desktop port-forward -n $NAMESPACE svc/adminer 8080:8080
        echo "Adminer est accessible à l'adresse: http://localhost:8080"
        echo "  - Système: PostgreSQL"
        echo "  - Serveur: postgres"
        echo "  - Utilisateur: postgres"
        echo "  - Mot de passe: postgres_password"
        echo "  - Base de données: trading_bot"
        ;;
    all)
        print_header "Port-forward pour tous les services"
        print_warning "Cette option va ouvrir plusieurs terminaux"
        
        # Forward PostgreSQL
        osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && kubectl --context docker-desktop port-forward -n evil2root-trading svc/postgres 5432:5432"'
        
        # Forward Redis
        osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && kubectl --context docker-desktop port-forward -n evil2root-trading svc/redis 6379:6379"'
        
        # Forward Adminer
        osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && kubectl --context docker-desktop port-forward -n evil2root-trading svc/adminer 8080:8080"'
        
        echo "Tous les services sont disponibles en port-forward:"
        echo "- PostgreSQL: localhost:5432"
        echo "- Redis: localhost:6379"
        echo "- Adminer: http://localhost:8080"
        echo ""
        echo "Pour Adminer:"
        echo "  - Système: PostgreSQL"
        echo "  - Serveur: postgres"
        echo "  - Utilisateur: postgres"
        echo "  - Mot de passe: postgres_password"
        echo "  - Base de données: trading_bot"
        ;;
    *)
        print_usage
        ;;
esac

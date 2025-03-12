#!/bin/bash

# Variables
NAMESPACE="evil2root-trading"
KUBE_CONTEXT=$(kubectl config current-context)

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
    
    # Vérifier si kubectl est installé
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl n'est pas installé. Veuillez l'installer et réessayer."
        exit 1
    fi
    
    # Vérifier si kustomize est installé
    if ! command -v kustomize &> /dev/null; then
        print_warning "kustomize n'est pas installé. Utilisation de kubectl apply -k à la place."
    fi
    
    # Vérifier la connexion au cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Impossible de se connecter au cluster Kubernetes. Vérifiez votre contexte."
        exit 1
    fi
    
    echo "Contexte Kubernetes actuel: $KUBE_CONTEXT"
    echo "Êtes-vous sûr de vouloir déployer sur ce contexte? (o/n)"
    read -r confirm
    if [[ "$confirm" != "o" ]]; then
        echo "Déploiement annulé."
        exit 0
    fi
}

create_namespace() {
    print_header "Création du namespace $NAMESPACE"
    kubectl apply -f namespace.yaml
}

check_gpu_support() {
    print_header "Vérification du support GPU"
    if kubectl get nodes -o=jsonpath='{.items[*].status.capacity.nvidia\.com/gpu}' | grep -q "[0-9]"; then
        echo "Support GPU détecté sur le cluster."
    else
        print_warning "Aucun support GPU détecté sur le cluster. Certains pods pourraient ne pas démarrer correctement."
        echo "Voulez-vous continuer sans GPU? (o/n)"
        read -r confirm
        if [[ "$confirm" != "o" ]]; then
            echo "Déploiement annulé."
            exit 0
        fi
    fi
}

apply_resources() {
    print_header "Déploiement des ressources avec kustomize"
    
    if command -v kustomize &> /dev/null; then
        kustomize build . | kubectl apply -f -
    else
        kubectl apply -k .
    fi
}

wait_for_deployments() {
    print_header "Attente du déploiement des ressources"
    
    echo "Attente des déploiements de base (DB, Redis)..."
    kubectl -n $NAMESPACE wait --for=condition=available --timeout=300s deployment/postgres deployment/redis
    
    echo "Attente des déploiements principaux..."
    kubectl -n $NAMESPACE wait --for=condition=available --timeout=300s deployment/trading-bot-web deployment/analysis-bot deployment/market-scheduler deployment/prometheus deployment/grafana
}

print_access_info() {
    print_header "Informations d'accès"
    
    echo "Interface Web: https://trading.example.com"
    echo "Grafana: https://grafana.trading.example.com"
    echo "Adminer: https://adminer.trading.example.com"
    
    # Récupérer les mots de passe
    echo -e "\nMots de passe (pour développement uniquement - changez-les en production!):"
    echo "Base de données: $(kubectl -n $NAMESPACE get secret trading-bot-secrets -o jsonpath='{.data.DB_PASSWORD}' | base64 --decode)"
    echo "Grafana: $(kubectl -n $NAMESPACE get secret trading-bot-secrets -o jsonpath='{.data.GRAFANA_ADMIN_PASSWORD}' | base64 --decode)"
}

# Exécution principale
check_prerequisites
check_gpu_support
create_namespace
apply_resources
wait_for_deployments
print_access_info

print_header "Déploiement terminé"
echo "Le trading bot a été déployé avec succès sur Kubernetes!" 
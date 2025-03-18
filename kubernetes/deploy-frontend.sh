#!/bin/bash

# Script de déploiement du frontend pour EVIL2ROOT Trading Bot
# Ce script construit l'image Docker du frontend, la pousse vers le registre
# et déploie l'application sur Kubernetes

set -e

# Variables
REGISTRY="registry.digitalocean.com/evil2root-registry"
IMAGE_NAME="evil2root-frontend"
IMAGE_TAG="latest"
NAMESPACE="evil2root-trading"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="${ROOT_DIR}/frontend"

# Vérifier que kubectl est disponible
if ! command -v kubectl &> /dev/null; then
    echo "kubectl n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Vérifier que docker est disponible
if ! command -v docker &> /dev/null; then
    echo "docker n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Fonction pour afficher les logs
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Construction de l'image Docker
build_docker_image() {
    log "Construction de l'image Docker du frontend..."
    docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -f ${FRONTEND_DIR}/Dockerfile ${FRONTEND_DIR}
}

# Pousser l'image vers le registre
push_docker_image() {
    log "Poussée de l'image vers le registre ${REGISTRY}..."
    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
}

# Déploiement sur Kubernetes
deploy_to_kubernetes() {
    log "Déploiement sur Kubernetes..."
    
    # Appliquer le namespace s'il n'existe pas déjà
    kubectl apply -f ${SCRIPT_DIR}/namespace.yaml
    
    # Appliquer le déploiement du frontend
    kubectl apply -f ${SCRIPT_DIR}/frontend-deployment.yaml
    
    # Attendre que le déploiement soit prêt
    kubectl rollout status deployment/trading-bot-frontend -n ${NAMESPACE} --timeout=300s
}

# Vérifier l'état du déploiement
check_deployment() {
    log "Vérification de l'état du déploiement..."
    kubectl get pods -n ${NAMESPACE} -l app=trading-bot,component=frontend
    
    log "Services:"
    kubectl get svc -n ${NAMESPACE} -l app=trading-bot,component=frontend
    
    log "Ingress:"
    kubectl get ingress -n ${NAMESPACE} trading-bot-frontend-ingress
}

# Menu principal
main() {
    log "Début du déploiement du frontend EVIL2ROOT Trading Bot..."
    
    # Demander confirmation
    read -p "Voulez-vous construire et déployer le frontend? (o/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Oo]$ ]]; then
        log "Opération annulée."
        exit 0
    fi
    
    # Exécuter les étapes
    build_docker_image
    push_docker_image
    deploy_to_kubernetes
    check_deployment
    
    log "Déploiement terminé avec succès!"
    log "L'interface utilisateur sera accessible à l'adresse: https://ui.trading.example.com (après configuration DNS)"
}

# Exécuter le script
main "$@" 
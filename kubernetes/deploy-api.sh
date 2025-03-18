#!/bin/bash

# Script de déploiement de l'API pour EVIL2ROOT Trading Bot
# Ce script construit l'image Docker de l'API, la pousse vers le registre
# et déploie l'application sur Kubernetes

set -e

# Variables
REGISTRY="registry.digitalocean.com/evil2root-registry"
IMAGE_NAME="evil2root-api"
IMAGE_TAG="latest"
NAMESPACE="evil2root-trading"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
API_DIR="${ROOT_DIR}/src/api"

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
    log "Construction de l'image Docker de l'API..."
    docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -f ${ROOT_DIR}/Dockerfile.api ${ROOT_DIR}
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
    
    # Appliquer le déploiement de l'API
    kubectl apply -f ${SCRIPT_DIR}/api-deployment.yaml
    
    # Attendre que le déploiement soit prêt
    kubectl rollout status deployment/trading-bot-api -n ${NAMESPACE} --timeout=300s
}

# Vérifier l'état du déploiement
check_deployment() {
    log "Vérification de l'état du déploiement..."
    kubectl get pods -n ${NAMESPACE} -l app=trading-bot,component=api
    
    log "Services:"
    kubectl get svc -n ${NAMESPACE} -l app=trading-bot,component=api
    
    log "Ingress:"
    kubectl get ingress -n ${NAMESPACE} trading-bot-api-ingress
}

# Menu principal
main() {
    log "Début du déploiement de l'API EVIL2ROOT Trading Bot..."
    
    # Demander confirmation
    read -p "Voulez-vous construire et déployer l'API? (o/n): " -n 1 -r
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
    log "L'API sera accessible à l'adresse: https://api.trading.example.com (après configuration DNS)"
}

# Exécuter le script
main "$@" 
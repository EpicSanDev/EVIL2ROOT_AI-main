#!/bin/bash

# Variables
NAMESPACE="evil2root-trading"
KUBE_CONTEXT=$(kubectl config current-context)
TIMESTAMP=$(date +%s)

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

check_cluster() {
    print_header "Vérification de la connexion au cluster"
    
    # Vérifier si kubectl est installé
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl n'est pas installé. Veuillez l'installer et réessayer."
        exit 1
    fi
    
    # Vérifier la connexion au cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Impossible de se connecter au cluster Kubernetes. Vérifiez votre contexte."
        exit 1
    fi
    
    # Vérifier si le namespace existe
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        print_error "Le namespace $NAMESPACE n'existe pas. Assurez-vous que le cluster est correctement configuré."
        exit 1
    fi
    
    echo "Contexte Kubernetes actuel: $KUBE_CONTEXT"
    echo "Êtes-vous sûr de vouloir appliquer les mises à jour sur ce contexte? (o/n)"
    read -r confirm
    if [[ "$confirm" != "o" ]]; then
        echo "Mise à jour annulée."
        exit 0
    fi
}

generate_checksums() {
    print_header "Génération des checksums pour les configurations"
    
    CONFIG_CHECKSUM=$(kubectl create configmap trading-bot-config --from-file=configmap.yaml --dry-run=client -o yaml | sha256sum | cut -d' ' -f1)
    SECRETS_CHECKSUM=$(kubectl create secret generic trading-bot-secrets --from-file=secrets.yaml --dry-run=client -o yaml | sha256sum | cut -d' ' -f1)
    
    echo "Checksum de configuration: $CONFIG_CHECKSUM"
    echo "Checksum de secrets: $SECRETS_CHECKSUM"
}

prepare_update_files() {
    print_header "Préparation des fichiers de mise à jour"
    
    # Créer une copie temporaire des fichiers de déploiement
    mkdir -p temp_updates
    cp *.yaml temp_updates/
    
    # Remplacer les variables dans les fichiers
    find temp_updates -type f -name "*.yaml" -exec sed -i "s/\${CHECKSUM_CONFIG}/$CONFIG_CHECKSUM/g" {} \;
    find temp_updates -type f -name "*.yaml" -exec sed -i "s/\${CHECKSUM_SECRETS}/$SECRETS_CHECKSUM/g" {} \;
    find temp_updates -type f -name "*.yaml" -exec sed -i "s/\${TIMESTAMP}/$TIMESTAMP/g" {} \;
    
    echo "Fichiers de déploiement préparés avec succès."
}

apply_updates() {
    print_header "Application des mises à jour"
    
    # Appliquer d'abord les changements de configuration
    echo "Mise à jour des ressources de configuration (configmap, secrets)..."
    kubectl apply -f temp_updates/configmap.yaml
    kubectl apply -f temp_updates/secrets.yaml
    
    # Appliquer les mises à jour des déploiements principaux
    echo "Mise à jour des déploiements principaux..."
    kubectl apply -f temp_updates/web-deployment.yaml
    kubectl apply -f temp_updates/analysis-bot-deployment.yaml
    kubectl apply -f temp_updates/market-scheduler-deployment.yaml
    
    # Appliquer les mises à jour des jobs
    echo "Mise à jour des configurations de jobs..."
    kubectl apply -f temp_updates/train-analyze-job.yaml
    
    # Appliquer les mises à jour de scaling et sécurité
    echo "Mise à jour des configurations de scaling et sécurité..."
    kubectl apply -f temp_updates/hpa.yaml
    kubectl apply -f temp_updates/network-policies.yaml
    kubectl apply -f temp_updates/pod-disruption-budgets.yaml
    
    # Nettoyage des fichiers temporaires
    echo "Nettoyage des fichiers temporaires..."
    rm -rf temp_updates
}

restart_deployments() {
    print_header "Redémarrage des déploiements"
    
    echo "Voulez-vous forcer un redémarrage progressif des pods pour appliquer les changements immédiatement? (o/n)"
    read -r restart
    
    if [[ "$restart" == "o" ]]; then
        echo "Redémarrage du déploiement web..."
        kubectl rollout restart deployment/trading-bot-web -n $NAMESPACE
        
        echo "Redémarrage du déploiement analysis-bot..."
        kubectl rollout restart deployment/analysis-bot -n $NAMESPACE
        
        echo "Redémarrage du déploiement market-scheduler..."
        kubectl rollout restart deployment/market-scheduler -n $NAMESPACE
        
        echo "Attente de la fin des redémarrages..."
        kubectl rollout status deployment/trading-bot-web -n $NAMESPACE
        kubectl rollout status deployment/analysis-bot -n $NAMESPACE
        kubectl rollout status deployment/market-scheduler -n $NAMESPACE
    else
        echo "Les pods seront redémarrés progressivement par Kubernetes en fonction des changements détectés."
    fi
}

check_pod_status() {
    print_header "Vérification du statut des pods"
    
    kubectl get pods -n $NAMESPACE
}

print_completion() {
    print_header "Mise à jour terminée"
    
    echo "Les mises à jour ont été appliquées avec succès sur le cluster!"
    echo "Pour surveiller le statut des déploiements:"
    echo "  kubectl rollout status deployment/trading-bot-web -n $NAMESPACE"
    
    echo -e "\nPour vérifier les logs des pods mis à jour:"
    echo "  kubectl logs -f deployment/trading-bot-web -n $NAMESPACE"
}

# Exécution principale
check_cluster
generate_checksums
prepare_update_files
apply_updates
restart_deployments
check_pod_status
print_completion 
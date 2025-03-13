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

    # Vérifier si yq est installé (pour manipuler YAML)
    if ! command -v yq &> /dev/null; then
        print_warning "yq n'est pas installé. Il est recommandé de l'installer pour les manipulations YAML."
        echo "Voulez-vous continuer sans yq? (o/n)"
        read -r confirm
        if [[ "$confirm" != "o" ]]; then
            echo "Déploiement annulé."
            exit 0
        fi
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

generate_checksums() {
    print_header "Génération des checksums pour les configurations"
    
    CONFIG_CHECKSUM=$(kubectl create configmap trading-bot-config --from-file=configmap.yaml --dry-run=client -o yaml | sha256sum | cut -d' ' -f1)
    SECRETS_CHECKSUM=$(kubectl create secret generic trading-bot-secrets --from-file=secrets.yaml --dry-run=client -o yaml | sha256sum | cut -d' ' -f1)
    
    echo "Checksum de configuration: $CONFIG_CHECKSUM"
    echo "Checksum de secrets: $SECRETS_CHECKSUM"
}

prepare_deployment_files() {
    print_header "Préparation des fichiers de déploiement"
    
    # Créer une copie temporaire des fichiers de déploiement
    mkdir -p temp_k8s
    cp *.yaml temp_k8s/
    
    # Remplacer les variables dans les fichiers
    find temp_k8s -type f -name "*.yaml" -exec sed -i "s/\${CHECKSUM_CONFIG}/$CONFIG_CHECKSUM/g" {} \;
    find temp_k8s -type f -name "*.yaml" -exec sed -i "s/\${CHECKSUM_SECRETS}/$SECRETS_CHECKSUM/g" {} \;
    find temp_k8s -type f -name "*.yaml" -exec sed -i "s/\${TIMESTAMP}/$TIMESTAMP/g" {} \;
    
    echo "Fichiers de déploiement préparés avec succès."
}

apply_resources() {
    print_header "Déploiement des ressources"
    
    # Appliquer d'abord les ressources de base
    echo "Déploiement des ressources de base (namespace, configmap, secrets, stockage)..."
    kubectl apply -f temp_k8s/namespace.yaml
    kubectl apply -f temp_k8s/configmap.yaml
    kubectl apply -f temp_k8s/secrets.yaml
    kubectl apply -f temp_k8s/storage.yaml
    
    echo "Attente de la création des PVCs..."
    sleep 10
    
    # Appliquer les déploiements d'infrastructure
    echo "Déploiement des bases de données et du cache..."
    kubectl apply -f temp_k8s/db-deployment.yaml
    kubectl apply -f temp_k8s/redis-deployment.yaml
    
    echo "Attente du démarrage des databases..."
    kubectl -n $NAMESPACE wait --for=condition=available --timeout=300s deployment/postgres deployment/redis || true
    
    # Appliquer les déploiements principaux
    echo "Déploiement des applications principales..."
    kubectl apply -f temp_k8s/web-deployment.yaml
    kubectl apply -f temp_k8s/analysis-bot-deployment.yaml
    kubectl apply -f temp_k8s/market-scheduler-deployment.yaml
    
    # Appliquer les autres ressources
    echo "Déploiement des ressources de surveillance..."
    kubectl apply -f temp_k8s/monitoring-deployment.yaml
    kubectl apply -f temp_k8s/adminer-deployment.yaml
    
    echo "Déploiement des ressources de scaling et de sécurité..."
    kubectl apply -f temp_k8s/hpa.yaml
    kubectl apply -f temp_k8s/network-policies.yaml
    kubectl apply -f temp_k8s/pod-disruption-budgets.yaml
    
    # Nettoyage des fichiers temporaires
    echo "Nettoyage des fichiers temporaires..."
    rm -rf temp_k8s
}

wait_for_deployments() {
    print_header "Attente du déploiement des ressources"
    
    echo "Attente des déploiements de base (DB, Redis)..."
    kubectl -n $NAMESPACE wait --for=condition=available --timeout=300s deployment/postgres deployment/redis || true
    
    echo "Attente des déploiements principaux..."
    kubectl -n $NAMESPACE wait --for=condition=available --timeout=300s deployment/trading-bot-web deployment/analysis-bot deployment/market-scheduler || true
    
    echo "Attente des déploiements de surveillance..."
    kubectl -n $NAMESPACE wait --for=condition=available --timeout=300s deployment/prometheus deployment/grafana || true
}

check_deployment_status() {
    print_header "Vérification du statut des déploiements"
    
    # Vérifier le statut des pods
    kubectl -n $NAMESPACE get pods
    
    # Vérifier le statut des services
    kubectl -n $NAMESPACE get svc
    
    # Vérifier le statut des ingress
    kubectl -n $NAMESPACE get ingress
}

print_access_info() {
    print_header "Informations d'accès"
    
    echo "Interface Web: https://trading.example.com"
    echo "Grafana: https://grafana.trading.example.com"
    echo "Adminer: https://adminer.trading.example.com"
    
    # Récupérer les mots de passe
    echo -e "\nMots de passe (pour développement uniquement - changez-les en production!):"
    echo "Base de données: $(kubectl -n $NAMESPACE get secret trading-bot-secrets -o jsonpath='{.data.DB_PASSWORD}' | base64 --decode 2>/dev/null || echo "Non disponible")"
    echo "Grafana: $(kubectl -n $NAMESPACE get secret trading-bot-secrets -o jsonpath='{.data.GRAFANA_ADMIN_PASSWORD}' | base64 --decode 2>/dev/null || echo "Non disponible")"
}

print_deployment_tips() {
    print_header "Conseils de déploiement"
    
    echo "1. Pour surveiller vos applications, accédez à Grafana via l'URL indiquée ci-dessus."
    echo "2. Pour exécuter un job d'entraînement et d'analyse manuellement:"
    echo "   kubectl -n $NAMESPACE create -f kubernetes/train-analyze-job.yaml"
    echo "3. Pour consulter les logs d'un pod spécifique:"
    echo "   kubectl -n $NAMESPACE logs -f deployment/trading-bot-web"
    echo "4. Pour mettre à jour l'application vers une nouvelle version:"
    echo "   kubectl -n $NAMESPACE set image deployment/trading-bot-web web=registry.digitalocean.com/epicsandev/evil2root-ai:nouvelle-version"
}

# Exécution principale
check_prerequisites
check_gpu_support
create_namespace
generate_checksums
prepare_deployment_files
apply_resources
wait_for_deployments
check_deployment_status
print_access_info
print_deployment_tips

print_header "Déploiement terminé"
echo "Le trading bot a été déployé avec succès sur Kubernetes!" 
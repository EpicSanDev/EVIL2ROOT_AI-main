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

check_docker_desktop() {
    print_header "Vérification de Docker Desktop"
    
    # Vérifier si Docker est installé et en cours d'exécution
    if ! docker info &> /dev/null; then
        print_error "Docker n'est pas en cours d'exécution. Veuillez démarrer Docker Desktop."
        exit 1
    fi
    
    echo "Docker Desktop est en cours d'exécution."
}

check_kubernetes_status() {
    print_header "Vérification du statut de Kubernetes"
    
    # Passer au contexte docker-desktop explicitement si disponible
    if kubectl config get-contexts | grep -q "docker-desktop"; then
        kubectl config use-context docker-desktop
        echo "Basculé vers le contexte: docker-desktop"
    fi
    
    # Vérifier si Kubernetes est activé dans Docker Desktop
    if ! kubectl get nodes &> /dev/null; then
        print_warning "Kubernetes ne semble pas être activé dans Docker Desktop."
        print_warning "Veuillez l'activer manuellement dans les paramètres de Docker Desktop:"
        print_warning "1. Ouvrez Docker Desktop"
        print_warning "2. Cliquez sur l'icône d'engrenage (Paramètres)"
        print_warning "3. Allez dans la section 'Kubernetes'"
        print_warning "4. Cochez 'Enable Kubernetes'"
        print_warning "5. Cliquez sur 'Apply & Restart'"
        print_warning "6. Attendez que Kubernetes démarre"
        
        read -p "Appuyez sur Entrée une fois que Kubernetes est activé et prêt... " -r
        
        # Vérifier à nouveau
        if ! kubectl get nodes &> /dev/null; then
            print_error "Kubernetes n'est toujours pas disponible. Veuillez résoudre le problème et réessayer."
            exit 1
        fi
    fi
    
    echo "Kubernetes est activé et prêt."
}

create_namespace() {
    print_header "Création du namespace"
    
    # Vérifier si le namespace existe déjà
    if kubectl --context docker-desktop get namespace | grep -q $NAMESPACE; then
        echo "Le namespace '$NAMESPACE' existe déjà."
    else
        # Créer le namespace
        kubectl --context docker-desktop create namespace $NAMESPACE
        echo "Namespace '$NAMESPACE' créé avec succès."
    fi
}

create_secrets() {
    print_header "Création des secrets"

    # Vérifier si le répertoire secrets existe et le créer si nécessaire
    if [ ! -d "./secrets" ]; then
        print_warning "Le répertoire secrets n'existe pas. Création du répertoire..."
        mkdir -p ./secrets
    fi
    
    # Générer les fichiers de secrets s'ils n'existent pas
    if [ ! -f "./secrets/db_user.txt" ]; then
        echo "postgres" > ./secrets/db_user.txt
        echo "Fichier db_user.txt créé."
    fi
    
    if [ ! -f "./secrets/db_password.txt" ]; then
        echo "postgres_password" > ./secrets/db_password.txt
        echo "Fichier db_password.txt créé."
    fi
    
    if [ ! -f "./secrets/secret_key.txt" ]; then
        echo "secret_key_$(openssl rand -hex 16)" > ./secrets/secret_key.txt
        echo "Fichier secret_key.txt créé."
    fi
    
    if [ ! -f "./secrets/admin_password.txt" ]; then
        echo "admin_password" > ./secrets/admin_password.txt
        echo "Fichier admin_password.txt créé."
    fi
    
    if [ ! -f "./secrets/telegram_token.txt" ]; then
        echo "dummy_token" > ./secrets/telegram_token.txt
        echo "Fichier telegram_token.txt créé."
    fi
    
    if [ ! -f "./secrets/finnhub_api_key.txt" ]; then
        echo "dummy_key" > ./secrets/finnhub_api_key.txt
        echo "Fichier finnhub_api_key.txt créé."
    fi
    
    if [ ! -f "./secrets/openrouter_api_key.txt" ]; then
        echo "dummy_key" > ./secrets/openrouter_api_key.txt
        echo "Fichier openrouter_api_key.txt créé."
    fi
    
    if [ ! -f "./secrets/coinbase_api_key.txt" ]; then
        echo "dummy_key" > ./secrets/coinbase_api_key.txt
        echo "Fichier coinbase_api_key.txt créé."
    fi
    
    if [ ! -f "./secrets/coinbase_webhook_secret.txt" ]; then
        echo "dummy_secret" > ./secrets/coinbase_webhook_secret.txt
        echo "Fichier coinbase_webhook_secret.txt créé."
    fi

    # Vérifier que tous les fichiers existent maintenant
    if [ -f "./secrets/db_user.txt" ] && 
       [ -f "./secrets/db_password.txt" ] && 
       [ -f "./secrets/secret_key.txt" ] && 
       [ -f "./secrets/admin_password.txt" ] && 
       [ -f "./secrets/telegram_token.txt" ] && 
       [ -f "./secrets/finnhub_api_key.txt" ] && 
       [ -f "./secrets/openrouter_api_key.txt" ] && 
       [ -f "./secrets/coinbase_api_key.txt" ] && 
       [ -f "./secrets/coinbase_webhook_secret.txt" ]; then
        
        # Créer les secrets Kubernetes
        echo "Création du secret Kubernetes 'trading-bot-secrets'..."
        kubectl --context docker-desktop -n $NAMESPACE create secret generic trading-bot-secrets \
            --from-file=./secrets/db_user.txt \
            --from-file=./secrets/db_password.txt \
            --from-file=./secrets/secret_key.txt \
            --from-file=./secrets/admin_password.txt \
            --from-file=./secrets/telegram_token.txt \
            --from-file=./secrets/finnhub_api_key.txt \
            --from-file=./secrets/openrouter_api_key.txt \
            --from-file=./secrets/coinbase_api_key.txt \
            --from-file=./secrets/coinbase_webhook_secret.txt \
            --dry-run=client -o yaml | kubectl --context docker-desktop -n $NAMESPACE apply -f -
    else
        print_error "Certains fichiers de secrets n'ont pas pu être créés."
        exit 1
    fi
}

create_configmap() {
    print_header "Création du ConfigMap"
    
    # Créer un ConfigMap simple pour le développement local
    cat <<EOF | kubectl --context docker-desktop -n $NAMESPACE apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-bot-config
data:
  DB_HOST: "postgres"
  DB_PORT: "5432"
  DB_NAME: "trading_bot"
  REDIS_HOST: "redis"
  REDIS_PORT: "6379"
  APP_ENV: "development"
  LOG_LEVEL: "DEBUG"
EOF
}

deploy_app() {
    print_header "Déploiement de l'application avec Kustomize"
    
    # Vérifier si kustomization.yaml existe dans le dossier kubernetes
    if [ ! -f "./kubernetes/kustomization.yaml" ]; then
        print_error "Le fichier kustomization.yaml n'existe pas dans le dossier kubernetes."
        exit 1
    fi
    
    # Appliquer la configuration Kustomize
    kubectl --context docker-desktop apply -k ./kubernetes -n $NAMESPACE
    
    print_warning "Le déploiement de l'application peut prendre quelques minutes..."
}

check_deployment_status() {
    print_header "Vérification du statut du déploiement"
    
    echo "Attente du démarrage des pods..."
    sleep 10
    
    kubectl --context docker-desktop get pods -n $NAMESPACE
    
    print_warning "Utilisez la commande suivante pour surveiller le statut du déploiement:"
    echo "kubectl --context docker-desktop get pods -n $NAMESPACE -w"
}

print_application_access() {
    print_header "Accès à l'application"
    
    # Récupérer l'adresse du service
    local service_name=$(kubectl --context docker-desktop get svc -n $NAMESPACE -l app=trading-bot-web -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -z "$service_name" ]; then
        print_warning "Service trading-bot-web non trouvé. Utilisation de port-forward..."
        echo "Pour accéder à l'application, exécutez la commande suivante:"
        echo "kubectl --context docker-desktop port-forward -n $NAMESPACE deployment/trading-bot-web 8080:8080"
        echo "Puis accédez à http://localhost:8080 dans votre navigateur."
    else
        local service_port=$(kubectl --context docker-desktop get svc -n $NAMESPACE $service_name -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null)
        
        if [ -n "$service_port" ]; then
            echo "L'application est accessible à l'adresse: http://localhost:$service_port"
        else
            echo "Pour accéder à l'application, exécutez la commande suivante:"
            echo "kubectl --context docker-desktop port-forward -n $NAMESPACE svc/$service_name 8080:80"
            echo "Puis accédez à http://localhost:8080 dans votre navigateur."
        fi
    fi
}

# Exécution principale
check_docker_desktop
check_kubernetes_status
create_namespace
create_secrets
create_configmap
deploy_app
check_deployment_status
print_application_access

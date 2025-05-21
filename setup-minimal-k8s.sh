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

build_local_images() {
    print_header "Construction des images Docker locales"
    
    # Construire l'image principale
    echo "Construction de l'image evil2root-main..."
    docker build -t evil2root-main:local -f Dockerfile .
    
    # Construire l'image API
    echo "Construction de l'image evil2root-api..."
    docker build -t evil2root-api:local -f Dockerfile.api .
    
    echo "Les images Docker ont été construites avec succès."
}

deploy_infrastructure() {
    print_header "Déploiement de l'infrastructure de base"
    
    # Vérifier si le namespace existe déjà
    if ! kubectl --context docker-desktop get namespace | grep -q $NAMESPACE; then
        echo "Création du namespace '$NAMESPACE'..."
        kubectl --context docker-desktop create namespace $NAMESPACE
    fi
    
    # Appliquer la configuration minimale
    kubectl --context docker-desktop apply -f k8s-minimal.yaml
    
    echo "Infrastructure de base déployée avec succès."
}

deploy_services() {
    print_header "Déploiement des services simplifiés"
    
    # Déployer une base de données PostgreSQL
    cat <<EOF | kubectl --context docker-desktop apply -n $NAMESPACE -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  labels:
    app: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: db_password.txt
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: db_user.txt
        - name: POSTGRES_DB
          value: trading_bot
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi"
            cpu: "250m"
      volumes:
      - name: postgres-data
        persistentVolumeClaim:
          claimName: postgres-data
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
EOF

    # Déployer Redis
    cat <<EOF | kubectl --context docker-desktop apply -n $NAMESPACE -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  labels:
    app: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:6
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
        resources:
          limits:
            memory: "512Mi"
            cpu: "300m"
          requests:
            memory: "256Mi"
            cpu: "100m"
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-data
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF

    # Déployer Adminer (interface d'administration pour la base de données)
    cat <<EOF | kubectl --context docker-desktop apply -n $NAMESPACE -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adminer
  labels:
    app: adminer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: adminer
  template:
    metadata:
      labels:
        app: adminer
    spec:
      containers:
      - name: adminer
        image: adminer:latest
        ports:
        - containerPort: 8080
        env:
        - name: ADMINER_DEFAULT_SERVER
          value: postgres
        resources:
          limits:
            memory: "256Mi"
            cpu: "300m"
          requests:
            memory: "128Mi"
            cpu: "100m"
---
apiVersion: v1
kind: Service
metadata:
  name: adminer
spec:
  selector:
    app: adminer
  ports:
  - port: 8080
    targetPort: 8080
EOF

    echo "Services de base déployés avec succès."
}

check_deployment_status() {
    print_header "Vérification du statut du déploiement"
    
    echo "Attente du démarrage des pods..."
    sleep 10
    
    kubectl --context docker-desktop get pods -n $NAMESPACE
    
    print_warning "Utilisez la commande suivante pour surveiller le statut du déploiement:"
    echo "kubectl --context docker-desktop get pods -n $NAMESPACE -w"
}

print_access_instructions() {
    print_header "Accès aux services"
    
    echo "Pour accéder à la base de données PostgreSQL depuis votre application locale:"
    echo "- Hôte: localhost"
    echo "- Port: Utilisez la commande suivante pour faire un port-forward:"
    echo "  kubectl --context docker-desktop port-forward -n $NAMESPACE svc/postgres 5432:5432"
    echo ""
    
    echo "Pour accéder à Redis depuis votre application locale:"
    echo "- Hôte: localhost"
    echo "- Port: Utilisez la commande suivante pour faire un port-forward:"
    echo "  kubectl --context docker-desktop port-forward -n $NAMESPACE svc/redis 6379:6379"
    echo ""
    
    echo "Pour accéder à l'interface Adminer (gestion de la base de données):"
    echo "- Utilisez la commande suivante pour faire un port-forward:"
    echo "  kubectl --context docker-desktop port-forward -n $NAMESPACE svc/adminer 8080:8080"
    echo "- Puis accédez à http://localhost:8080 dans votre navigateur"
    echo "  - Système: PostgreSQL"
    echo "  - Serveur: postgres"
    echo "  - Utilisateur: postgres (ou la valeur de db_user.txt)"
    echo "  - Mot de passe: postgres_password (ou la valeur de db_password.txt)"
    echo "  - Base de données: trading_bot"
    echo ""
    
    echo "Pour arrêter/démarrer les services, utilisez le script k8s-control.sh:"
    echo "  ./k8s-control.sh start|stop|status|logs|restart"
}

# Exécution principale
check_docker_desktop
check_kubernetes_status
#build_local_images  # Commenté pour l'instant, à activer si vous souhaitez construire vos propres images
deploy_infrastructure
deploy_services
check_deployment_status
print_access_instructions

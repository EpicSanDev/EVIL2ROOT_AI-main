#!/bin/bash

# Variables
NAMESPACE="evil2root-trading"
KUBE_CONTEXT="kind-evil2root-trading"
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
    
    # Vérifier la connexion au cluster
    if ! kubectl --context $KUBE_CONTEXT cluster-info &> /dev/null; then
        print_error "Impossible de se connecter au cluster Kubernetes. Vérifiez votre contexte."
        exit 1
    fi
    
    echo "Contexte Kubernetes actuel: $KUBE_CONTEXT"
}

create_secrets() {
    print_header "Création des secrets"

    # Vérifier si le répertoire secrets existe
    if [ ! -d "./secrets" ]; then
        print_warning "Le répertoire secrets n'existe pas. Création de secrets par défaut pour le développement."
        mkdir -p ./secrets
        
        # Générer des secrets par défaut pour le développement
        echo "postgres" > ./secrets/db_user.txt
        echo "postgres_password" > ./secrets/db_password.txt
        echo "secret_key_$(openssl rand -hex 16)" > ./secrets/secret_key.txt
        echo "admin_password" > ./secrets/admin_password.txt
        echo "dummy_token" > ./secrets/telegram_token.txt
        echo "dummy_key" > ./secrets/finnhub_api_key.txt
        echo "dummy_key" > ./secrets/openrouter_api_key.txt
        echo "dummy_key" > ./secrets/coinbase_api_key.txt
        echo "dummy_secret" > ./secrets/coinbase_webhook_secret.txt
    fi

    # Créer les secrets Kubernetes
    kubectl --context $KUBE_CONTEXT -n $NAMESPACE create secret generic trading-bot-secrets \
        --from-file=./secrets/db_user.txt \
        --from-file=./secrets/db_password.txt \
        --from-file=./secrets/secret_key.txt \
        --from-file=./secrets/admin_password.txt \
        --from-file=./secrets/telegram_token.txt \
        --from-file=./secrets/finnhub_api_key.txt \
        --from-file=./secrets/openrouter_api_key.txt \
        --from-file=./secrets/coinbase_api_key.txt \
        --from-file=./secrets/coinbase_webhook_secret.txt \
        --dry-run=client -o yaml | kubectl --context $KUBE_CONTEXT -n $NAMESPACE apply -f -
}

create_configmap() {
    print_header "Création du ConfigMap"
    
    # Créer un ConfigMap simple pour le développement local
    cat <<EOF | kubectl --context $KUBE_CONTEXT -n $NAMESPACE apply -f -
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

deploy_storage() {
    print_header "Déploiement du stockage"
    
    # Créer les volumes persistants
    cat <<EOF | kubectl --context $KUBE_CONTEXT -n $NAMESPACE apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: app-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF
}

deploy_infrastructure() {
    print_header "Déploiement de l'infrastructure (PostgreSQL et Redis)"
    
    # Déployer PostgreSQL
    cat <<EOF | kubectl --context $KUBE_CONTEXT -n $NAMESPACE apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
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
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: db_user.txt
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: db_password.txt
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: DB_NAME
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
    cat <<EOF | kubectl --context $KUBE_CONTEXT -n $NAMESPACE apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
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
}

deploy_application() {
    print_header "Déploiement de l'application"
    
    # Construire l'image Docker localement
    docker build -t evil2root/trading-bot:local .
    
    # Charger l'image dans le cluster Kind
    kind load docker-image evil2root/trading-bot:local --name evil2root-trading
    
    # Déployer l'application web
    cat <<EOF | kubectl --context $KUBE_CONTEXT -n $NAMESPACE apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-bot-web
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-bot-web
  template:
    metadata:
      labels:
        app: trading-bot-web
    spec:
      containers:
      - name: web
        image: evil2root/trading-bot:local
        command: ["web-with-scheduler"]
        ports:
        - containerPort: 5000
        - containerPort: 9090
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: DB_HOST
        - name: DB_PORT
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: DB_PORT
        - name: DB_NAME
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: DB_NAME
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: db_user.txt
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: db_password.txt
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: secret_key.txt
        - name: ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: admin_password.txt
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: REDIS_HOST
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: REDIS_PORT
        - name: APP_ENV
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: APP_ENV
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: trading-bot-config
              key: LOG_LEVEL
        volumeMounts:
        - name: app-data
          mountPath: /app/data
        - name: app-logs
          mountPath: /app/logs
        - name: app-models
          mountPath: /app/saved_models
        resources:
          limits:
            memory: "4Gi"
            cpu: "1"
          requests:
            memory: "2Gi"
            cpu: "500m"
      volumes:
      - name: app-data
        persistentVolumeClaim:
          claimName: app-data
      - name: app-logs
        emptyDir: {}
      - name: app-models
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: trading-bot-web
spec:
  selector:
    app: trading-bot-web
  ports:
  - name: http
    port: 5000
    targetPort: 5000
  - name: metrics
    port: 9090
    targetPort: 9090
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-bot-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: trading.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: trading-bot-web
            port:
              number: 5000
EOF
}

deploy_monitoring() {
    print_header "Déploiement de la surveillance (Grafana)"
    
    # Déployer Grafana
    cat <<EOF | kubectl --context $KUBE_CONTEXT -n $NAMESPACE apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_USER
          value: "admin"
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin"
        - name: GF_USERS_ALLOW_SIGN_UP
          value: "false"
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi"
            cpu: "250m"
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grafana-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: grafana.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
EOF
}

update_hosts() {
    print_header "Mise à jour du fichier hosts"
    
    echo "Pour accéder à votre application, ajoutez les lignes suivantes à votre fichier /etc/hosts :"
    echo "127.0.0.1 trading.local"
    echo "127.0.0.1 grafana.local"
    
    # Vérifier si l'utilisateur souhaite mettre à jour automatiquement le fichier hosts
    read -p "Voulez-vous mettre à jour automatiquement le fichier hosts ? (o/n) " response
    if [[ "$response" =~ ^[Oo]$ ]]; then
        echo "Mise à jour du fichier hosts..."
        # Vérifier si les entrées existent déjà
        if ! grep -q "trading.local" /etc/hosts; then
            echo "127.0.0.1 trading.local" | sudo tee -a /etc/hosts
        fi
        if ! grep -q "grafana.local" /etc/hosts; then
            echo "127.0.0.1 grafana.local" | sudo tee -a /etc/hosts
        fi
        echo "Fichier hosts mis à jour avec succès."
    fi
}

print_summary() {
    print_header "Résumé du déploiement"
    
    echo "Application déployée avec succès sur le cluster Kubernetes local."
    echo "URLs d'accès :"
    echo "- Interface web : http://trading.local"
    echo "- Grafana : http://grafana.local"
    echo ""
    echo "Pour afficher les pods :"
    echo "kubectl --context $KUBE_CONTEXT -n $NAMESPACE get pods"
    echo ""
    echo "Pour afficher les logs d'un pod :"
    echo "kubectl --context $KUBE_CONTEXT -n $NAMESPACE logs -f <nom-du-pod>"
    echo ""
    echo "Pour supprimer le cluster :"
    echo "kind delete cluster --name evil2root-trading"
}

# Exécution principale
check_prerequisites
create_secrets
create_configmap
deploy_storage
deploy_infrastructure
deploy_application
deploy_monitoring
update_hosts
print_summary 
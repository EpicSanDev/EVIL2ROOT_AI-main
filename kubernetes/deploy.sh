#!/bin/bash

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
log() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Vérification des prérequis
log "Vérification des prérequis..."
if ! command -v kubectl &> /dev/null; then
  error "kubectl n'est pas installé ou n'est pas dans le PATH."
  exit 1
fi

if ! command -v doctl &> /dev/null; then
  warn "doctl n'est pas installé. Vous ne pourrez pas vous authentifier automatiquement au registre DigitalOcean."
fi

# Création du namespace
log "Création du namespace evil2root-trading..."
kubectl create namespace evil2root-trading --dry-run=client -o yaml | kubectl apply -f -

# Authentification au registre DigitalOcean
log "Vérification de l'authentification au registre DigitalOcean..."

# Vérifier si le secret registry-evil2root-registry existe déjà
if kubectl get secret registry-evil2root-registry -n evil2root-trading &> /dev/null; then
  log "Le secret registry-evil2root-registry existe déjà."
else
  log "Le secret registry-evil2root-registry n'existe pas encore."
  if command -v doctl &> /dev/null; then
    log "Tentative d'authentification au registre DigitalOcean..."
    if ! doctl registry login; then
      warn "Impossible de se connecter automatiquement au registre DigitalOcean."
      
      # Demander les informations manuellement
      read -p "Veuillez entrer votre token API DigitalOcean: " DO_API_TOKEN
      read -p "Veuillez entrer votre adresse email: " DO_EMAIL
      
      if [ -n "$DO_API_TOKEN" ] && [ -n "$DO_EMAIL" ]; then
        log "Création manuelle du secret registry-evil2root-registry..."
        kubectl create secret docker-registry registry-evil2root-registry \
          --docker-server=registry.digitalocean.com \
          --docker-username=$DO_API_TOKEN \
          --docker-password=$DO_API_TOKEN \
          --docker-email=$DO_EMAIL \
          -n evil2root-trading
      else
        warn "Token API ou email non fourni. Le secret de registre n'a pas été créé."
        read -p "Voulez-vous continuer le déploiement? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
          error "Déploiement annulé."
          exit 1
        fi
      fi
    fi
  else
    warn "La commande doctl n'est pas disponible. L'authentification au registre doit être configurée manuellement."
    # Demander les informations manuellement
    read -p "Voulez-vous configurer manuellement le secret registry? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      read -p "Veuillez entrer votre token API DigitalOcean: " DO_API_TOKEN
      read -p "Veuillez entrer votre adresse email: " DO_EMAIL
      
      if [ -n "$DO_API_TOKEN" ] && [ -n "$DO_EMAIL" ]; then
        log "Création manuelle du secret registry-evil2root-registry..."
        kubectl create secret docker-registry registry-evil2root-registry \
          --docker-server=registry.digitalocean.com \
          --docker-username=$DO_API_TOKEN \
          --docker-password=$DO_API_TOKEN \
          --docker-email=$DO_EMAIL \
          -n evil2root-trading
      else
        warn "Token API ou email non fourni. Le secret de registre n'a pas été créé."
      fi
    fi
  fi
fi

# Application des secrets
log "Application des secrets..."
kubectl apply -f kubernetes/secrets.yaml

# Déploiement des services de base
log "Déploiement de Redis..."
kubectl apply -f kubernetes/services/redis.yaml
log "Déploiement de PostgreSQL..."
kubectl apply -f kubernetes/services/postgres.yaml

# Vérification de l'installation du metrics-server
log "Vérification de l'installation du metrics-server..."
if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
  log "Installation du metrics-server..."
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
else
  log "Le metrics-server est déjà installé."
fi

# Vérification de l'installation de Nginx Ingress Controller
log "Vérification de l'installation de Nginx Ingress Controller..."
if ! kubectl get namespace ingress-nginx &> /dev/null; then
  log "Installation de Nginx Ingress Controller..."
  kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
else
  log "Nginx Ingress Controller est déjà installé."
fi

# Déploiement des applications
log "Déploiement des applications..."
kubectl apply -f kubernetes/deployments/trading-bot-web.yaml
kubectl apply -f kubernetes/deployments/analysis-bot.yaml
kubectl apply -f kubernetes/deployments/market-scheduler.yaml

# Déploiement des services
log "Déploiement des services..."
kubectl apply -f kubernetes/services/trading-bot-web.yaml
kubectl apply -f kubernetes/services/web-service.yaml

# Déploiement de l'ingress
log "Déploiement de l'ingress..."
kubectl apply -f kubernetes/ingress/trading-bot-web-ingress.yaml

# Déploiement des autoscalers
log "Déploiement des autoscalers..."
kubectl apply -f kubernetes/hpa/trading-bot-web-hpa.yaml
kubectl apply -f kubernetes/hpa/analysis-bot-hpa.yaml

log "Attente de l'initialisation des services..."
sleep 10

# Vérification du déploiement
log "Vérification du déploiement..."
kubectl get pods -n evil2root-trading
kubectl get services -n evil2root-trading
kubectl get ingress -n evil2root-trading

log "Déploiement terminé ! Vous pouvez maintenant accéder à votre application via l'URL configurée dans l'ingress."
log "Pour surveiller les pods: kubectl get pods -n evil2root-trading -w"
log "Pour voir les logs d'un pod: kubectl logs -n evil2root-trading <nom-du-pod>" 
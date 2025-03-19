#!/bin/bash

# Script de déploiement du vérificateur de santé des composants
# Evil2Root AI Trading

set -e

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions pour les messages
log() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

header() {
  echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Vérification des dépendances
if ! command -v kubectl &> /dev/null; then
  error "kubectl n'est pas installé"
  exit 1
fi

if ! command -v docker &> /dev/null; then
  error "docker n'est pas installé"
  exit 1
fi

# Récupération des paramètres
NAMESPACE=${NAMESPACE:-"evil2root-trading"}
REGISTRY_URL=${REGISTRY_URL:-"registry.evil2root.ai"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
IMAGE_NAME="component-health-checker"

# Récupération du répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

header "DÉPLOIEMENT DU VÉRIFICATEUR DE SANTÉ DES COMPOSANTS"
log "Namespace: $NAMESPACE"
log "Registry: $REGISTRY_URL"
log "Image: $IMAGE_NAME:$IMAGE_TAG"

# Construction de l'image Docker
header "CONSTRUCTION DE L'IMAGE DOCKER"
log "Construction de l'image $IMAGE_NAME:$IMAGE_TAG..."

docker build -t "$REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG" \
  -f "$REPO_ROOT/Dockerfile.component-health-checker" \
  "$REPO_ROOT"

if [ $? -ne 0 ]; then
  error "Erreur lors de la construction de l'image"
  exit 1
fi

log "Image construite avec succès"

# Push de l'image dans le registry
header "PUBLICATION DE L'IMAGE DANS LE REGISTRY"
log "Publication de l'image $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG..."

docker push "$REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG"

if [ $? -ne 0 ]; then
  error "Erreur lors de la publication de l'image"
  exit 1
fi

log "Image publiée avec succès"

# Création du namespace si nécessaire
header "PRÉPARATION DU NAMESPACE"
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
  log "Création du namespace $NAMESPACE..."
  kubectl create namespace "$NAMESPACE"
else
  log "Le namespace $NAMESPACE existe déjà"
fi

# Application des manifestes Kubernetes
header "DÉPLOIEMENT DANS KUBERNETES"
log "Application des manifestes..."

# Remplacement des variables dans le fichier YAML
sed "s|\${REGISTRY_URL}|$REGISTRY_URL|g" \
  "$SCRIPT_DIR/component-health-checker-cronjob.yaml" > "$SCRIPT_DIR/component-health-checker-cronjob-temp.yaml"

# Application du fichier
kubectl apply -f "$SCRIPT_DIR/component-health-checker-cronjob-temp.yaml"

# Nettoyage
rm "$SCRIPT_DIR/component-health-checker-cronjob-temp.yaml"

log "Vérification du déploiement..."
kubectl get cronjob -n "$NAMESPACE" | grep "component-health-checker"

log "Vérification des ressources créées..."
kubectl get serviceaccount,role,rolebinding,pvc,configmap -n "$NAMESPACE" | grep "component-health-checker"

header "VALIDATION"
log "Pour déclencher une exécution manuelle du vérificateur de santé, exécutez:"
echo "kubectl create job --from=cronjob/component-health-checker component-health-checker-manual -n $NAMESPACE"

log "Pour consulter les logs du job, exécutez:"
echo "kubectl logs -f job/component-health-checker-manual -n $NAMESPACE"

log "Pour supprimer le job manuel après exécution:"
echo "kubectl delete job component-health-checker-manual -n $NAMESPACE"

log "Déploiement terminé avec succès!" 
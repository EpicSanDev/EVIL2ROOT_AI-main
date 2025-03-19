#!/bin/bash

# Script de correction des problèmes courants dans le cluster Kubernetes
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

# Récupération des paramètres
NAMESPACE=${NAMESPACE:-"evil2root-trading"}
REGISTRY_URL=${REGISTRY_URL:-"registry.digitalocean.com/evil2root-registry"}

header "CORRECTION DES PROBLÈMES COURANTS DANS LE CLUSTER KUBERNETES"
log "Namespace: $NAMESPACE"

# Vérification et correction de l'espace de nom
header "VÉRIFICATION DU NAMESPACE"
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
  log "Création du namespace $NAMESPACE..."
  kubectl create namespace "$NAMESPACE"
else
  log "Le namespace $NAMESPACE existe déjà"
fi

# Vérification et correction des PVCs en attente
header "VÉRIFICATION DES PVCs"
log "Recherche des PVCs en état Pending..."

PENDING_PVCS=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{range .items[?(@.status.phase=="Pending")]}{.metadata.name}{"\n"}{end}')

if [ -n "$PENDING_PVCS" ]; then
  warn "PVCs en attente trouvés:"
  echo "$PENDING_PVCS"
  
  # Correction des PVCs en attente
  for PVC in $PENDING_PVCS; do
    log "Analyse du PVC $PVC..."
    STORAGE_CLASS=$(kubectl get pvc "$PVC" -n "$NAMESPACE" -o jsonpath='{.spec.storageClassName}')
    
    if [ "$STORAGE_CLASS" == "standard" ]; then
      warn "Le PVC $PVC utilise la classe de stockage 'standard' qui n'existe pas"
      
      log "Sauvegarde de la configuration actuelle..."
      kubectl get pvc "$PVC" -n "$NAMESPACE" -o yaml > "/tmp/pvc-$PVC-backup.yaml"
      
      log "Suppression du PVC..."
      kubectl delete pvc "$PVC" -n "$NAMESPACE"
      
      log "Création d'un nouveau PVC avec la classe de stockage do-block-storage..."
      sed 's/storageClassName: standard/storageClassName: do-block-storage/g' "/tmp/pvc-$PVC-backup.yaml" | kubectl apply -f -
      
      log "Nettoyage..."
      rm -f "/tmp/pvc-$PVC-backup.yaml"
    fi
  done
else
  log "Aucun PVC en attente trouvé"
fi

# Vérification et correction des pods en CrashLoopBackOff ou OOMKilled
header "VÉRIFICATION DES PODS"
log "Recherche des pods en échec..."

FAILED_PODS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[?(@.status.phase!="Running")]}{.metadata.name}{"\n"}{end}')

if [ -n "$FAILED_PODS" ]; then
  warn "Pods en échec trouvés:"
  echo "$FAILED_PODS"
  
  # Récupération des pods OOMKilled
  OOM_PODS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[?(@.status.containerStatuses[0].lastState.terminated.reason=="OOMKilled")]}{.metadata.name}{"\n"}{end}')
  
  if [ -n "$OOM_PODS" ]; then
    warn "Pods tués par manque de mémoire (OOMKilled):"
    echo "$OOM_PODS"
    
    log "Mise à jour des limites de ressources dans les déploiements..."
    
    for DEPLOYMENT in $(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'); do
      log "Augmentation des ressources pour le déploiement $DEPLOYMENT..."
      
      # Sauvegarde du déploiement
      kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o yaml > "/tmp/deployment-$DEPLOYMENT-backup.yaml"
      
      # Augmentation des ressources
      PATCH='{
        "spec": {
          "template": {
            "spec": {
              "containers": [
                {
                  "resources": {
                    "requests": {
                      "memory": "512Mi"
                    },
                    "limits": {
                      "memory": "1Gi"
                    }
                  }
                }
              ]
            }
          }
        }
      }'
      
      # Application du patch
      kubectl patch deployment "$DEPLOYMENT" -n "$NAMESPACE" --type=merge -p "$PATCH" || {
        warn "Échec du patch pour $DEPLOYMENT, tentative avec une méthode alternative..."
        sed -i 's/memory: "256Mi"/memory: "512Mi"/g' "/tmp/deployment-$DEPLOYMENT-backup.yaml"
        sed -i 's/memory: "512Mi"/memory: "1Gi"/g' "/tmp/deployment-$DEPLOYMENT-backup.yaml"
        kubectl apply -f "/tmp/deployment-$DEPLOYMENT-backup.yaml"
      }
      
      # Nettoyage
      rm -f "/tmp/deployment-$DEPLOYMENT-backup.yaml"
    done
  fi
  
  # Suppression des pods bloqués
  log "Suppression des pods bloqués pour les recréer..."
  for POD in $FAILED_PODS; do
    if echo "$POD" | grep -q "component-health-checker-test\|test-health-checker"; then
      log "Ignoring job pod $POD..."
      continue
    fi
    
    log "Suppression du pod $POD..."
    kubectl delete pod "$POD" -n "$NAMESPACE" --grace-period=0 --force
  done
else
  log "Aucun pod en échec trouvé"
fi

# Vérification et correction des CronJobs
header "VÉRIFICATION DES CRONJOBS"
log "Recherche des CronJobs avec des problèmes d'image..."

CRONJOBS=$(kubectl get cronjobs -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')

for CRONJOB in $CRONJOBS; do
  log "Analyse du CronJob $CRONJOB..."
  
  # Vérification de l'image
  IMAGE=$(kubectl get cronjob "$CRONJOB" -n "$NAMESPACE" -o jsonpath='{.spec.jobTemplate.spec.template.spec.containers[0].image}')
  
  if [[ "$IMAGE" == *\$\{REGISTRY_URL\}* ]]; then
    warn "Le CronJob $CRONJOB utilise une variable non résolue dans l'image: $IMAGE"
    
    log "Sauvegarde de la configuration actuelle..."
    kubectl get cronjob "$CRONJOB" -n "$NAMESPACE" -o yaml > "/tmp/cronjob-$CRONJOB-backup.yaml"
    
    log "Mise à jour de l'image..."
    FIXED_IMAGE=$(echo "$IMAGE" | sed "s|\${REGISTRY_URL}|$REGISTRY_URL|g")
    
    # Mise à jour du CronJob
    PATCH="{
      \"spec\": {
        \"jobTemplate\": {
          \"spec\": {
            \"template\": {
              \"spec\": {
                \"containers\": [
                  {
                    \"image\": \"$FIXED_IMAGE\"
                  }
                ]
              }
            }
          }
        }
      }
    }"
    
    kubectl patch cronjob "$CRONJOB" -n "$NAMESPACE" --type=merge -p "$PATCH" || {
      warn "Échec du patch pour $CRONJOB, tentative avec une méthode alternative..."
      sed -i "s|\${REGISTRY_URL}|$REGISTRY_URL|g" "/tmp/cronjob-$CRONJOB-backup.yaml"
      kubectl apply -f "/tmp/cronjob-$CRONJOB-backup.yaml"
    }
    
    # Nettoyage
    rm -f "/tmp/cronjob-$CRONJOB-backup.yaml"
  fi
done

header "TERMINÉ"
log "Toutes les vérifications et corrections ont été effectuées"
log "Veuillez vérifier l'état du cluster avec:"
echo "kubectl get pods,deployments,svc,pvc -n $NAMESPACE" 
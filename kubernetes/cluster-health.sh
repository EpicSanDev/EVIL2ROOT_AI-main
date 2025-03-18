#!/bin/bash

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

header() {
  echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Vérification de kubectl
if ! command -v kubectl &> /dev/null; then
  error "kubectl n'est pas installé ou n'est pas dans le PATH."
  exit 1
fi

# Namespace à vérifier
NAMESPACE="evil2root-trading"

header "VÉRIFICATION DE L'ÉTAT DU CLUSTER"

# Vérification de l'accès au cluster
log "Vérification de l'accès au cluster..."
if ! kubectl cluster-info &> /dev/null; then
  error "Impossible de se connecter au cluster Kubernetes."
  exit 1
fi
log "✅ Connexion au cluster OK"

# Vérification du namespace
log "Vérification du namespace ${NAMESPACE}..."
if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
  error "Le namespace ${NAMESPACE} n'existe pas."
  exit 1
fi
log "✅ Le namespace ${NAMESPACE} existe"

# Vérification des pods
header "VÉRIFICATION DES PODS"
log "Liste des pods en erreur ou non prêts:"
NOT_READY_PODS=$(kubectl get pods -n "${NAMESPACE}" -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,READY_CONTAINERS:.status.containerStatuses[*].ready | grep -v "Running" | grep -v "Completed" | grep -v "NAME")

if [ -n "$NOT_READY_PODS" ]; then
  echo "$NOT_READY_PODS"
  warn "Des pods ne sont pas prêts ou en erreur."
  
  # Afficher les détails des pods en erreur
  echo
  log "Détails des pods en erreur:"
  POD_NAMES=$(echo "$NOT_READY_PODS" | awk '{print $1}')
  for pod in $POD_NAMES; do
    echo -e "\n${YELLOW}=== Pod: ${pod} ===${NC}"
    kubectl describe pod "$pod" -n "${NAMESPACE}" | grep -E "^  Warning|^  Error|Reason:|Message:" | head -20
    echo -e "\n${YELLOW}Dernières lignes de logs:${NC}"
    kubectl logs "$pod" -n "${NAMESPACE}" --tail=20 || echo "Impossible d'obtenir les logs"
  done
else
  log "✅ Tous les pods sont en cours d'exécution"
fi

# Vérification des services
header "VÉRIFICATION DES SERVICES"
log "Services dans le namespace ${NAMESPACE}:"
kubectl get services -n "${NAMESPACE}" -o wide

# Vérification des endpoints
log "Vérification des endpoints..."
MISSING_ENDPOINTS=$(kubectl get endpoints -n "${NAMESPACE}" -o custom-columns=NAME:.metadata.name,ENDPOINTS:.subsets[*].addresses | grep -E "<none>|<nil>")
if [ -n "$MISSING_ENDPOINTS" ]; then
  warn "Certains services n'ont pas d'endpoints:"
  echo "$MISSING_ENDPOINTS"
else
  log "✅ Tous les services ont des endpoints"
fi

# Vérification des déploiements
header "VÉRIFICATION DES DÉPLOIEMENTS"
log "Déploiements dans le namespace ${NAMESPACE}:"
DEPLOYMENTS_STATUS=$(kubectl get deployments -n "${NAMESPACE}" -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,AVAILABLE:.status.availableReplicas,UP-TO-DATE:.status.updatedReplicas)
echo "$DEPLOYMENTS_STATUS"

# Vérification des déploiements non disponibles
UNHEALTHY_DEPLOYMENTS=$(echo "$DEPLOYMENTS_STATUS" | grep -v "NAME" | awk '$2 != $3 {print $1}')
if [ -n "$UNHEALTHY_DEPLOYMENTS" ]; then
  warn "Déploiements non sains:"
  for deployment in $UNHEALTHY_DEPLOYMENTS; do
    echo -e "\n${YELLOW}=== Déploiement: ${deployment} ===${NC}"
    kubectl describe deployment "$deployment" -n "${NAMESPACE}" | grep -E "^  Warning|Message:|Reason:|Replicas:"
  done
else
  log "✅ Tous les déploiements sont sains"
fi

# Vérification des events récents
header "ÉVÉNEMENTS RÉCENTS"
log "Derniers événements du namespace ${NAMESPACE}:"
kubectl get events -n "${NAMESPACE}" --sort-by='.lastTimestamp' | tail -10

# Vérification des PersistentVolumeClaims
header "VÉRIFICATION DES VOLUMES PERSISTANTS"
log "PersistentVolumeClaims dans le namespace ${NAMESPACE}:"
kubectl get pvc -n "${NAMESPACE}" -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,CAPACITY:.spec.resources.requests.storage

# Vérification des problèmes réseau
header "VÉRIFICATION DES PROBLÈMES RÉSEAU"
log "NetworkPolicies dans le namespace ${NAMESPACE}:"
kubectl get networkpolicies -n "${NAMESPACE}" -o custom-columns=NAME:.metadata.name,POD-SELECTOR:.spec.podSelector.matchLabels || echo "Aucune NetworkPolicy trouvée"

# Vérification de l'utilisation des ressources
header "UTILISATION DES RESSOURCES"
log "Nœuds du cluster:"
kubectl top nodes || echo "metrics-server non disponible"

log "Utilisation des ressources des pods:"
kubectl top pods -n "${NAMESPACE}" || echo "metrics-server non disponible"

# Vérification de l'état des secrets
header "VÉRIFICATION DES SECRETS"
log "Secrets dans le namespace ${NAMESPACE}:"
kubectl get secrets -n "${NAMESPACE}" -o custom-columns=NAME:.metadata.name,TYPE:.type,AGE:.metadata.creationTimestamp

# Vérification de l'état des ingress
header "VÉRIFICATION DES INGRESS"
log "Ingress dans le namespace ${NAMESPACE}:"
kubectl get ingress -n "${NAMESPACE}" -o custom-columns=NAME:.metadata.name,HOSTS:.spec.rules[*].host,ADDRESS:.status.loadBalancer.ingress[*].ip || echo "Aucun Ingress trouvé"

# Vérification des pods qui consomment beaucoup de ressources
header "TOP 5 DES PODS QUI CONSOMMENT LE PLUS DE RESSOURCES"
log "Pods avec la consommation CPU la plus élevée:"
kubectl top pods -n "${NAMESPACE}" --sort-by=cpu | tail -5 || echo "metrics-server non disponible"

log "Pods avec la consommation mémoire la plus élevée:"
kubectl top pods -n "${NAMESPACE}" --sort-by=memory | tail -5 || echo "metrics-server non disponible"

header "RÉSUMÉ DE L'ÉTAT DU CLUSTER"
log "Namespace: ${NAMESPACE}"
TOTAL_PODS=$(kubectl get pods -n "${NAMESPACE}" | grep -v "NAME" | wc -l)
RUNNING_PODS=$(kubectl get pods -n "${NAMESPACE}" | grep "Running" | wc -l)
FAILING_PODS=$((TOTAL_PODS - RUNNING_PODS))

log "Total des pods: ${TOTAL_PODS}"
log "Pods en cours d'exécution: ${RUNNING_PODS}"

if [ "$FAILING_PODS" -gt 0 ]; then
  warn "Pods en échec: ${FAILING_PODS}"
else
  log "✅ Aucun pod en échec"
fi

TOTAL_DEPLOYMENTS=$(kubectl get deployments -n "${NAMESPACE}" | grep -v "NAME" | wc -l)
log "Total des déploiements: ${TOTAL_DEPLOYMENTS}"

if [ -n "$UNHEALTHY_DEPLOYMENTS" ]; then
  UNHEALTHY_COUNT=$(echo "$UNHEALTHY_DEPLOYMENTS" | wc -w)
  warn "Déploiements non sains: ${UNHEALTHY_COUNT}"
else
  log "✅ Tous les déploiements sont sains"
fi

if [ -n "$MISSING_ENDPOINTS" ]; then
  MISSING_COUNT=$(echo "$MISSING_ENDPOINTS" | wc -l)
  warn "Services sans endpoints: ${MISSING_COUNT}"
else
  log "✅ Tous les services ont des endpoints"
fi

echo
if [ "$FAILING_PODS" -gt 0 ] || [ -n "$UNHEALTHY_DEPLOYMENTS" ] || [ -n "$MISSING_ENDPOINTS" ]; then
  warn "⚠️ Des problèmes ont été détectés. Veuillez consulter le rapport pour plus de détails."
else
  log "🎉 Le cluster semble être en bon état !"
fi 
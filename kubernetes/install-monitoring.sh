#!/bin/bash

# Script d'installation du système de monitoring (Prometheus + Grafana)
# Evil2Root AI Trading

set -e

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions pour les messages
log() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

header() {
  echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Namespace pour les services de monitoring
NAMESPACE="monitoring"
NAMESPACE_TRADING="evil2root-trading"

header "INSTALLATION DU SYSTÈME DE MONITORING (PROMETHEUS + GRAFANA)"
log "Namespace: $NAMESPACE"

# Création du namespace monitoring
log "Création du namespace monitoring..."
kubectl apply -f kubernetes/monitoring/namespace.yaml

# Installation de Prometheus Operator
log "Installation de l'opérateur Prometheus..."
kubectl apply -f kubernetes/monitoring/prometheus/prometheus-operator.yaml

log "Attente de l'initialisation de l'opérateur Prometheus..."
sleep 30

# Installation de Prometheus
log "Installation de Prometheus..."
kubectl apply -f kubernetes/monitoring/prometheus/prometheus.yaml

log "Attente de l'initialisation de Prometheus..."
sleep 30

# Installation de Grafana
log "Installation de Grafana..."
kubectl apply -f kubernetes/monitoring/grafana/grafana.yaml
kubectl apply -f kubernetes/monitoring/grafana/trading-bot-dashboard.yaml

log "Attente de l'initialisation de Grafana..."
sleep 30

# Installation des Service Monitors
log "Installation des Service Monitors..."
kubectl apply -f kubernetes/monitoring/servicemonitors/

# Installation des règles d'alerte
log "Installation des règles d'alerte..."
kubectl apply -f kubernetes/monitoring/alertrules/

header "INSTALLATION DU VERTICAL POD AUTOSCALER (VPA)"
log "Installation du VPA en cours..."

# Cloner le dépôt VPA
log "Clonage du dépôt VPA..."
git clone --depth 1 https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler/

# Installation du VPA
log "Installation du Vertical Pod Autoscaler..."
./hack/vpa-up.sh

log "VPA installé avec succès !"

# Retour au répertoire précédent
cd ../..
rm -rf autoscaler

# Application des configurations VPA
log "Application des configurations VPA..."
kubectl apply -f kubernetes/vpa/

header "INSTALLATION DES POD DISRUPTION BUDGETS (PDB)"
log "Application des Pod Disruption Budgets..."
kubectl apply -f kubernetes/pdb/

header "VÉRIFICATION DE L'INSTALLATION"
log "Vérification des pods de monitoring..."
kubectl get pods -n $NAMESPACE

log "Vérification des VPAs..."
kubectl get vpa -n $NAMESPACE_TRADING

log "Vérification des PDBs..."
kubectl get pdb -n $NAMESPACE_TRADING

header "ACCÈS À GRAFANA"
log "Pour accéder à Grafana, exécutez la commande suivante:"
echo "kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000"
log "Puis accédez à http://localhost:3000 dans votre navigateur"
log "Utilisateur: admin"
log "Mot de passe: admin123"

header "INSTALLATION TERMINÉE AVEC SUCCÈS" 
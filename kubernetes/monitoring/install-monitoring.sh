#!/bin/bash

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "${BLUE}Installation du système de surveillance...${NC}"

# Création du namespace monitoring s'il n'existe pas déjà
echo "${GREEN}Création du namespace monitoring...${NC}"
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Ajout du repo Helm pour Prometheus
echo "${GREEN}Ajout du repo Helm pour Prometheus...${NC}"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Installation de kube-prometheus-stack avec Helm
echo "${GREEN}Installation de Prometheus et Grafana avec Helm...${NC}"
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.adminPassword=admin123 \
  --set grafana.service.type=ClusterIP \
  --set grafana.ingress.enabled=true \
  --set grafana.ingress.annotations."kubernetes\.io/ingress\.class"=nginx \
  --set grafana.ingress.hosts[0]=grafana.evil2root.com \
  --set prometheus.prometheusSpec.serviceMonitorSelector.matchLabels.app=trading-bot \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=do-block-storage \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=10Gi \
  --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.storageClassName=do-block-storage \
  --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.resources.requests.storage=2Gi

# Déploiement des ServiceMonitors
echo "${GREEN}Déploiement du ServiceMonitor pour le trading bot...${NC}"
kubectl apply -f kubernetes/monitoring/servicemonitors/trading-bot-servicemonitor.yaml

# Déploiement des règles d'alerte
echo "${GREEN}Déploiement des règles d'alerte...${NC}"
kubectl apply -f kubernetes/monitoring/alertrules/resource-alerts.yaml

# Attente de la disponibilité des pods
echo "${GREEN}Attente du démarrage des pods...${NC}"
kubectl wait --for=condition=ready pod --selector=app.kubernetes.io/name=grafana --namespace monitoring --timeout=120s || echo "Timeout en attendant le démarrage de Grafana"
kubectl wait --for=condition=ready pod --selector=app=prometheus --namespace monitoring --timeout=120s || echo "Timeout en attendant le démarrage de Prometheus"

# Affichage des informations sur les services
echo "${GREEN}Services de surveillance déployés :${NC}"
kubectl get svc -n monitoring

echo "${BLUE}Installation du système de surveillance terminée !${NC}" 
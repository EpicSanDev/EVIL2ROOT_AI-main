#!/bin/bash

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "${BLUE}Redémarrage du système de surveillance...${NC}"

# Vérification de l'état actuel
echo "${GREEN}État actuel des pods de surveillance :${NC}"
kubectl get pods -n monitoring

# Redémarrage de Prometheus
echo "${GREEN}Redémarrage de Prometheus...${NC}"
kubectl delete pod -n monitoring -l app=prometheus --force --grace-period=0 || echo "Pods Prometheus non trouvés."

# Redémarrage de Grafana
echo "${GREEN}Redémarrage de Grafana...${NC}"
kubectl delete pod -n monitoring -l app.kubernetes.io/name=grafana --force --grace-period=0 || echo "Pods Grafana non trouvés."

# Redémarrage de AlertManager
echo "${GREEN}Redémarrage de AlertManager...${NC}"
kubectl delete pod -n monitoring -l app=alertmanager --force --grace-period=0 || echo "Pods AlertManager non trouvés."

# Attente et vérification de l'état après redémarrage
echo "${GREEN}Attente du redémarrage des pods (60 secondes)...${NC}"
sleep 60

# Vérification de l'état final
echo "${GREEN}État final des pods de surveillance :${NC}"
kubectl get pods -n monitoring

echo "${BLUE}Redémarrage du système de surveillance terminé !${NC}" 
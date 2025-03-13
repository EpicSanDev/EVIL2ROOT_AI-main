#!/bin/bash

# Script pour arrêter tous les services Docker
# Usage: ./stop_docker.sh

echo "=== Evil2Root Trading Bot - Arrêt des services Docker ==="
echo ""

# Vérifier si Docker et Docker Compose sont installés
if ! command -v docker &> /dev/null; then
    echo "Erreur: Docker n'est pas installé."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Erreur: Docker Compose n'est pas installé."
    exit 1
fi

echo "Arrêt des services Docker..."

# Arrêter les services
docker-compose down

echo ""
echo "Services arrêtés avec succès!" 
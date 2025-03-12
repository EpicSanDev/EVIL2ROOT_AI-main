#!/bin/bash

# Script pour démarrer les services Docker avec entraînement forcé des modèles
# Usage: ./start_docker_force_train.sh

echo "=== Evil2Root Trading Bot - Démarrage avec entraînement forcé ==="
echo "Ce script lance le bot avec un entraînement forcé des modèles"
echo ""

# Exécuter le script de démarrage avec l'option force-train
./start_docker.sh --force-train 
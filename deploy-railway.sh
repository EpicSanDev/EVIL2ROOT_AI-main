#!/bin/bash

# Script pour déployer sur Railway

set -e

echo "Préparation du déploiement sur Railway..."

# Vérifier si Railway CLI est installé
if ! command -v railway &> /dev/null; then
    echo "Railway CLI n'est pas installé. Installation en cours..."
    npm install -g @railway/cli
fi

echo "Connexion à Railway (vous devrez vous authentifier si nécessaire)..."
railway login

echo "Initialisation du projet Railway..."
railway init

echo "Construction et déploiement du projet sur Railway..."
railway up

echo "Déploiement terminé avec succès!"
echo "Vous pouvez accéder à votre projet sur https://railway.app/dashboard"

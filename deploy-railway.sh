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

# Configurer les variables d'environnement pour TA-Lib (aide le build)
echo "Configuration des variables d'environnement pour TA-Lib..."
railway vars set TA_INCLUDE_PATH=/usr/include
railway vars set TA_LIBRARY_PATH=/usr/lib
railway vars set TA_LIB_VERSION=0.4.28

echo "Construction et déploiement du projet sur Railway..."
railway up

echo "Déploiement terminé avec succès!"
echo "Vous pouvez accéder à votre projet sur https://railway.app/dashboard"
echo "Si vous rencontrez des problèmes avec TA-Lib, vérifiez les logs et considérez ajouter un override pour la commande de build."

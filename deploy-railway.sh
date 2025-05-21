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
railway vars set NUMPY_VERSION=1.24.3
railway vars set TALIB_USE_BINARY=true

echo "Construction et déploiement du projet sur Railway..."
railway up

echo "Déploiement terminé avec succès!"
echo "Vous pouvez accéder à votre projet sur https://railway.app/dashboard"
echo ""
echo "=== GUIDE DE DÉPANNAGE POUR TA-LIB ==="
echo "Si vous rencontrez des problèmes avec l'installation de TA-Lib, essayez les solutions suivantes:"
echo ""
echo "1. Utiliser la configuration de build Railway personnalisée:"
echo "   - Allez dans 'Settings' > 'Build' dans votre service Railway"
echo "   - Remplacez la commande de build par: 'docker/talib-binary-install.sh && pip install -r requirements.txt'"
echo ""
echo "2. Utiliser le script webhook fourni:"
echo "   - Allez dans 'Settings' > 'Webhooks' dans votre service Railway"
echo "   - Créez un webhook de type 'Build'"
echo "   - Configurez-le pour exécuter le script 'railway-build-webhook.sh'"
echo ""
echo "3. Modifier temporairement votre code pour utiliser une implémentation alternative:"
echo "   - Remplacez 'import talib' par une implémentation basée sur pandas ou numpy"
echo "   - Utilisez des bibliothèques alternatives comme pandas-ta ou ta (pip install ta)"
echo ""
echo "4. Références utiles:"
echo "   - Documentation Railway: https://docs.railway.app/"
echo "   - Alternatives à TA-Lib: pandas-ta, ta, fin-tech"
echo "================================================================="

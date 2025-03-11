#!/bin/bash

# Script pour démarrer l'application Evil2Root Trading Bot en production

# Vérifier que docker et docker-compose sont installés
if ! command -v docker &> /dev/null; then
    echo "Docker n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Créer les dossiers nécessaires s'ils n'existent pas
mkdir -p data logs saved_models

# S'assurer que les permissions sont correctes
chmod -R 777 logs data saved_models

# Vérifier si le fichier .env existe, sinon le créer à partir de .env.example
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Fichier .env non trouvé. Création à partir du modèle .env.example..."
        cp .env.example .env
        echo "Veuillez éditer le fichier .env avec vos paramètres avant de continuer."
        exit 0
    else
        echo "Ni .env ni .env.example n'ont été trouvés. Veuillez créer un fichier .env manuellement."
        exit 1
    fi
fi

# Modifier .env pour activer le mode production
sed -i 's/ENABLE_LIVE_TRADING=false/ENABLE_LIVE_TRADING=true/g' .env
echo "Mode production activé dans .env (ENABLE_LIVE_TRADING=true)"

# Accorder les droits d'exécution aux scripts d'entrée
chmod +x docker/services/entrypoint-*.sh

# Arrêter les conteneurs existants si nécessaire
echo "Arrêt des conteneurs existants..."
docker-compose down

# Construire et démarrer les conteneurs en mode détaché
echo "Construction et démarrage des conteneurs..."
docker-compose up -d --build

echo "Vérification de l'état des conteneurs..."
sleep 5
docker-compose ps

echo ""
echo "L'application Evil2Root Trading Bot est démarrée en mode production."
echo "Interface web disponible à l'adresse: http://localhost:5000"
echo ""
echo "Pour voir les logs: docker-compose logs -f"
echo "Pour arrêter l'application: docker-compose down" 
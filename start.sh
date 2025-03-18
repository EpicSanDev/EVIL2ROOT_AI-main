#!/bin/bash

# Script de démarrage pour EVIL2ROOT Trading Bot

set -e

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "Docker n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Vérifier si Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Créer le fichier .env s'il n'existe pas
if [ ! -f .env ]; then
    echo "Création du fichier .env à partir de .env.example..."
    cp .env.example .env
    echo "Veuillez modifier le fichier .env avec vos paramètres avant de continuer."
    exit 1
fi

# Créer les répertoires nécessaires
mkdir -p logs data db

# Construire et démarrer les conteneurs
echo "Démarrage des services EVIL2ROOT Trading Bot..."
docker-compose up -d

echo "Vérification de l'état des services..."
sleep 5
docker-compose ps

echo "
EVIL2ROOT Trading Bot est maintenant en cours d'exécution !

- API: http://localhost:8000
- Frontend: http://localhost:3000
- Adminer (gestion de base de données): http://localhost:8080

Pour arrêter les services, exécutez: docker-compose down
Pour afficher les logs, exécutez: docker-compose logs -f
" 
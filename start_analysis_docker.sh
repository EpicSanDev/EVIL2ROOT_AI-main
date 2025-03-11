#!/bin/bash

# Script pour démarrer le bot d'analyse quotidienne avec Docker
# Utilisation: ./start_analysis_docker.sh [--build]

# Variables de couleur pour l'affichage
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier si l'utilisateur a demandé de reconstruire l'image
REBUILD=false
if [ "$1" == "--build" ]; then
    REBUILD=true
fi

echo -e "${YELLOW}Démarrage du bot d'analyse quotidienne avec Docker...${NC}"

# Vérifier si le fichier .env existe
if [ ! -f .env ]; then
    echo -e "${RED}Erreur: Fichier .env non trouvé.${NC}"
    echo -e "Veuillez créer un fichier .env avec les variables nécessaires:"
    echo -e " - TELEGRAM_TOKEN"
    echo -e " - TELEGRAM_CHAT_ID"
    echo -e " - OPENROUTER_API_KEY"
    echo -e " - SYMBOLS"
    exit 1
fi

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Erreur: Docker n'est pas installé.${NC}"
    echo -e "Veuillez installer Docker pour continuer."
    exit 1
fi

# Vérifier si Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Erreur: Docker Compose n'est pas installé.${NC}"
    echo -e "Veuillez installer Docker Compose pour continuer."
    exit 1
fi

# Créer les dossiers nécessaires s'ils n'existent pas
echo -e "Création des dossiers de données..."
mkdir -p data logs saved_models

# Reconstruire l'image si demandé
if [ "$REBUILD" = true ]; then
    echo -e "${YELLOW}Reconstruction de l'image Docker...${NC}"
    docker-compose build analysis-bot
fi

# Arrêter le service existant s'il est en cours d'exécution
echo -e "Arrêt du service existant s'il est en cours d'exécution..."
docker-compose stop analysis-bot
docker-compose rm -f analysis-bot

# Démarrer le service d'analyse
echo -e "${GREEN}Démarrage du service d'analyse...${NC}"
docker-compose up -d analysis-bot

# Vérifier si le service a démarré correctement
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Le bot d'analyse quotidienne a été démarré avec succès!${NC}"
    echo -e "Vous pouvez consulter les logs avec: docker-compose logs -f analysis-bot"
else
    echo -e "${RED}Erreur lors du démarrage du bot d'analyse.${NC}"
    echo -e "Consultez les logs pour plus d'informations: docker-compose logs analysis-bot"
    exit 1
fi

# Affichage des logs
echo -e "${YELLOW}Affichage des logs (Ctrl+C pour quitter):${NC}"
docker-compose logs -f analysis-bot 
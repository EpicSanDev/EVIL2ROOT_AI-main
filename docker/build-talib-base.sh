#!/bin/bash
# Script pour construire une image de base avec TA-Lib préinstallé

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Nom et tag de l'image
IMAGE_NAME="evil2root/talib-base"
IMAGE_TAG="3.10-slim"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Affichage d'information
echo -e "${BLUE}Construction de l'image de base avec TA-Lib préinstallé${NC}"
echo -e "${YELLOW}Image: ${FULL_IMAGE_NAME}${NC}"

# Vérifier si Docker est disponible
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé ou n'est pas disponible.${NC}"
    exit 1
fi

# Construire l'image
echo -e "${GREEN}Démarrage de la construction...${NC}"
docker build -t "${FULL_IMAGE_NAME}" -f docker/Dockerfile.talib-base .

# Vérifier si la construction a réussi
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Construction de l'image terminée avec succès!${NC}"
    
    # Demander s'il faut publier l'image
    read -p "Voulez-vous publier cette image sur Docker Hub? (o/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        echo -e "${YELLOW}Publication de l'image...${NC}"
        docker push "${FULL_IMAGE_NAME}"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Image publiée avec succès!${NC}"
            echo -e "${YELLOW}L'image est maintenant disponible à l'adresse: ${FULL_IMAGE_NAME}${NC}"
            
            # Mise à jour du Dockerfile principal pour utiliser cette image de base
            echo -e "${BLUE}Voulez-vous mettre à jour le Dockerfile principal pour utiliser cette image de base? (o/n) ${NC}"
            read -n 1 -r
            echo
            if [[ $REPLY =~ ^[Oo]$ ]]; then
                DOCKERFILE="Dockerfile"
                BACKUP_FILE="${DOCKERFILE}.bak"
                
                # Créer une sauvegarde
                cp "${DOCKERFILE}" "${BACKUP_FILE}"
                
                # Mettre à jour le Dockerfile
                echo -e "${YELLOW}Mise à jour du Dockerfile...${NC}"
                sed -i "s|FROM python:3.10-slim as builder|FROM ${FULL_IMAGE_NAME} as builder|" "${DOCKERFILE}"
                
                echo -e "${GREEN}Dockerfile mis à jour avec succès! Une sauvegarde a été créée: ${BACKUP_FILE}${NC}"
            fi
        else
            echo -e "${RED}Erreur lors de la publication de l'image.${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}L'image n'a pas été publiée. Elle est disponible localement sous le nom: ${FULL_IMAGE_NAME}${NC}"
    fi
else
    echo -e "${RED}Erreur lors de la construction de l'image.${NC}"
    exit 1
fi

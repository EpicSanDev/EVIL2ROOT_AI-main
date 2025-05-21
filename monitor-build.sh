#!/bin/bash
# Script pour surveiller et gérer l'installation des dépendances dans Docker

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'aide
show_help() {
    echo -e "${BLUE}Script de surveillance pour l'installation des dépendances Docker${NC}"
    echo ""
    echo "Ce script permet de surveiller l'installation des dépendances pendant"
    echo "le build Docker et d'intervenir en cas de timeout."
    echo ""
    echo "Options:"
    echo "  --timeout N       Définit le timeout en minutes (défaut: 60)"
    echo "  --help            Affiche cette aide"
    echo ""
    echo "Exemples d'utilisation:"
    echo "  ./monitor-build.sh --timeout 90    # Définit un timeout de 90 minutes"
    echo ""
}

# Valeurs par défaut
TIMEOUT_MINUTES=60

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT_MINUTES="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Option non reconnue: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Conversion du timeout en secondes
TIMEOUT_SECONDS=$((TIMEOUT_MINUTES * 60))
echo -e "${BLUE}Timeout configuré à ${TIMEOUT_MINUTES} minutes (${TIMEOUT_SECONDS} secondes)${NC}"

# Lancement du build avec un timeout
echo -e "${GREEN}Démarrage de la construction Docker...${NC}"
echo -e "${YELLOW}Le processus sera interrompu après ${TIMEOUT_MINUTES} minutes si non terminé${NC}"

# Fonction pour construire avec mock TA-Lib en cas de timeout
build_fallback() {
    echo -e "${YELLOW}Tentative de construction avec le mode fallback (--use-mock-talib)...${NC}"
    ./build-docker.sh --use-mock-talib
}

# Fonction pour construire avec seulement les dépendances essentielles en cas d'échec
build_minimal() {
    echo -e "${YELLOW}Tentative de construction minimale (--essential-only --use-mock-talib)...${NC}"
    ./build-docker.sh --essential-only --use-mock-talib
}

# Exécuter le build standard
if timeout ${TIMEOUT_SECONDS}s docker compose build; then
    echo -e "${GREEN}Construction terminée avec succès!${NC}"
    exit 0
else
    echo -e "${RED}Timeout ou erreur lors de la construction standard.${NC}"
    
    # Essayer la construction fallback
    echo -e "${YELLOW}Tentative avec un mode de construction alternatif...${NC}"
    if build_fallback; then
        echo -e "${GREEN}Construction fallback terminée avec succès!${NC}"
        exit 0
    else
        echo -e "${RED}Échec de la construction fallback.${NC}"
        
        # Dernière tentative avec la construction minimale
        echo -e "${YELLOW}Tentative avec la construction minimale...${NC}"
        if build_minimal; then
            echo -e "${GREEN}Construction minimale terminée avec succès!${NC}"
            echo -e "${YELLOW}Note: Seules les fonctionnalités de base seront disponibles.${NC}"
            exit 0
        else
            echo -e "${RED}Toutes les tentatives de construction ont échoué.${NC}"
            echo -e "${RED}Veuillez consulter les logs pour plus d'informations.${NC}"
            exit 1
        fi
    fi
fi

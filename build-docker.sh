#!/bin/bash
# Script pour construire l'image Docker avec différentes options pour éviter les timeouts

# Couleurs pour affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'aide
show_help() {
    echo -e "${BLUE}Script de construction Docker optimisé pour EVIL2ROOT_AI${NC}"
    echo ""
    echo "Options:"
    echo "  --use-mock-talib       Utilise un mock pour TA-Lib au lieu de compiler la version native"
    echo "  --skip-tensorflow      Ne pas installer TensorFlow (réduit le temps de build)"
    echo "  --skip-torch           Ne pas installer PyTorch (réduit le temps de build)"
    echo "  --essential-only       Installer uniquement les dépendances essentielles"
    echo "  --no-cache             Construire sans utiliser le cache Docker"
    echo "  --help                 Affiche cette aide"
    echo ""
    echo "Exemples:"
    echo "  ./build-docker.sh --use-mock-talib               # Construction rapide avec TA-Lib simulé"
    echo "  ./build-docker.sh --skip-tensorflow --skip-torch # Sans les frameworks ML lourds"
    echo "  ./build-docker.sh --essential-only               # Version minimale fonctionnelle"
    echo ""
}

# Initialisation des variables
USE_MOCK_TALIB=false
SKIP_OPTIONAL_DEPS=false
INSTALL_FULL_DEPS=true
NO_CACHE=""

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-mock-talib)
            USE_MOCK_TALIB=true
            shift
            ;;
        --skip-tensorflow)
            SKIP_OPTIONAL_DEPS=true
            shift
            ;;
        --skip-torch)
            SKIP_OPTIONAL_DEPS=true
            shift
            ;;
        --essential-only)
            INSTALL_FULL_DEPS=false
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
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

# Affichage de la configuration
echo -e "${BLUE}=== Configuration de build ===${NC}"
echo -e "TA-Lib Mock: ${YELLOW}$USE_MOCK_TALIB${NC}"
echo -e "Dépendances complètes: ${YELLOW}$INSTALL_FULL_DEPS${NC}"
echo -e "Ignorer dépendances optionnelles: ${YELLOW}$SKIP_OPTIONAL_DEPS${NC}"
echo -e "Utiliser cache Docker: ${YELLOW}$([ -z "$NO_CACHE" ] && echo "oui" || echo "non")${NC}"

# Construction de l'image avec les arguments appropriés
echo -e "${GREEN}Démarrage de la construction de l'image Docker...${NC}"
docker build $NO_CACHE \
    --build-arg USE_TALIB_MOCK=$USE_MOCK_TALIB \
    --build-arg INSTALL_FULL_DEPS=$INSTALL_FULL_DEPS \
    --build-arg SKIP_OPTIONAL_DEPS=$SKIP_OPTIONAL_DEPS \
    -t evil2root_ai:latest .

# Vérification du résultat
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Construction de l'image terminée avec succès!${NC}"
    echo -e "Pour exécuter le conteneur: ${YELLOW}docker run -p 8000:8000 evil2root_ai:latest${NC}"
else
    echo -e "${RED}Erreur lors de la construction de l'image.${NC}"
    echo -e "Essayez avec l'option ${YELLOW}--use-mock-talib${NC} pour une construction plus rapide."
    echo -e "Ou avec ${YELLOW}--essential-only${NC} pour une version minimale."
fi

#!/bin/bash
# Script pour construire l'image complète avec une méthode optimisée pour ARM64

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Nom et tag de l'image
IMAGE_NAME="evil2root-ai"
IMAGE_TAG="arm64"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Affichage d'information
echo -e "${BLUE}Construction de l'image optimisée pour ARM64${NC}"
echo -e "${YELLOW}Image: ${FULL_IMAGE_NAME}${NC}"

# Vérifier si Docker est disponible
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé ou n'est pas disponible.${NC}"
    exit 1
fi

# Vérifier l'architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" && "$ARCH" != "aarch64" ]]; then
    echo -e "${YELLOW}Attention: Votre architecture est $ARCH. Ce script est optimisé pour ARM64.${NC}"
    echo -e "${YELLOW}Voulez-vous continuer quand même? (o/n)${NC}"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Oo]$ ]]; then
        echo -e "${RED}Construction annulée.${NC}"
        exit 1
    fi
fi

# Vérifier les ressources système disponibles
echo -e "${BLUE}Vérification des ressources système...${NC}"
# Mémoire disponible en MB
MEM_AVAILABLE=$(free -m | awk '/^Mem:/{print $7}')
# Espace disque disponible en GB
DISK_AVAILABLE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/[A-Za-z]//g')

echo -e "${YELLOW}Mémoire disponible: ${MEM_AVAILABLE}MB${NC}"
echo -e "${YELLOW}Espace disque disponible: ${DISK_AVAILABLE}GB${NC}"

# Recommandations basées sur les ressources
if [[ $MEM_AVAILABLE -lt 4000 ]]; then
    echo -e "${RED}Attention: Mémoire faible détectée (< 4GB). Recommandé d'utiliser --essential-only --mock-talib${NC}"
fi

if (( $(echo "$DISK_AVAILABLE < 10" | bc -l) )); then
    echo -e "${RED}Attention: Espace disque faible (< 10GB). Recommandé d'utiliser --essential-only${NC}"
fi

# Paramètres supplémentaires
MOCK_TALIB=false
ESSENTIAL_ONLY=false
SKIP_ML_DEPS=false
BUILD_TIMEOUT=60  # En minutes

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-mock-talib|--mock-talib)
            MOCK_TALIB=true
            shift
            ;;
        --essential-only)
            ESSENTIAL_ONLY=true
            shift
            ;;
        --skip-ml)
            SKIP_ML_DEPS=true
            shift
            ;;
        --timeout)
            BUILD_TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo -e "${BLUE}Options disponibles:${NC}"
            echo -e "  --use-mock-talib    Utiliser une version mock de TA-Lib (plus rapide)"
            echo -e "  --essential-only    Installer uniquement les dépendances essentielles"
            echo -e "  --skip-ml           Ne pas installer les bibliothèques ML lourdes"
            echo -e "  --timeout N         Définir un timeout en minutes (défaut: 60)"
            echo -e "  --help              Afficher cette aide"
            exit 0
            ;;
        *)
            echo -e "${RED}Option non reconnue: $1${NC}"
            echo -e "Options valides: --use-mock-talib, --essential-only, --skip-ml, --timeout"
            echo -e "Utilisez --help pour plus d'informations"
            exit 1
            ;;
    esac
done

# Construction avec les arguments appropriés
BUILD_ARGS=""
if [ "$MOCK_TALIB" = "true" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg USE_TALIB_MOCK=true"
    echo -e "${YELLOW}→ Utilisation de TA-Lib mockup${NC}"
fi

if [ "$ESSENTIAL_ONLY" = "true" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg INSTALL_FULL_DEPS=false"
    echo -e "${YELLOW}→ Installation des dépendances essentielles uniquement${NC}"
fi

if [ "$SKIP_ML_DEPS" = "true" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg SKIP_OPTIONAL_DEPS=true"
    echo -e "${YELLOW}→ Installation sans les dépendances ML lourdes${NC}"
fi

BUILD_ARGS="$BUILD_ARGS --build-arg TARGETARCH=arm64"

echo -e "${GREEN}Démarrage de la construction avec les arguments suivants:${NC}"
echo -e "${YELLOW}$BUILD_ARGS${NC}"

# Fonction pour gérer le timeout
build_with_timeout() {
    # Utiliser timeout pour limiter le temps de build
    echo -e "${BLUE}Construction avec timeout de ${BUILD_TIMEOUT} minutes...${NC}"
    
    # Conversion en secondes
    TIMEOUT_SECONDS=$((BUILD_TIMEOUT * 60))
    
    # Démarrer la construction avec timeout
    timeout $TIMEOUT_SECONDS docker build $BUILD_ARGS -t "$FULL_IMAGE_NAME" .
    BUILD_STATUS=$?
    
    if [ $BUILD_STATUS -eq 124 ]; then
        echo -e "${RED}Timeout atteint après ${BUILD_TIMEOUT} minutes.${NC}"
        echo -e "${YELLOW}Tentative avec options de secours...${NC}"
        
        # Si timeout avec options standards, essayer avec mock
        if [ "$MOCK_TALIB" = "false" ]; then
            echo -e "${YELLOW}Tentative avec TA-Lib mock...${NC}"
            BUILD_ARGS="$BUILD_ARGS --build-arg USE_TALIB_MOCK=true"
            timeout $TIMEOUT_SECONDS docker build $BUILD_ARGS -t "$FULL_IMAGE_NAME" .
            BUILD_STATUS=$?
        fi
        
        # Si toujours en échec, essayer avec essential-only
        if [ $BUILD_STATUS -ne 0 ]; then
            echo -e "${YELLOW}Tentative avec dépendances essentielles uniquement...${NC}"
            BUILD_ARGS="$BUILD_ARGS --build-arg INSTALL_FULL_DEPS=false"
            timeout $TIMEOUT_SECONDS docker build $BUILD_ARGS -t "$FULL_IMAGE_NAME" .
            BUILD_STATUS=$?
        fi
    fi
    
    return $BUILD_STATUS
}

# Lancer la construction avec gestion de timeout
time build_with_timeout

# Vérifier si la construction a réussi
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Construction de l'image terminée avec succès!${NC}"
    echo -e "${YELLOW}Vous pouvez maintenant l'exécuter avec:${NC}"
    echo -e "${BLUE}docker run -p 8000:8000 $FULL_IMAGE_NAME${NC}"
else
    echo -e "${RED}Erreur lors de la construction de l'image.${NC}"
    echo -e "${YELLOW}Essayez avec les options:${NC}"
    echo -e "${BLUE}./docker/build-arm64.sh --use-mock-talib --essential-only${NC}"
    exit 1
fi

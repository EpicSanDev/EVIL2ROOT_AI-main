#!/bin/bash
# Script pour détecter l'architecture et recommander la meilleure méthode de build

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo
}

# Détecter l'architecture
print_header "Détection de l'architecture"
ARCH=$(uname -m)
OS=$(uname -s)

echo -e "${GREEN}Architecture détectée: ${ARCH}${NC}"
echo -e "${GREEN}Système d'exploitation: ${OS}${NC}"

# Détecter et vérifier les ressources système
print_header "Analyse des ressources système"

# Obtenir le nombre de CPUs
if [[ "$OS" == "Darwin" ]]; then
    CPU_COUNT=$(sysctl -n hw.ncpu)
    MEM_TOTAL=$(sysctl hw.memsize | awk '{print $2 / 1024 / 1024 / 1024}')
    MEM_TOTAL_ROUNDED=$(printf "%.2f" $MEM_TOTAL)
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
elif [[ "$OS" == "Linux" ]]; then
    CPU_COUNT=$(nproc)
    MEM_TOTAL=$(free -m | awk '/^Mem:/{print $2/1024}')
    MEM_TOTAL_ROUNDED=$(printf "%.2f" $MEM_TOTAL)
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
else
    CPU_COUNT="Inconnu"
    MEM_TOTAL_ROUNDED="Inconnu"
    DISK_SPACE="Inconnu"
fi

echo -e "${GREEN}CPUs disponibles: ${CPU_COUNT}${NC}"
echo -e "${GREEN}RAM disponible: ${MEM_TOTAL_ROUNDED} GB${NC}"
echo -e "${GREEN}Espace disque disponible: ${DISK_SPACE}${NC}"

# Vérifier si Docker est installé
print_header "Vérification de Docker"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker est installé${NC}"
    
    # Vérifier la version
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}Version: ${DOCKER_VERSION}${NC}"
    
    # Vérifier si Docker est en cours d'exécution
    if docker info &> /dev/null; then
        echo -e "${GREEN}Docker est en cours d'exécution${NC}"
        
        # Obtenir les limites de ressources Docker si disponibles
        DOCKER_INFO=$(docker info 2>/dev/null)
        DOCKER_MEM=$(echo "$DOCKER_INFO" | grep "Total Memory" || echo "Non disponible")
        DOCKER_CPU=$(echo "$DOCKER_INFO" | grep "CPUs" || echo "Non disponible")
        
        if [[ "$DOCKER_MEM" != "Non disponible" || "$DOCKER_CPU" != "Non disponible" ]]; then
            echo -e "${GREEN}Ressources Docker:${NC}"
            echo -e "${GREEN}${DOCKER_MEM}${NC}"
            echo -e "${GREEN}${DOCKER_CPU}${NC}"
        fi
    else
        echo -e "${YELLOW}Docker n'est pas en cours d'exécution${NC}"
    fi
else
    echo -e "${RED}Docker n'est pas installé${NC}"
    echo -e "${YELLOW}Veuillez installer Docker avant de continuer${NC}"
fi

# Recommandations basées sur l'architecture et les ressources
print_header "Recommandations pour la construction"

# Variables pour les recommandations
USE_ARM64_SPECIFIC=false
USE_MOCK_TALIB=false
USE_MINIMAL_BUILD=false
USE_MONITORED_BUILD=true

# Recommandations basées sur l'architecture
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    USE_ARM64_SPECIFIC=true
    echo -e "${YELLOW}→ Architecture ARM64 (Apple Silicon) détectée${NC}"
    
    # Vérifier si les scripts ARM64 existent
    if [ -f "docker/build-arm64.sh" ]; then
        echo -e "${GREEN}→ Scripts ARM64 spécifiques disponibles${NC}"
    else
        echo -e "${YELLOW}→ Scripts ARM64 spécifiques non trouvés, utilisation des méthodes génériques${NC}"
        USE_ARM64_SPECIFIC=false
    fi
fi

# Recommandations basées sur la mémoire
if [[ "$MEM_TOTAL_ROUNDED" != "Inconnu" ]]; then
    if (( $(echo "$MEM_TOTAL_ROUNDED < 4" | bc -l) )); then
        USE_MOCK_TALIB=true
        USE_MINIMAL_BUILD=true
        echo -e "${RED}→ Mémoire très limitée détectée (< 4GB)${NC}"
        echo -e "${YELLOW}→ Recommandation: Utiliser une construction minimale avec mock TA-Lib${NC}"
    elif (( $(echo "$MEM_TOTAL_ROUNDED < 8" | bc -l) )); then
        USE_MOCK_TALIB=true
        echo -e "${YELLOW}→ Mémoire limitée détectée (< 8GB)${NC}"
        echo -e "${YELLOW}→ Recommandation: Utiliser une construction avec mock TA-Lib${NC}"
    else
        echo -e "${GREEN}→ Mémoire suffisante détectée (>= 8GB)${NC}"
    fi
fi

# Recommandations basées sur les CPUs
if [[ "$CPU_COUNT" != "Inconnu" ]]; then
    if (( $CPU_COUNT < 4 )); then
        echo -e "${YELLOW}→ Nombre limité de CPUs détecté (< 4)${NC}"
        if [ "$USE_MOCK_TALIB" = false ]; then
            echo -e "${YELLOW}→ Recommandation: Construction surveillée pour éviter les timeouts${NC}"
        fi
    else
        echo -e "${GREEN}→ Nombre de CPUs suffisant détecté (>= 4)${NC}"
    fi
fi

# Recommandations basées sur l'espace disque
if [[ "$DISK_SPACE" != "Inconnu" ]]; then
    # Extraire le nombre de l'espace disque (supprimer les unités)
    DISK_NUM=$(echo $DISK_SPACE | sed 's/[A-Za-z]//g')
    DISK_UNIT=$(echo $DISK_SPACE | sed 's/[0-9.]//g')
    
    # Convertir en GB si nécessaire
    if [[ "$DISK_UNIT" == "T" || "$DISK_UNIT" == "Ti" ]]; then
        # Si en TB, convertir en GB (1 TB = 1000 GB)
        DISK_GB=$(echo "$DISK_NUM * 1000" | bc)
    elif [[ "$DISK_UNIT" == "M" || "$DISK_UNIT" == "Mi" ]]; then
        # Si en MB, convertir en GB (1 GB = 1000 MB)
        DISK_GB=$(echo "$DISK_NUM / 1000" | bc)
    else
        # Sinon supposer déjà en GB
        DISK_GB=$DISK_NUM
    fi
    
    if (( $(echo "$DISK_GB < 10" | bc -l) )); then
        USE_MINIMAL_BUILD=true
        echo -e "${RED}→ Espace disque très limité détecté (< 10GB)${NC}"
        echo -e "${YELLOW}→ Recommandation: Utiliser une construction minimale${NC}"
    elif (( $(echo "$DISK_GB < 20" | bc -l) )); then
        echo -e "${YELLOW}→ Espace disque limité détecté (< 20GB)${NC}"
        if [ "$USE_MINIMAL_BUILD" = false ]; then
            echo -e "${YELLOW}→ Recommandation: Considérer une construction avec --skip-optional-deps${NC}"
        fi
    else
        echo -e "${GREEN}→ Espace disque suffisant détecté (>= 20GB)${NC}"
    fi
fi

# Recommandation finale basée sur les facteurs détectés
print_header "Meilleure commande de build recommandée"

if [ "$USE_ARM64_SPECIFIC" = true ]; then
    # Recommandations pour ARM64
    if [ "$USE_MINIMAL_BUILD" = true ]; then
        echo -e "${YELLOW}make build-arm64-minimal${NC}"
        echo -e "${GREEN}# ou${NC}"
        echo -e "${YELLOW}./docker/build-arm64.sh --essential-only --use-mock-talib${NC}"
    elif [ "$USE_MOCK_TALIB" = true ]; then
        echo -e "${YELLOW}make build-arm64-mock${NC}"
        echo -e "${GREEN}# ou${NC}"
        echo -e "${YELLOW}./docker/build-arm64.sh --use-mock-talib${NC}"
    else
        echo -e "${YELLOW}make build-arm64${NC}"
        echo -e "${GREEN}# ou${NC}"
        echo -e "${YELLOW}./docker/build-arm64.sh${NC}"
    fi
    
    echo
    echo -e "${GREEN}Pour plus d'informations sur les builds ARM64:${NC}"
    echo -e "${YELLOW}cat ARM64_BUILD_GUIDE.md${NC}"
else
    # Recommandations génériques
    if [ "$USE_MINIMAL_BUILD" = true ]; then
        echo -e "${YELLOW}make build-minimal${NC}"
        echo -e "${GREEN}# ou${NC}"
        echo -e "${YELLOW}./build-docker.sh --essential-only --use-mock-talib${NC}"
    elif [ "$USE_MOCK_TALIB" = true ]; then
        echo -e "${YELLOW}make build-fast${NC}"
        echo -e "${GREEN}# ou${NC}"
        echo -e "${YELLOW}./build-docker.sh --use-mock-talib${NC}"
    elif [ "$USE_MONITORED_BUILD" = true ]; then
        echo -e "${YELLOW}make build-monitored${NC}"
        echo -e "${GREEN}# ou${NC}"
        echo -e "${YELLOW}./monitor-build.sh --timeout 60${NC}"
    else
        echo -e "${YELLOW}make build${NC}"
        echo -e "${GREEN}# ou${NC}"
        echo -e "${YELLOW}docker compose build${NC}"
    fi
    
    echo
    echo -e "${GREEN}Pour plus d'informations sur tous les builds disponibles:${NC}"
    echo -e "${YELLOW}cat DOCKER_BUILD_GUIDE.md${NC}"
fi

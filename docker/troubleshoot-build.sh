#!/bin/bash
# Script pour diagnostiquer et résoudre les problèmes de build Docker

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'aide
show_help() {
    echo -e "${BLUE}Diagnostic et résolution des problèmes de build Docker${NC}"
    echo ""
    echo "Ce script analyse les problèmes courants de build et propose des solutions."
    echo ""
    echo "Options:"
    echo "  --check-system   Vérifie la configuration système (mémoire, espace disque)"
    echo "  --fix-talib      Tente une réparation spécifique pour TA-Lib"
    echo "  --test-minimal   Teste une construction minimale"
    echo "  --help           Affiche cette aide"
    echo ""
}

# Fonction pour vérifier la configuration système
check_system() {
    echo -e "${BLUE}=== Vérification de la configuration système ===${NC}"
    
    # Vérifier l'espace disque
    echo -e "${YELLOW}Espace disque disponible:${NC}"
    df -h | grep -v "tmpfs" | grep -v "udev"
    
    # Vérifier la mémoire
    echo -e "${YELLOW}Mémoire disponible:${NC}"
    free -h
    
    # Vérifier si Docker est configuré correctement
    echo -e "${YELLOW}Configuration Docker:${NC}"
    docker info | grep -E "Total Memory|CPU|Storage Driver"
    
    # Vérifier les paramètres de Docker
    echo -e "${YELLOW}Vérification des paramètres Docker...${NC}"
    
    # Vérifier le cache Docker
    DOCKER_SIZE=$(docker system df | grep "Images" | awk '{print $4}')
    echo -e "Taille du cache Docker: ${DOCKER_SIZE}"
    
    if command -v docker-compose &> /dev/null; then
        echo -e "${GREEN}docker-compose est installé.${NC}"
    else
        echo -e "${RED}docker-compose n'est pas installé.${NC}"
        echo -e "Installez-le avec: pip install docker-compose"
    fi
    
    # Vérifier les variables d'environnement importantes
    echo -e "${YELLOW}Variables d'environnement importantes:${NC}"
    env | grep -E "DOCKER|PYTHON|PATH" | sort
}

# Fonction pour tester une solution spécifique pour TA-Lib
fix_talib() {
    echo -e "${BLUE}=== Tentative de réparation pour TA-Lib ===${NC}"
    
    # Vérifier si les scripts sont présents
    if [ ! -f "./docker/fix-talib-install.sh" ]; then
        echo -e "${RED}Script fix-talib-install.sh non trouvé!${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}1. Nettoyage du cache Docker pour les images liées à TA-Lib...${NC}"
    docker images | grep -i talib | awk '{print $3}' | xargs -r docker rmi -f
    
    echo -e "${YELLOW}2. Test d'installation directe de TA-Lib...${NC}"
    docker run --rm -it python:3.10-slim bash -c "apt-get update && apt-get install -y build-essential wget && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib/ && ./configure --prefix=/usr && make && make install && pip install numpy TA-Lib && python -c 'import talib; print(\"TA-Lib installé avec succès!\")'"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Installation réussie dans un conteneur de test!${NC}"
        echo -e "${YELLOW}Suggestion: Utilisez ./docker/build-talib-base.sh pour créer une image de base réutilisable.${NC}"
    else
        echo -e "${RED}L'installation a échoué.${NC}"
        echo -e "${YELLOW}Suggestion: Utilisez l'option --use-mock-talib avec le script build-docker.sh.${NC}"
    fi
}

# Fonction pour tester une construction minimale
test_minimal() {
    echo -e "${BLUE}=== Test de construction minimale ===${NC}"
    
    echo -e "${YELLOW}Construction d'une image minimale avec seulement les dépendances essentielles...${NC}"
    time ./build-docker.sh --essential-only --use-mock-talib
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Construction minimale réussie!${NC}"
        echo -e "${YELLOW}C'est une bonne base de départ. Vous pouvez maintenant ajouter progressivement plus de fonctionnalités.${NC}"
    else
        echo -e "${RED}Même la construction minimale a échoué.${NC}"
        echo -e "${YELLOW}Problèmes possibles:${NC}"
        echo -e "1. Configuration Docker incorrecte"
        echo -e "2. Problèmes de réseau ou de proxy"
        echo -e "3. Conflit entre packages Python"
        echo -e "4. Espace disque insuffisant"
        echo -e "5. Mémoire RAM insuffisante"
    fi
}

# Si aucun argument, afficher l'aide
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-system)
            check_system
            shift
            ;;
        --fix-talib)
            fix_talib
            shift
            ;;
        --test-minimal)
            test_minimal
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

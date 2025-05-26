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
    echo "  --arm64           Active les optimisations spécifiques pour ARM64"
    echo "  --aggressive      Mode agressif qui utilise toutes les optimisations disponibles"
    echo "  --help            Affiche cette aide"
    echo ""
    echo "Exemples d'utilisation:"
    echo "  ./monitor-build.sh --timeout 90                # Définit un timeout de 90 minutes"
    echo "  ./monitor-build.sh --arm64                     # Optimisations pour Apple Silicon"
    echo "  ./monitor-build.sh --arm64 --aggressive        # Mode ultra-optimisé pour ARM64"
    echo ""
}

# Valeurs par défaut
TIMEOUT_MINUTES=60
IS_ARM64=false
AGGRESSIVE_MODE=false

# Détection automatique de l'architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    IS_ARM64=true
    echo -e "${YELLOW}Architecture ARM64 détectée automatiquement.${NC}"
fi

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT_MINUTES="$2"
            shift 2
            ;;
        --arm64)
            IS_ARM64=true
            shift
            ;;
        --aggressive)
            AGGRESSIVE_MODE=true
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

# Conversion du timeout en secondes
TIMEOUT_SECONDS=$((TIMEOUT_MINUTES * 60))
echo -e "${BLUE}Timeout configuré à ${TIMEOUT_MINUTES} minutes (${TIMEOUT_SECONDS} secondes)${NC}"

# Afficher les options activées
if [ "$IS_ARM64" = true ]; then
    echo -e "${YELLOW}Mode ARM64 activé - Utilisation des optimisations spécifiques à Apple Silicon${NC}"
fi

if [ "$AGGRESSIVE_MODE" = true ]; then
    echo -e "${YELLOW}Mode agressif activé - Utilisation de toutes les optimisations possibles${NC}"
fi

# Lancement du build avec un timeout
echo -e "${GREEN}Démarrage de la construction Docker...${NC}"
echo -e "${YELLOW}Le processus sera interrompu après ${TIMEOUT_MINUTES} minutes si non terminé${NC}"

# Fonction pour construire avec mock TA-Lib en cas de timeout
build_fallback() {
    echo -e "${YELLOW}Tentative de construction avec le mode fallback (--use-mock-talib)...${NC}"
    if [ "$IS_ARM64" = true ]; then
        if [ -f "docker/build-arm64.sh" ]; then
            echo -e "${YELLOW}Utilisation du build ARM64 avec mock TA-Lib...${NC}"
            ./docker/build-arm64.sh --use-mock-talib
        else
            ./build-docker.sh --use-mock-talib
        fi
    else
        ./build-docker.sh --use-mock-talib
    fi
}

# Fonction pour construire avec seulement les dépendances essentielles en cas d'échec
build_minimal() {
    echo -e "${YELLOW}Tentative de construction minimale (--essential-only --use-mock-talib)...${NC}"
    if [ "$IS_ARM64" = true ]; then
        if [ -f "docker/build-arm64.sh" ]; then
            echo -e "${YELLOW}Utilisation du build ARM64 minimal...${NC}"
            ./docker/build-arm64.sh --essential-only --use-mock-talib
        else
            ./build-docker.sh --essential-only --use-mock-talib
        fi
    else
        ./build-docker.sh --essential-only --use-mock-talib
    fi
}

# Fonction pour le mode ultra-optimisé
build_ultra_optimized() {
    echo -e "${YELLOW}Tentative de construction ultra-optimisée...${NC}"
    if [ "$IS_ARM64" = true ]; then
        if [ -f "docker/arm64-troubleshoot.sh" ]; then
            echo -e "${YELLOW}Utilisation de la configuration ultra-optimisée ARM64...${NC}"
            ./docker/arm64-troubleshoot.sh --force-mock
            if [ -f "quick-build-arm64.sh" ]; then
                ./quick-build-arm64.sh
                return $?
            fi
        fi
    fi
    
    # Fallback au build minimal
    build_minimal
}

# Démarrer le build en fonction des options
if [ "$AGGRESSIVE_MODE" = true ]; then
    # Mode agressif: commencer directement par la version optimisée
    if [ "$IS_ARM64" = true ]; then
        echo -e "${YELLOW}Mode agressif ARM64 - Utilisation directe du build ultra-optimisé${NC}"
        if build_ultra_optimized; then
            echo -e "${GREEN}Construction ultra-optimisée terminée avec succès!${NC}"
            exit 0
        else
            echo -e "${RED}Échec de la construction ultra-optimisée.${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Mode agressif - Utilisation directe du build minimal${NC}"
        if build_minimal; then
            echo -e "${GREEN}Construction minimale terminée avec succès!${NC}"
            exit 0
        else
            echo -e "${RED}Échec de la construction minimale.${NC}"
            exit 1
        fi
    fi
else
    # Mode standard avec failsafe
    if [ "$IS_ARM64" = true ] && [ -f "docker/build-arm64.sh" ]; then
        # Build ARM64 natif
        echo -e "${YELLOW}Utilisation du build ARM64 natif...${NC}"
        if timeout ${TIMEOUT_SECONDS}s ./docker/build-arm64.sh; then
            echo -e "${GREEN}Construction ARM64 terminée avec succès!${NC}"
            exit 0
        else
            echo -e "${RED}Timeout ou erreur lors de la construction ARM64.${NC}"
        fi
    else
        # Build standard
        if timeout ${TIMEOUT_SECONDS}s docker compose build; then
            echo -e "${GREEN}Construction terminée avec succès!${NC}"
            exit 0
        else
            echo -e "${RED}Timeout ou erreur lors de la construction standard.${NC}"
        fi
    fi
    
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
            
            # Ultime tentative avec le mode ultra-optimisé pour ARM64
            if [ "$IS_ARM64" = true ]; then
                echo -e "${YELLOW}Tentative ultime avec le mode ultra-optimisé ARM64...${NC}"
                if build_ultra_optimized; then
                    echo -e "${GREEN}Construction ultra-optimisée réussie après tous les échecs!${NC}"
                    exit 0
                fi
            fi
            
            echo -e "${RED}Veuillez consulter les logs pour plus d'informations.${NC}"
            echo -e "${YELLOW}Conseil: Pour ARM64, consultez le guide ARM64_BUILD_GUIDE.md${NC}"
            exit 1
        fi
    fi
fi

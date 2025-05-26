#!/bin/bash
# Script de dépannage spécifique pour ARM64 (Apple Silicon)
set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo
}

# Fonction d'aide
show_help() {
    echo -e "${BLUE}Script de dépannage pour ARM64 (Apple Silicon)${NC}"
    echo
    echo "Ce script aide à diagnostiquer et résoudre les problèmes spécifiques"
    echo "à l'architecture ARM64 (M1/M2/M3) pour le build Docker de EVIL2ROOT AI."
    echo
    echo "Options:"
    echo "  --diagnose           Diagnostiquer l'environnement et les problèmes potentiels"
    echo "  --fix-talib          Tenter de réparer l'installation de TA-Lib"
    echo "  --check-rosetta      Vérifier si Rosetta est nécessaire et installé"
    echo "  --prepare-env        Préparer l'environnement pour un build optimal"
    echo "  --force-mock         Forcer l'utilisation de la version mock de TA-Lib"
    echo "  --help               Afficher cette aide"
    echo
    echo "Exemples d'utilisation:"
    echo "  ./arm64-troubleshoot.sh --diagnose    # Diagnostiquer l'environnement"
    echo "  ./arm64-troubleshoot.sh --fix-talib   # Réparer TA-Lib"
    echo
}

# Vérifier l'architecture
check_architecture() {
    ARCH=$(uname -m)
    if [[ "$ARCH" != "arm64" && "$ARCH" != "aarch64" ]]; then
        echo -e "${RED}Ce script est conçu pour l'architecture ARM64 (Apple Silicon).${NC}"
        echo -e "${YELLOW}Votre architecture détectée est: ${ARCH}${NC}"
        echo
        echo -e "${YELLOW}Voulez-vous continuer quand même? (o/n)${NC}"
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Oo]$ ]]; then
            echo -e "${RED}Script terminé.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Architecture ARM64 détectée: ${ARCH}${NC}"
    fi
}

# Diagnostiquer l'environnement
diagnose_environment() {
    print_banner "Diagnostic de l'environnement ARM64"
    
    # Vérifier Docker
    echo -e "${YELLOW}Vérification de Docker...${NC}"
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker n'est pas installé.${NC}"
        echo -e "${YELLOW}Installez Docker Desktop pour Mac avec le support ARM64.${NC}"
    else
        echo -e "${GREEN}Docker est installé.${NC}"
        # Vérifier la version de Docker
        docker_version=$(docker --version)
        echo -e "${GREEN}Version: ${docker_version}${NC}"
        
        # Vérifier les paramètres de Docker
        echo -e "${YELLOW}Vérification des paramètres Docker...${NC}"
        docker_mem=$(docker info 2>/dev/null | grep "Total Memory" || echo "Non disponible")
        docker_cpu=$(docker info 2>/dev/null | grep "CPUs" || echo "Non disponible")
        echo -e "${GREEN}Ressources Docker: ${docker_mem}, ${docker_cpu}${NC}"
        
        # Vérifier Rosetta (pour Docker)
        echo -e "${YELLOW}Vérification de l'option Rosetta dans Docker...${NC}"
        if pgrep -f "docker desktop" > /dev/null; then
            echo -e "${GREEN}Docker Desktop est en cours d'exécution.${NC}"
            echo -e "${YELLOW}Conseil: Vérifiez que l'option 'Use Rosetta for x86/amd64 emulation' est activée${NC}"
            echo -e "${YELLOW}dans Docker Desktop > Paramètres > Général pour une meilleure compatibilité.${NC}"
        else
            echo -e "${YELLOW}Docker Desktop ne semble pas en cours d'exécution.${NC}"
        fi
    fi
    
    # Vérifier les ressources système
    echo -e "${YELLOW}Vérification des ressources système...${NC}"
    # Mémoire totale en GB
    mem_total=$(sysctl hw.memsize | awk '{print $2 / 1024 / 1024 / 1024}')
    mem_total_rounded=$(printf "%.2f" $mem_total)
    echo -e "${GREEN}Mémoire totale: ${mem_total_rounded} GB${NC}"
    
    # CPUs
    cpu_count=$(sysctl -n hw.ncpu)
    echo -e "${GREEN}Nombre de CPUs: ${cpu_count}${NC}"
    
    # Modèle de processeur
    cpu_model=$(sysctl -n machdep.cpu.brand_string)
    echo -e "${GREEN}Processeur: ${cpu_model}${NC}"
    
    # Espace disque
    disk_space=$(df -h . | awk 'NR==2 {print $4}')
    echo -e "${GREEN}Espace disque disponible: ${disk_space}${NC}"
    
    # Recommandations
    echo
    echo -e "${YELLOW}Recommandations basées sur votre configuration:${NC}"
    
    if (( $(echo "$mem_total < 8" | bc -l) )); then
        echo -e "${RED}- Mémoire faible détectée: Utilisez --essential-only et --use-mock-talib${NC}"
    elif (( $(echo "$mem_total < 16" | bc -l) )); then
        echo -e "${YELLOW}- Mémoire moyenne: Considérez --use-mock-talib pour éviter les timeouts${NC}"
    else
        echo -e "${GREEN}- Bonne configuration mémoire${NC}"
    fi
    
    # Recommandations pour Docker
    echo -e "${YELLOW}Recommandations pour Docker:${NC}"
    echo -e "${GREEN}- Assurez-vous que Docker dispose d'au moins 4GB de RAM dans les paramètres${NC}"
    echo -e "${GREEN}- Activez 'Use Rosetta for x86/amd64 emulation' pour une meilleure compatibilité${NC}"
    echo -e "${GREEN}- Privilégiez les builds natifs ARM64 pour de meilleures performances${NC}"
}

# Vérifier et réparer TA-Lib
fix_talib() {
    print_banner "Réparation de TA-Lib pour ARM64"
    
    # Vérifier si les scripts existent
    if [ ! -f "docker/improved-talib-install.sh" ] || [ ! -f "docker/talib-arm64-mock.sh" ]; then
        echo -e "${RED}Scripts d'installation TA-Lib non trouvés.${NC}"
        echo -e "${YELLOW}Assurez-vous d'être dans le répertoire racine du projet.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Préparation de l'environnement Docker pour TA-Lib...${NC}"
    
    # Créer une image de test pour diagnostiquer TA-Lib
    echo -e "${YELLOW}Création d'une image de test pour diagnostiquer TA-Lib...${NC}"
    
    # Créer un Dockerfile temporaire
    cat > tmp_talib_test.Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        gcc \
        g++ \
        make \
        pkg-config \
        curl \
        ca-certificates \
        autoconf \
        libtool \
        git

# Copier les scripts de diagnostic
COPY docker/improved-talib-install.sh /tmp/
COPY docker/talib-arm64-mock.sh /tmp/
RUN chmod +x /tmp/improved-talib-install.sh /tmp/talib-arm64-mock.sh

# Installer numpy
RUN pip install numpy==1.24.3

# Point d'entrée pour le diagnostic
CMD ["bash"]
EOF
    
    echo -e "${YELLOW}Construction de l'image de diagnostic...${NC}"
    docker build -t talib-arm64-test -f tmp_talib_test.Dockerfile .
    
    echo -e "${GREEN}Image de diagnostic construite avec succès!${NC}"
    echo -e "${YELLOW}Exécution du diagnostic TA-Lib...${NC}"
    
    # Exécuter le conteneur avec le script de diagnostic
    docker run --rm -it talib-arm64-test bash -c "set -e; echo 'Test installation TA-Lib...'; /tmp/improved-talib-install.sh || { echo 'Installation standard échouée, test du mock...'; /tmp/talib-arm64-mock.sh; }"
    
    # Nettoyer
    echo -e "${YELLOW}Nettoyage des fichiers temporaires...${NC}"
    rm -f tmp_talib_test.Dockerfile
    
    echo -e "${GREEN}Diagnostic TA-Lib terminé.${NC}"
    echo -e "${YELLOW}Si l'installation a échoué dans le conteneur de test, utilisez l'option --force-mock${NC}"
}

# Vérifier Rosetta
check_rosetta() {
    print_banner "Vérification de Rosetta 2"
    
    echo -e "${YELLOW}Vérification de l'installation de Rosetta 2...${NC}"
    
    # Vérifier si Rosetta est installé
    if /usr/bin/pgrep -q oahd; then
        echo -e "${GREEN}Rosetta 2 est installé et en cours d'exécution.${NC}"
    else
        echo -e "${YELLOW}Rosetta 2 ne semble pas être en cours d'exécution.${NC}"
        echo -e "${YELLOW}Cela peut affecter la compatibilité avec certains conteneurs x86_64.${NC}"
        
        echo -e "${YELLOW}Voulez-vous installer Rosetta 2? (o/n)${NC}"
        read -n 1 -r
        echo
        if [[ $REPLY =~ ^[Oo]$ ]]; then
            echo -e "${YELLOW}Installation de Rosetta 2...${NC}"
            softwareupdate --install-rosetta --agree-to-license
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Rosetta 2 installé avec succès!${NC}"
            else
                echo -e "${RED}Erreur lors de l'installation de Rosetta 2.${NC}"
            fi
        fi
    fi
    
    echo -e "${YELLOW}Vérification de l'option Rosetta dans Docker Desktop...${NC}"
    echo -e "${YELLOW}Assurez-vous que l'option 'Use Rosetta for x86/amd64 emulation' est activée${NC}"
    echo -e "${YELLOW}dans Docker Desktop > Paramètres > Fonctionnalités expérimentales${NC}"
}

# Préparer l'environnement
prepare_environment() {
    print_banner "Préparation de l'environnement pour ARM64"
    
    echo -e "${YELLOW}Configuration de l'environnement optimal pour ARM64...${NC}"
    
    # Créer un script pour optimiser Docker
    echo -e "${YELLOW}Création d'un script d'optimisation Docker...${NC}"
    
    cat > optimize-docker-arm64.sh << 'EOF'
#!/bin/bash
# Script pour optimiser Docker sur ARM64

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Optimisation de Docker pour ARM64${NC}"

# Arrêter Docker s'il est en cours d'exécution
echo -e "${YELLOW}Arrêt de Docker...${NC}"
osascript -e 'quit app "Docker"'
sleep 2

# Créer le fichier de configuration Docker si nécessaire
DOCKER_CONFIG="$HOME/.docker/config.json"
mkdir -p "$HOME/.docker"

if [ ! -f "$DOCKER_CONFIG" ]; then
    echo -e "${YELLOW}Création du fichier de configuration Docker...${NC}"
    echo "{}" > "$DOCKER_CONFIG"
fi

# Mettre à jour la configuration pour optimiser pour ARM64
echo -e "${YELLOW}Mise à jour de la configuration Docker...${NC}"
cat > "$DOCKER_CONFIG" << 'CONF'
{
  "experimental": true,
  "builder": {
    "gc": {
      "enabled": true,
      "defaultKeepStorage": "20GB"
    }
  },
  "features": {
    "buildkit": true
  }
}
CONF

echo -e "${GREEN}Configuration Docker mise à jour avec les paramètres optimaux pour ARM64.${NC}"
echo -e "${YELLOW}Redémarrez Docker Desktop manuellement et assurez-vous que:${NC}"
echo -e "${YELLOW}1. Dans Paramètres > Ressources:${NC}"
echo -e "${YELLOW}   - Allouez au moins 4GB de RAM${NC}"
echo -e "${YELLOW}   - Allouez au moins 2 CPUs${NC}"
echo -e "${YELLOW}2. Dans Paramètres > Fonctionnalités:${NC}"
echo -e "${YELLOW}   - Activez 'Use Rosetta for x86/amd64 emulation'${NC}"
echo -e "${YELLOW}   - Activez 'VirtioFS'${NC}"
echo -e "${GREEN}Ces paramètres optimiseront les performances des builds Docker sur ARM64.${NC}"
EOF
    
    chmod +x optimize-docker-arm64.sh
    
    echo -e "${GREEN}Script d'optimisation créé: optimize-docker-arm64.sh${NC}"
    echo -e "${YELLOW}Exécutez ./optimize-docker-arm64.sh pour appliquer les optimisations.${NC}"
    
    # Créer des variables d'environnement optimales pour les builds
    echo -e "${YELLOW}Création d'un fichier d'environnement pour les builds ARM64...${NC}"
    
    cat > .env.arm64 << 'EOF'
# Variables d'environnement optimisées pour ARM64
DOCKER_BUILDKIT=1
COMPOSE_DOCKER_CLI_BUILD=1
DOCKER_DEFAULT_PLATFORM=linux/arm64
PYTHON_OPTIMIZE=2
PIP_DEFAULT_TIMEOUT=200
EOF
    
    echo -e "${GREEN}Fichier d'environnement créé: .env.arm64${NC}"
    echo -e "${YELLOW}Utilisez source .env.arm64 avant de lancer un build.${NC}"
    
    echo -e "${GREEN}Environnement préparé pour ARM64!${NC}"
}

# Forcer l'utilisation du mock TA-Lib
force_mock_talib() {
    print_banner "Configuration pour utiliser le mock TA-Lib"
    
    # Créer un Dockerfile personnalisé avec mock TA-Lib
    echo -e "${YELLOW}Création d'un Dockerfile optimisé pour ARM64 avec mock TA-Lib...${NC}"
    
    cat > Dockerfile.arm64-optimized << 'EOF'
# Stage 1: Builder optimisé pour ARM64
FROM python:3.10-slim as builder

WORKDIR /app

# Variables d'environnement pour le build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=200 \
    PYTHONIOENCODING=UTF-8

# Copie des fichiers requirements et scripts
COPY requirements-essential.txt ./
COPY docker/talib-arm64-mock.sh /tmp/
RUN chmod +x /tmp/talib-arm64-mock.sh

# Installer les dépendances système et de compilation nécessaires
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*

# Installation du mock TA-Lib optimisé pour ARM64
RUN /tmp/talib-arm64-mock.sh

# Installation des dépendances essentielles
RUN pip install --no-cache-dir -r requirements-essential.txt

# Stage 2: Image finale
FROM python:3.10-slim

WORKDIR /app

# Variables d'environnement pour la production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

# Copier les dépendances Python du builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copier l'application
COPY . .

# Installer les dépendances système minimales nécessaires à l'exécution
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Entrypoint
CMD ["./docker-entrypoint.sh"]
EOF
    
    echo -e "${GREEN}Dockerfile optimisé créé: Dockerfile.arm64-optimized${NC}"
    echo -e "${YELLOW}Utilisez-le avec la commande:${NC}"
    echo -e "${BLUE}docker build -t evil2root-ai:arm64-mock -f Dockerfile.arm64-optimized .${NC}"
    
    # Créer un script de build rapide
    echo -e "${YELLOW}Création d'un script de build rapide...${NC}"
    
    cat > quick-build-arm64.sh << 'EOF'
#!/bin/bash
# Script pour un build rapide sur ARM64 avec mock TA-Lib

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Build rapide ARM64 avec mock TA-Lib${NC}"

# Vérifier l'architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" && "$ARCH" != "aarch64" ]]; then
    echo -e "${YELLOW}Attention: Architecture non-ARM64 détectée (${ARCH})${NC}"
fi

# Activer BuildKit
export DOCKER_BUILDKIT=1

# Construire l'image
echo -e "${YELLOW}Construction de l'image...${NC}"
time docker build -t evil2root-ai:arm64-mock -f Dockerfile.arm64-optimized .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build terminé avec succès!${NC}"
    echo -e "${YELLOW}Vous pouvez maintenant démarrer l'application avec:${NC}"
    echo -e "${BLUE}docker run -p 8000:8000 evil2root-ai:arm64-mock${NC}"
else
    echo -e "${RED}Erreur lors du build.${NC}"
fi
EOF
    
    chmod +x quick-build-arm64.sh
    
    echo -e "${GREEN}Script de build rapide créé: quick-build-arm64.sh${NC}"
    echo -e "${YELLOW}Utilisez ./quick-build-arm64.sh pour un build rapide optimisé.${NC}"
}

# Vérifier si aucun argument n'est fourni
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Analyse des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --diagnose)
            check_architecture
            diagnose_environment
            shift
            ;;
        --fix-talib)
            check_architecture
            fix_talib
            shift
            ;;
        --check-rosetta)
            check_architecture
            check_rosetta
            shift
            ;;
        --prepare-env)
            check_architecture
            prepare_environment
            shift
            ;;
        --force-mock)
            check_architecture
            force_mock_talib
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Option non reconnue: $1${NC}"
            echo -e "Utilisez --help pour voir les options disponibles."
            exit 1
            ;;
    esac
done

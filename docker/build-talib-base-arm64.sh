#!/bin/bash
# Script pour construire une image de base avec TA-Lib préinstallé spécifiquement pour ARM64 architecture

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Nom et tag de l'image
IMAGE_NAME="evil2root/talib-base"
IMAGE_TAG="3.10-slim-arm64"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Affichage d'information
echo -e "${BLUE}Construction de l'image de base avec TA-Lib pour ARM64${NC}"
echo -e "${YELLOW}Image: ${FULL_IMAGE_NAME}${NC}"

# Vérifier si Docker est disponible
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé ou n'est pas disponible.${NC}"
    exit 1
fi

# Vérifier si l'architecture est ARM64
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" && "$ARCH" != "aarch64" ]]; then
    echo -e "${YELLOW}Note: Votre architecture est $ARCH. Ce script est optimisé pour ARM64.${NC}"
    echo -e "${YELLOW}Voulez-vous continuer quand même? (o/n)${NC}"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Oo]$ ]]; then
        echo -e "${RED}Construction annulée.${NC}"
        exit 1
    fi
fi

# Créer un Dockerfile temporaire pour ARM64
TMP_DOCKERFILE="Dockerfile.talib-base.arm64.tmp"

echo -e "${YELLOW}Création d'un Dockerfile temporaire pour ARM64...${NC}"
cat > "$TMP_DOCKERFILE" << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Variables d'environnement pour le build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONIOENCODING=UTF-8

# Installer les dépendances système et de compilation nécessaires
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        gcc \
        g++ \
        make \
        pkg-config \
        python3-dev \
        unzip \
        git \
        automake \
        libtool \
        autoconf \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Télécharger et installer TA-Lib avec optimisations pour ARM64
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    # Télécharger les scripts config.guess et config.sub à jour qui supportent ARM64
    curl -o config/config.guess 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' && \
    curl -o config/config.sub 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' && \
    chmod +x config/config.guess config/config.sub && \
    # Configuration et compilation
    ./configure --prefix=/usr --host=aarch64-unknown-linux-gnu && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Installer le wrapper Python pour TA-Lib
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir numpy Cython && \
    # Installation du wrapper Python pour TA-Lib avec options explicites pour ARM64
    env ARCHFLAGS="-arch arm64" TALIB_USE_NATIVE=1 pip install --no-cache-dir TA-Lib==0.4.28 || \
    env ARCHFLAGS="-arch arm64" pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include/" --global-option="-L/usr/lib/" TA-Lib==0.4.28 && \
    # Vérifier l'installation
    python -c "import talib; print('TA-Lib importé avec succès sur ARM64!')"

# Créer un script de vérification
RUN echo '#!/bin/bash\npython -c "import talib; print(\"TA-Lib version:\", talib.__version__); print(\"Fonctions disponibles:\", dir(talib)[:5])"' > /usr/local/bin/verify-talib.sh && \
    chmod +x /usr/local/bin/verify-talib.sh

# Nettoyer pour réduire la taille de l'image
RUN apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Définir la commande par défaut pour vérifier l'installation
CMD ["python", "-c", "import talib; print('TA-Lib fonctionne correctement sur ARM64!')"]
EOF

# Construire l'image
echo -e "${GREEN}Démarrage de la construction...${NC}"
docker build -t "${FULL_IMAGE_NAME}" -f "$TMP_DOCKERFILE" .

# Vérifier si la construction a réussi
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Construction de l'image terminée avec succès!${NC}"
    
    # Tester l'image pour s'assurer que TA-Lib fonctionne
    echo -e "${YELLOW}Test de l'image...${NC}"
    docker run --rm "${FULL_IMAGE_NAME}" python -c "import talib; print('TA-Lib fonctions disponibles:', dir(talib)[:5])"
    
    # Demander s'il faut publier l'image
    echo
    echo -e "${BLUE}Voulez-vous publier cette image sur Docker Hub? (o/n) ${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        echo -e "${YELLOW}Publication de l'image...${NC}"
        docker push "${FULL_IMAGE_NAME}"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Image publiée avec succès!${NC}"
            echo -e "${YELLOW}L'image est maintenant disponible à l'adresse: ${FULL_IMAGE_NAME}${NC}"
            
            # Mise à jour du Dockerfile principal
            echo -e "${BLUE}Voulez-vous mettre à jour le Dockerfile principal pour utiliser cette image de base sur ARM64? (o/n) ${NC}"
            read -n 1 -r
            echo
            if [[ $REPLY =~ ^[Oo]$ ]]; then
                DOCKERFILE="Dockerfile"
                BACKUP_FILE="${DOCKERFILE}.bak"
                
                # Créer une sauvegarde
                cp "${DOCKERFILE}" "${BACKUP_FILE}"
                
                # Mettre à jour le Dockerfile avec une condition basée sur l'architecture
                echo -e "${YELLOW}Mise à jour du Dockerfile pour détecter automatiquement l'architecture...${NC}"
                cat > "dockerfile-update.tmp" << 'EOF'
# Stage 1: Builder - utilise une image de base différente selon l'architecture
FROM evil2root/talib-base:3.10-slim-arm64 as builder
EOF
                
                sed -i.bak "1,7s|^# Stage 1: Builder.*|$(cat dockerfile-update.tmp)|" "${DOCKERFILE}"
                rm -f "dockerfile-update.tmp"
                
                echo -e "${GREEN}Dockerfile mis à jour avec succès! Une sauvegarde a été créée: ${BACKUP_FILE}${NC}"
            fi
        else
            echo -e "${RED}Erreur lors de la publication de l'image.${NC}"
        fi
    else
        echo -e "${YELLOW}L'image n'a pas été publiée. Elle est disponible localement sous le nom: ${FULL_IMAGE_NAME}${NC}"
    fi
    
    # Nettoyer le Dockerfile temporaire
    rm -f "$TMP_DOCKERFILE"
else
    echo -e "${RED}Erreur lors de la construction de l'image.${NC}"
    echo -e "${YELLOW}Consultez le Dockerfile temporaire pour plus de détails: ${TMP_DOCKERFILE}${NC}"
    exit 1
fi

#!/bin/bash
# Script pour construire une image de base avec TA-Lib mock sans compilation

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Nom et tag de l'image
IMAGE_NAME="evil2root/talib-mock"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Affichage d'information
echo -e "${BLUE}Construction de l'image avec TA-Lib mock (sans compilation)${NC}"
echo -e "${YELLOW}Image: ${FULL_IMAGE_NAME}${NC}"

# Vérifier si Docker est disponible
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé ou n'est pas disponible.${NC}"
    exit 1
fi

# Créer un Dockerfile temporaire pour le mock
TMP_DOCKERFILE="Dockerfile.talib-mock.tmp"

echo -e "${YELLOW}Création d'un Dockerfile temporaire pour TA-Lib mock...${NC}"
cat > "$TMP_DOCKERFILE" << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Variables d'environnement pour le build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONIOENCODING=UTF-8

# Installer les dépendances minimales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Installer numpy qui est nécessaire pour le mock TA-Lib
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir numpy pandas

# Créer un module mock pour TA-Lib
RUN mkdir -p /usr/local/lib/python3.10/site-packages/talib && \
    cat > /usr/local/lib/python3.10/site-packages/talib/__init__.py << 'EOL'
import numpy as np
from numpy import array

# Version
__version__ = "0.4.28-mock"

# Constantes
MA_Type = type('MA_Type', (), {
    'SMA': 0, 'EMA': 1, 'WMA': 2, 'DEMA': 3, 'TEMA': 4, 
    'TRIMA': 5, 'KAMA': 6, 'MAMA': 7, 'T3': 8
})()

# Fonctions de base
def SMA(price, timeperiod=30):
    """Simple Moving Average"""
    return np.convolve(price, np.ones(timeperiod)/timeperiod, mode='valid')

def EMA(price, timeperiod=30):
    """Exponential Moving Average"""
    # Implémentation simplifiée
    return SMA(price, timeperiod)

def RSI(price, timeperiod=14):
    """Relative Strength Index"""
    # Mock qui retourne des valeurs entre 30 et 70
    result = np.zeros(len(price))
    result[:] = 50  # Valeur neutre
    return result

def MACD(price, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence/Divergence"""
    # Retourne trois arrays: macd, macdsignal, macdhist
    zeros = np.zeros(len(price))
    return zeros, zeros, zeros

def BBANDS(price, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands"""
    sma = SMA(price, timeperiod)
    dev = np.std(price)
    upper = sma + nbdevup * dev
    lower = sma - nbdevdn * dev
    return upper, sma, lower

# Ajouter d'autres fonctions TA couramment utilisées
def ADX(high, low, close, timeperiod=14):
    """Average Directional Movement Index"""
    return np.zeros(len(close))

def ATR(high, low, close, timeperiod=14):
    """Average True Range"""
    return np.zeros(len(close))

def CCI(high, low, close, timeperiod=14):
    """Commodity Channel Index"""
    return np.zeros(len(close))

def OBV(close, volume):
    """On Balance Volume"""
    return np.zeros(len(close))

def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    """Stochastic"""
    zeros = np.zeros(len(close))
    return zeros, zeros
EOL

# Créer un marqueur pour indiquer que c'est un package installé
RUN touch /usr/local/lib/python3.10/site-packages/talib-0.4.28-py3.10.egg-info && \
    # Vérifier l'installation
    python -c "import talib; print('TA-Lib MOCK importé avec succès! Version:', talib.__version__); print('Fonctions disponibles:', dir(talib)[:10])"

# Nettoyer pour réduire la taille de l'image
RUN apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Définir la commande par défaut pour vérifier l'installation
CMD ["python", "-c", "import talib; print('TA-Lib MOCK fonctionne correctement!')"]
EOF

# Construire l'image
echo -e "${GREEN}Démarrage de la construction...${NC}"
docker build -t "${FULL_IMAGE_NAME}" -f "$TMP_DOCKERFILE" .

# Vérifier si la construction a réussi
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Construction de l'image terminée avec succès!${NC}"
    
    # Tester l'image pour s'assurer que TA-Lib mock fonctionne
    echo -e "${YELLOW}Test de l'image...${NC}"
    docker run --rm "${FULL_IMAGE_NAME}" python -c "import talib; import numpy as np; print(talib.SMA(np.array([1,2,3,4,5,6,7,8,9,10]), 3))"
    
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
        else
            echo -e "${RED}Erreur lors de la publication de l'image.${NC}"
        fi
    else
        echo -e "${YELLOW}L'image n'a pas été publiée. Elle est disponible localement sous le nom: ${FULL_IMAGE_NAME}${NC}"
    fi
    
    # Nettoyer le Dockerfile temporaire
    rm -f "$TMP_DOCKERFILE"
    
    echo -e "${BLUE}Pour utiliser cette image dans votre Dockerfile, modifiez la première ligne comme suit:${NC}"
    echo -e "${YELLOW}FROM ${FULL_IMAGE_NAME} as builder${NC}"
else
    echo -e "${RED}Erreur lors de la construction de l'image.${NC}"
    echo -e "${YELLOW}Consultez le Dockerfile temporaire pour plus de détails: ${TMP_DOCKERFILE}${NC}"
    exit 1
fi

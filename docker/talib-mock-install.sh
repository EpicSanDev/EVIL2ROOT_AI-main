#!/bin/bash
# Script pour un dernier recours - installation d'une version minimale/mock de TA-Lib
# À n'utiliser que si toutes les autres méthodes ont échoué

set -e

echo "Installation du pseudo-TA-Lib (version de secours)..."

# Créer un répertoire temporaire
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Créer un module TA-Lib minimal qui fournit les fonctions de base
mkdir -p talib
cat > talib/__init__.py << 'EOL'
# Pseudo-TA-Lib implementation
import numpy as np
import warnings

warnings.warn("Using minimal TA-Lib mock implementation. Limited functionality available.")

# Version information
__version__ = "0.4.28-mock"

# Basic MA function as an example
def MA(real, timeperiod=30, matype=0):
    """Moving Average"""
    if not isinstance(real, np.ndarray):
        real = np.array(real)
    
    result = np.full_like(real, np.nan)
    for i in range(timeperiod-1, len(real)):
        result[i] = np.mean(real[i-timeperiod+1:i+1])
    
    return result

# Basic SMA
def SMA(real, timeperiod=30):
    """Simple Moving Average"""
    return MA(real, timeperiod=timeperiod)

# Stubs for other common functions
def RSI(real, timeperiod=14):
    """Relative Strength Index - Minimal implementation"""
    warnings.warn("Using minimal RSI implementation")
    if not isinstance(real, np.ndarray):
        real = np.array(real)
    
    result = np.full_like(real, np.nan)
    # Simple implementation
    for i in range(timeperiod, len(real)):
        diff = np.diff(real[i-timeperiod:i+1])
        gain = np.sum(diff[diff > 0])
        loss = np.abs(np.sum(diff[diff < 0]))
        
        if loss == 0:
            result[i] = 100
        else:
            rs = gain / loss
            result[i] = 100 - (100 / (1 + rs))
    
    return result

# Add more stubs for other functions as needed
def BBANDS(real, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands - Minimal implementation"""
    warnings.warn("Using minimal BBANDS implementation")
    if not isinstance(real, np.ndarray):
        real = np.array(real)
    
    middle = MA(real, timeperiod, matype)
    
    std = np.full_like(real, np.nan)
    for i in range(timeperiod-1, len(real)):
        std[i] = np.std(real[i-timeperiod+1:i+1])
    
    upper = middle + nbdevup * std
    lower = middle - nbdevdn * std
    
    return upper, middle, lower

# Empty stubs for compatibility
def MACD(*args, **kwargs):
    warnings.warn("MACD function is a stub in this mock implementation")
    return np.array([]), np.array([]), np.array([])

def STOCH(*args, **kwargs):
    warnings.warn("STOCH function is a stub in this mock implementation")
    return np.array([]), np.array([])

print("Pseudo TA-Lib mock module created")
EOL

# Créer un fichier setup.py
cat > setup.py << 'EOL'
from setuptools import setup, find_packages

setup(
    name="TA-Lib",
    version="0.4.28.mock",
    packages=find_packages(),
    install_requires=["numpy"],
    author="Mock Implementation",
    author_email="mock@example.com",
    description="Mock implementation of TA-Lib for compatibility",
)
EOL

# Installer le package
pip install .

# Tester l'importation
python -c "import talib; print(f'Pseudo TA-Lib mock importé avec succès! Version: {talib.__version__}')"

echo "Installation du pseudo-TA-Lib terminée!"

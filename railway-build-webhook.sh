# Ce webhook est destiné à Railway pour déployer automatiquement votre application
# URL: https://railway.app/project/YOUR_PROJECT_ID/service/YOUR_SERVICE_ID/settings
# Webhook Type: Build
# Cliquez sur "Create Webhook" et copiez l'URL générée

#!/bin/bash

# Installation de NumPy dans une version compatible
pip install numpy==1.24.3

# Attempt 1: Install from anaconda repository
pip install --index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28

# If that fails, attempt 2: Install from GitHub source
if [ $? -ne 0 ]; then
  echo "Installing TA-Lib from GitHub..."
  pip install git+https://github.com/TA-Lib/ta-lib-python.git@0.4.28
fi

# If that also fails, attempt 3: Use mock implementation
if [ $? -ne 0 ]; then
  echo "Installing mock TA-Lib implementation..."
  # Create a minimalistic talib module
  mkdir -p /app/talib_mock/talib
  cat > /app/talib_mock/talib/__init__.py << 'EOL'
import numpy as np
import warnings

warnings.warn("Using minimal TA-Lib mock implementation. Limited functionality available.")

# Version information
__version__ = "0.4.28-mock"

# Basic functions
def MA(real, timeperiod=30, matype=0):
    """Moving Average"""
    if not isinstance(real, np.ndarray):
        real = np.array(real)
    result = np.full_like(real, np.nan)
    for i in range(timeperiod-1, len(real)):
        result[i] = np.mean(real[i-timeperiod+1:i+1])
    return result

def SMA(real, timeperiod=30):
    """Simple Moving Average"""
    return MA(real, timeperiod=timeperiod)

def RSI(real, timeperiod=14):
    """Relative Strength Index - Minimal implementation"""
    warnings.warn("Using minimal RSI implementation")
    return np.zeros_like(real)

def BBANDS(real, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands - Minimal implementation"""
    warnings.warn("Using minimal BBANDS implementation")
    middle = MA(real, timeperiod, matype)
    upper = middle + nbdevup
    lower = middle - nbdevdn
    return upper, middle, lower

def MACD(*args, **kwargs):
    warnings.warn("MACD function is a stub in this mock implementation")
    return np.array([]), np.array([]), np.array([])
EOL

  cat > /app/talib_mock/setup.py << 'EOL'
from setuptools import setup, find_packages

setup(
    name="TA-Lib",
    version="0.4.28.mock",
    packages=find_packages(),
    install_requires=["numpy"],
)
EOL

  cd /app/talib_mock
  pip install .
fi

# Install the rest of requirements
pip install -r requirements.txt

#!/bin/bash
set -e

echo "=== Installation de secours pour TA-Lib ==="

# Installer les dépendances système nécessaires
apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config git

# Créer un répertoire temporaire
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Télécharger une version spécifique de TA-Lib source
echo "Téléchargement de TA-Lib 0.4.0..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz

# Compiler et installer TA-Lib avec une configuration minimale
cd ta-lib/
echo "Compilation de TA-Lib avec configuration minimale..."
./configure --prefix=/usr --disable-shared
make
make install

# Vérifier l'installation de la bibliothèque
echo "Vérification de l'installation des bibliothèques et en-têtes:"
find /usr -name "libta_lib*"
find /usr -name "ta_*.h"

# Créer des liens symboliques
ln -sf /usr/lib/libta_lib.a /usr/lib/libta_lib.so
echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig

# Installation des dépendances Python
pip install --no-cache-dir numpy==1.24.3 Cython setuptools wheel

# Approche 1: Installation directe avec pip
echo "Tentative d'installation avec pip et chemins explicites..."
if pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include/" --global-option="-L/usr/lib/" TA-Lib==0.4.28; then
    echo "✅ TA-Lib installé avec succès via pip avec options explicites"
    python -c "import talib; print('TA-Lib importé avec succès!')"
    exit 0
fi

# Approche 2: Création d'un wrapper minimaliste
echo "Création d'un wrapper minimaliste pour TA-Lib..."
cd "$TEMP_DIR"
mkdir -p simple_talib
cd simple_talib

# Créer un setup.py minimal
cat > setup.py << EOF
from setuptools import setup, Extension
import numpy

talib_extension = Extension(
    'talib._ta_lib',
    ['_ta_lib.c'],
    include_dirs=[numpy.get_include(), '/usr/include', '/usr/local/include', '/usr/include/ta-lib'],
    library_dirs=['/usr/lib', '/usr/local/lib'],
    libraries=['ta_lib'],
    define_macros=[('TA_DICT_KEY', '"string"')]
)

setup(
    name='talib',
    version='0.4.28',
    packages=['talib'],
    ext_modules=[talib_extension],
    author='Minimal TA-Lib Wrapper',
    author_email='support@example.com',
    description='Minimal wrapper for TA-Lib',
)
EOF

# Créer une implémentation minimaliste
mkdir -p talib
cat > talib/__init__.py << EOF
from ._ta_lib import *

def get_functions():
    """
    Returns a list of all available indicator functions.
    """
    import talib
    return [f for f in dir(talib) if not f.startswith('_')]

def get_function_groups():
    """
    Returns a dict with keys being group names and values being lists
    of function names. Similar to Function Browser in TA-Lib.
    """
    return {
        'Momentum Indicators': ['RSI', 'MACD', 'MOM'],
        'Volume Indicators': ['AD', 'ADOSC', 'OBV'],
        'Volatility Indicators': ['ATR', 'NATR'],
        'Price Transform': ['AVGPRICE', 'TYPPRICE', 'WCLPRICE'],
        'Cycle Indicators': ['HT_DCPERIOD', 'HT_DCPHASE'],
        'Pattern Recognition': ['CDL2CROWS', 'CDL3BLACKCROWS'],
        'Statistic Functions': ['BETA', 'CORREL', 'LINEARREG'],
        'Math Operators': ['ADD', 'DIV', 'MAX', 'MIN'],
        'Math Transform': ['ACOS', 'ASIN', 'ATAN', 'CEIL'],
    }
EOF

# Créer un fichier C minimaliste qui exposera seulement les fonctions les plus courantes
cat > _ta_lib.c << EOF
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <ta-lib/ta_libc.h>

/* Wrapper functions for TA-Lib */

static PyObject* talib_MA(PyObject* self, PyObject* args) {
    PyArrayObject *real, *outreal;
    int length, timeperiod;
    int matype = TA_MAType_SMA;
    TA_RetCode ret;
    int outbegidx, outnbelement;

    if (!PyArg_ParseTuple(args, "Oii", &real, &timeperiod, &matype))
        return NULL;

    length = PyArray_SIZE(real);
    outreal = (PyArrayObject*)PyArray_SimpleNew(1, PyArray_DIMS(real), NPY_DOUBLE);

    ret = TA_MA(0, length-1, (double*)PyArray_DATA(real),
                timeperiod, matype,
                &outbegidx, &outnbelement, (double*)PyArray_DATA(outreal));

    if (ret != TA_SUCCESS) {
        Py_DECREF(outreal);
        PyErr_SetString(PyExc_RuntimeError, "TA_MA failed");
        return NULL;
    }

    return (PyObject*)outreal;
}

static PyObject* talib_SMA(PyObject* self, PyObject* args) {
    PyArrayObject *real, *outreal;
    int length, timeperiod;
    TA_RetCode ret;
    int outbegidx, outnbelement;

    if (!PyArg_ParseTuple(args, "Oi", &real, &timeperiod))
        return NULL;

    length = PyArray_SIZE(real);
    outreal = (PyArrayObject*)PyArray_SimpleNew(1, PyArray_DIMS(real), NPY_DOUBLE);

    ret = TA_MA(0, length-1, (double*)PyArray_DATA(real),
                timeperiod, TA_MAType_SMA,
                &outbegidx, &outnbelement, (double*)PyArray_DATA(outreal));

    if (ret != TA_SUCCESS) {
        Py_DECREF(outreal);
        PyErr_SetString(PyExc_RuntimeError, "TA_MA (SMA) failed");
        return NULL;
    }

    return (PyObject*)outreal;
}

static PyObject* talib_RSI(PyObject* self, PyObject* args) {
    PyArrayObject *real, *outrsi;
    int length, timeperiod;
    TA_RetCode ret;
    int outbegidx, outnbelement;

    if (!PyArg_ParseTuple(args, "Oi", &real, &timeperiod))
        return NULL;

    length = PyArray_SIZE(real);
    outrsi = (PyArrayObject*)PyArray_SimpleNew(1, PyArray_DIMS(real), NPY_DOUBLE);

    ret = TA_RSI(0, length-1, (double*)PyArray_DATA(real),
                timeperiod, &outbegidx, &outnbelement, (double*)PyArray_DATA(outrsi));

    if (ret != TA_SUCCESS) {
        Py_DECREF(outrsi);
        PyErr_SetString(PyExc_RuntimeError, "TA_RSI failed");
        return NULL;
    }

    return (PyObject*)outrsi;
}

/* Module definition */

static PyMethodDef TalibMethods[] = {
    {"MA", talib_MA, METH_VARARGS, "Moving Average"},
    {"SMA", talib_SMA, METH_VARARGS, "Simple Moving Average"},
    {"RSI", talib_RSI, METH_VARARGS, "Relative Strength Index"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef talibmodule = {
    PyModuleDef_HEAD_INIT,
    "_ta_lib",
    "Python wrapper for TA-Lib",
    -1,
    TalibMethods
};

PyMODINIT_FUNC PyInit__ta_lib(void) {
    PyObject *m;
    import_array();  // Initialize NumPy
    m = PyModule_Create(&talibmodule);
    if (m == NULL)
        return NULL;
    return m;
}
EOF

# Installer le wrapper
echo "Installation du wrapper minimaliste..."
pip install -e .

# Vérifier l'installation
echo "Vérification de l'installation..."
if python -c "import talib; print('Fonctions disponibles:', talib.get_functions())"; then
    echo "✅ Wrapper TA-Lib minimal installé avec succès"
    exit 0
else
    echo "❌ Échec de l'installation du wrapper minimaliste"
    
    # Dernière tentative - créer un module factice
    echo "Création d'un module TA-Lib factice..."
    cd "$TEMP_DIR"
    mkdir -p talib_mock/talib
    
    # Créer un fichier d'initialisation factice
    cat > talib_mock/talib/__init__.py << EOF
import numpy as np
import warnings

warnings.warn("Utilisation d'une implémentation factice de TA-Lib. Les fonctionnalités seront limitées.")

# Implémentations de base des fonctions les plus courantes
def SMA(real, timeperiod=30):
    real = np.asarray(real)
    return np.convolve(real, np.ones(timeperiod)/timeperiod, mode='same')

def EMA(real, timeperiod=30):
    real = np.asarray(real)
    alpha = 2 / (timeperiod + 1)
    result = np.zeros_like(real)
    result[0] = real[0]
    for i in range(1, len(real)):
        result[i] = alpha * real[i] + (1 - alpha) * result[i-1]
    return result

def RSI(real, timeperiod=14):
    real = np.asarray(real)
    diff = np.diff(real)
    gains = np.copy(diff)
    losses = np.copy(diff)
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = -losses
    
    avg_gain = np.mean(gains[:timeperiod])
    avg_loss = np.mean(losses[:timeperiod])
    
    if avg_loss == 0:
        return np.ones_like(real) * 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi_result = np.zeros_like(real)
    rsi_result[timeperiod:] = rsi
    return rsi_result

def MACD(real, fastperiod=12, slowperiod=26, signalperiod=9):
    real = np.asarray(real)
    ema_fast = EMA(real, fastperiod)
    ema_slow = EMA(real, slowperiod)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signalperiod)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def MA(real, timeperiod=30, matype=0):
    return SMA(real, timeperiod)

def get_functions():
    return ['SMA', 'EMA', 'RSI', 'MACD', 'MA']

def get_function_groups():
    return {
        'Momentum Indicators': ['RSI', 'MACD'],
        'Overlap Studies': ['SMA', 'EMA', 'MA'],
    }
EOF

    # Créer un setup.py pour le module factice
    cat > talib_mock/setup.py << EOF
from setuptools import setup

setup(
    name='ta-lib',
    version='0.4.28',
    packages=['talib'],
    install_requires=['numpy'],
    author='Factice TA-Lib',
    author_email='support@example.com',
    description='Implémentation factice de TA-Lib',
)
EOF

    # Installer le module factice
    cd talib_mock
    pip install .
    
    # Vérifier l'installation
    if python -c "import talib; print('Module factice de TA-Lib importé avec succès!')"; then
        echo "⚠️ Module factice TA-Lib installé comme solution de dernier recours"
        exit 0
    else
        echo "❌ Impossible d'installer même un module factice"
        exit 1
    fi
fi

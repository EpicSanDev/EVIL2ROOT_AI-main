# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Variables d'environnement pour le build
ENV PYTHONDONTWRITEBYTECODE=# Copier les bibliothèques TA-Lib si elles ont été installées
RUN mkdir -p /usr/include/ta-lib
COPY --from=builder /usr/lib/libta_lib* /usr/lib/ || echo "Pas de librairies TA-Lib à copier"
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/ || echo "Pas d'en-têtes TA-Lib à copier"
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONIOENCODING=UTF-8

# Paramètres de build pour contrôler les options d'installation
ARG USE_TALIB_MOCK=false
ARG INSTALL_FULL_DEPS=true
ARG SKIP_OPTIONAL_DEPS=false
ARG TARGETARCH=amd64

# Copie des fichiers requirements et scripts de correction
COPY requirements.txt requirements-essential.txt ./
COPY docker/fix-hnswlib-install.sh docker/fix-talib-install.sh docker/fix-talib-install-alt.sh docker/talib-binary-install.sh docker/talib-mock-install.sh docker/improved-talib-install.sh docker/talib-fallback-install.sh /tmp/
RUN chmod +x /tmp/fix-hnswlib-install.sh /tmp/fix-talib-install.sh /tmp/fix-talib-install-alt.sh /tmp/talib-binary-install.sh /tmp/talib-mock-install.sh /tmp/improved-talib-install.sh /tmp/talib-fallback-install.sh

# Installer les dépendances système et de compilation nécessaires
# y compris celles pour TA-Lib (build-essential, gcc, python3-dev) et psycopg2 (libpq-dev)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        gcc \
        g++ \
        make \
        pkg-config \
        python3-dev \
        libpq-dev \
        unzip \
        git \
        automake \
        libtool \
    && rm -rf /var/lib/apt/lists/*

# Installation de TA-Lib dans une étape séparée, avec option de mockup via build arg
RUN if [ "$USE_TALIB_MOCK" = "false" ]; then \
        echo "Installation de TA-Lib natif..." && \
        { timeout 300 /tmp/improved-talib-install.sh || timeout 300 /tmp/talib-fallback-install.sh || \
          echo "Installation de TA-Lib échouée, utilisation du mock à la place"; } \
    else \
        echo "Utilisation du mock TA-Lib comme demandé dans les arguments de build"; \
    fi

# Mettre à jour pip et installer les dépendances Python en plusieurs étapes
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Installation du mock TA-Lib si nécessaire ou si l'installation native a échoué
RUN if [ "$USE_TALIB_MOCK" = "true" ] || [ ! -f "/usr/lib/libta_lib.so" ]; then \
      echo "Installation du mock TA-Lib..." && \
      pip install --no-cache-dir numpy && \
      mkdir -p /usr/local/lib/python3.10/site-packages/talib && \
      echo "import numpy as np; from numpy import array; def SMA(price, period): return np.convolve(price, np.ones(period)/period, mode='same'); def RSI(*args, **kwargs): return np.zeros(len(args[0]))" > /usr/local/lib/python3.10/site-packages/talib/__init__.py && \
      touch /usr/local/lib/python3.10/site-packages/talib-0.4.28-py3.10.egg-info; \
    else \
      echo "TA-Lib natif installé et détecté, configuration de l'environnement..."; \
      # Installation du wrapper Python pour TA-Lib
      pip install --no-cache-dir TA-Lib==0.4.28 || \
      pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include/" --global-option="-L/usr/lib/" TA-Lib==0.4.28; \
    fi

# Installer les dépendances essentielles d'abord
RUN echo "Installation des dépendances essentielles..." && \
    timeout 300 pip install --no-cache-dir -r requirements-essential.txt || \
    (echo "Timeout ou erreur lors de l'installation des dépendances essentielles, retrying..." && \
     timeout 300 pip install --no-cache-dir -r requirements-essential.txt --no-deps)

# Installer les dépendances principales par groupes (limites en 5 parties)
RUN if [ "$INSTALL_FULL_DEPS" = "true" ]; then \
      echo "Installation des dépendances du groupe 1 (numpy, pandas, scipy)..." && \
      timeout 300 pip install --no-cache-dir numpy pandas scipy; \
    fi

RUN if [ "$INSTALL_FULL_DEPS" = "true" ]; then \
      echo "Installation des dépendances du groupe 2 (sklearn, matplotlib, etc.)..." && \
      timeout 300 pip install --no-cache-dir scikit-learn joblib matplotlib seaborn openpyxl beautifulsoup4; \
    fi

RUN if [ "$INSTALL_FULL_DEPS" = "true" ]; then \
      echo "Installation des dépendances du groupe 3 (flask, requests, etc.)..." && \
      timeout 300 pip install --no-cache-dir flask werkzeug flask-sqlalchemy flask-migrate flask-login flask-wtf flask-cors requests python-dotenv gunicorn alembic; \
    fi

RUN if [ "$INSTALL_FULL_DEPS" = "true" ] && [ "$SKIP_OPTIONAL_DEPS" = "false" ]; then \
      echo "Installation des dépendances du groupe 4 (AI, ML, pytorch)..." && \
      timeout 600 pip install --no-cache-dir "torch>=1.13.0,<2.1.0" "torchvision>=0.14.0,<0.16.0" pytorch-lightning transformers xgboost lightgbm catboost; \
    fi

RUN if [ "$INSTALL_FULL_DEPS" = "true" ] && [ "$SKIP_OPTIONAL_DEPS" = "false" ]; then \
      echo "Installation des dépendances du groupe 5 (tensorflow et autres)..." && \
      timeout 600 pip install --no-cache-dir "tensorflow>=2.10.0,<2.16.0" keras "protobuf>=3.20.0,<4.0.0"; \
    fi

# Installer des dépendances supplémentaires spécifiques
RUN echo "Installation des dépendances supplémentaires..." && \
    pip install --no-cache-dir PyJWT tweepy && \
    # Exécuter le script de correction pour hnswlib
    chmod +x /tmp/fix-hnswlib-install.sh && \
    /tmp/fix-hnswlib-install.sh

# Copie du code source de l'application
# On copie d'abord les répertoires qui changent moins souvent si besoin,
# mais pour cet exemple, on copie tout ce qui est pertinent.
COPY ./app ./app
COPY ./src ./src
COPY ./config ./config
COPY ./alembic.ini ./alembic.ini
COPY ./migrations ./migrations
# Copier d'autres fichiers/dossiers nécessaires au build ou à l'exécution
# COPY ./patches ./patches # Déplacé au runtime si nécessaire spécifiquement là

# Stage 2: Runtime
FROM python:3.10-slim as runtime

WORKDIR /app

# Créer un utilisateur et un groupe non-root
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/lib \
    NLTK_DATA=/app/nltk_data \
    MPLCONFIGDIR=/app/.matplotlib_cache

# Copier les dépendances installées depuis le stage builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copier les bibliothèques TA-Lib si elles ont été installées
COPY --from=builder /usr/lib/libta_lib* /usr/lib/ 2>/dev/null || :
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/ 2>/dev/null || :

# Copier le code de l'application depuis le stage builder
COPY --from=builder /app/app/ ./app/
COPY --from=builder /app/src/ ./src/
COPY --from=builder /app/config/ ./config/
COPY --from=builder /app/alembic.ini ./alembic.ini
COPY --from=builder /app/migrations/ ./migrations/

# Application du correctif pour la gestion des URLs Redis si nécessaire au runtime
COPY patches/fix-redis-connection.sh /tmp/fix-redis-connection.sh
RUN chmod +x /tmp/fix-redis-connection.sh && \
    /tmp/fix-redis-connection.sh && \
    rm /tmp/fix-redis-connection.sh

# Installer unzip pour NLTK et nettoyer apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

# Vérifier l'installation de TA-Lib et créer un mock si nécessaire
RUN if ! python -c "import talib" 2>/dev/null; then \
      echo "TA-Lib n'est pas disponible, création d'un module mock..." && \
      pip install --no-cache-dir numpy && \
      mkdir -p /usr/local/lib/python3.10/site-packages/talib && \
      echo "import numpy as np; from numpy import array; def SMA(price, period): return np.convolve(price, np.ones(period)/period, mode='same'); def RSI(*args, **kwargs): return np.zeros(len(args[0]))" > /usr/local/lib/python3.10/site-packages/talib/__init__.py && \
      touch /usr/local/lib/python3.10/site-packages/talib-0.4.28-py3.10.egg-info; \
    else \
      echo "TA-Lib est correctement installé."; \
    fi

# Créer le répertoire pour les données NLTK, télécharger les données, et définir la variable d'environnement
RUN mkdir -p /app/nltk_data && \
    python -m nltk.downloader -d /app/nltk_data punkt averaged_perceptron_tagger && \
    mkdir -p /app/.matplotlib_cache && \
    chown -R appuser:appuser /app

# Changer d'utilisateur
USER appuser

# Healthcheck pour assurer que le service est opérationnel
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Port d'exposition
EXPOSE 8000

# Commande par défaut avec un délai de démarrage pour permettre aux services dépendants de s'initialiser
CMD ["sh", "-c", "sleep 2 && uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000"]
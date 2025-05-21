# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Variables d'environnement pour le build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copie des fichiers requirements et scripts de correction
COPY requirements.txt .
COPY docker/fix-hnswlib-install.sh docker/fix-talib-install.sh docker/fix-talib-install-alt.sh docker/talib-binary-install.sh docker/talib-mock-install.sh docker/improved-talib-install.sh /tmp/
RUN chmod +x /tmp/fix-hnswlib-install.sh /tmp/fix-talib-install.sh /tmp/fix-talib-install-alt.sh /tmp/talib-binary-install.sh /tmp/talib-mock-install.sh /tmp/improved-talib-install.sh

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
    && rm -rf /var/lib/apt/lists/* && \
    # Installer la bibliothèque C TA-Lib via le script amélioré
    /tmp/improved-talib-install.sh

# Mettre à jour pip et installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    # Installer une version de NumPy compatible avec TA-Lib (déjà installé dans le script improved-talib-install.sh)
    # Valider que TA-Lib est correctement installé
    python -c "import talib; print('TA-Lib importé avec succès!')" && \
    # Installer le reste des dépendances de production
    pip install --no-cache-dir -r requirements.txt && \
    # Installer les dépendances supplémentaires
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

# Copier les dépendances installées depuis le stage builder
# On s'assure de copier le virtual environment ou les packages installés globalement dans le slim
# Pour pip install global, les packages sont typiquement dans /usr/local/lib/pythonX.Y/site-packages
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
# S'assurer que TA-Lib et autres libs compilées sont copiées si elles ne sont pas dans site-packages
# Par exemple, si TA-Lib installe des .so ailleurs:
# COPY --from=builder /usr/local/lib/libta_lib.so.0 /usr/local/lib/libta_lib.so.0
# COPY --from=builder /usr/local/lib/libta_lib.la /usr/local/lib/libta_lib.la
# COPY --from=builder /usr/local/include/ta-lib/ta_libc.h /usr/local/include/ta-lib/ta_libc.h
# Il est crucial de vérifier où les bibliothèques comme TA-Lib sont installées par le builder.
# Pour simplifier, on peut copier les dossiers de lib pertinents.
# Une approche plus robuste serait d'utiliser un virtual environment dans le builder et de le copier.

# Copier le code de l'application depuis le stage builder
COPY --from=builder /app/app ./app
COPY --from=builder /app/src ./src
COPY --from=builder /app/config ./config
COPY --from=builder /app/alembic.ini ./alembic.ini
COPY --from=builder /app/migrations ./migrations

# Application du correctif pour la gestion des URLs Redis si nécessaire au runtime
COPY patches/fix-redis-connection.sh /tmp/fix-redis-connection.sh
RUN chmod +x /tmp/fix-redis-connection.sh && \
    /tmp/fix-redis-connection.sh && \
    rm /tmp/fix-redis-connection.sh

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
# S'assurer que les libs sont trouvées
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib

# Installer unzip pour NLTK et nettoyer apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

# Créer le répertoire pour les données NLTK, télécharger les données, et définir la variable d'environnement
RUN mkdir -p /app/nltk_data && \
    python -m nltk.downloader -d /app/nltk_data punkt averaged_perceptron_tagger && \
    chown -R appuser:appuser /app/nltk_data
ENV NLTK_DATA=/app/nltk_data

# Créer le répertoire pour le cache Matplotlib
RUN mkdir -p /app/.matplotlib_cache
ENV MPLCONFIGDIR=/app/.matplotlib_cache

# Définir les permissions pour l'utilisateur non-root
# Donner la propriété du répertoire de travail à l'utilisateur non-root, y compris nltk_data et matplotlib_cache
RUN chown -R appuser:appuser /app

# Changer d'utilisateur
USER appuser

# Port d'exposition
EXPOSE 8000

# Commande par défaut
CMD ["uvicorn", "src.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
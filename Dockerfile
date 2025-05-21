# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Variables d'environnement pour le build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copie des fichiers requirements et scripts de correction
COPY requirements.txt .
COPY docker/fix-hnswlib-install.sh /tmp/fix-hnswlib-install.sh
COPY docker/fix-talib-install.sh /tmp/fix-talib-install.sh
RUN chmod +x /tmp/fix-hnswlib-install.sh && \
    chmod +x /tmp/fix-talib-install.sh

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
        libpq-dev \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip et installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    # Étape 1: Installer la bibliothèque C TA-Lib en utilisant le script
    # Le script fix-talib-install.sh doit être copié dans /tmp/ au préalable
    /tmp/fix-talib-install.sh && \
    # Étape 2: Installer le wrapper Python TA-Lib. Il devrait maintenant trouver la lib C.
    pip install --no-cache-dir TA-Lib>=0.4.28 && \
    # Maintenant, vérifier que TA-Lib est correctement installé
    python -c "import talib; print('TA-Lib importé avec succès!')" && \
    # Étape 3: Installer le reste des dépendances de production
    pip install --no-cache-dir -r requirements.txt && \
    # Installer les dépendances supplémentaires qui pourraient être nécessaires pour le build ou runtime
    pip install --no-cache-dir PyJWT tweepy && \
    # Exécuter le script de correction pour hnswlib après l'installation des requirements
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
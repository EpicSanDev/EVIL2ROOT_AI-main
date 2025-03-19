FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    g++ \
    make \
    cmake \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    git \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers requirements et scripts
COPY requirements.txt .
COPY docker/fix-hnswlib-install.sh /tmp/fix-hnswlib-install.sh
COPY docker/fix-talib-install.sh /tmp/fix-talib-install.sh
RUN chmod +x /tmp/fix-hnswlib-install.sh && \
    chmod +x /tmp/fix-talib-install.sh

# Installation des dépendances Python et TA-Lib
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    /tmp/fix-talib-install.sh && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir PyJWT tweepy && \
    /tmp/fix-hnswlib-install.sh

# Copie du code source
COPY . .

# Application du correctif pour la gestion des URLs Redis
COPY patches/fix-redis-connection.sh /tmp/fix-redis-connection.sh
RUN chmod +x /tmp/fix-redis-connection.sh && \
    /tmp/fix-redis-connection.sh

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Port d'exposition
EXPOSE 8000

# Commande par défaut
CMD ["uvicorn", "src.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
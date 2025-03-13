FROM python:3.10-slim AS builder

LABEL maintainer="EVIL2ROOT Team <contact@evil2root.ai>"
LABEL description="EVIL2ROOT Trading Bot - Système de trading automatisé avec validation IA"
LABEL version="1.0.0"

# Définir les variables d'environnement pour le builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    TZ=Europe/Paris \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH"

# Installer les dépendances système et les outils nécessaires pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    gcc \
    g++ \
    make \
    tzdata \
    ca-certificates \
    && python -m venv /opt/venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances - séparées pour un meilleur caching
COPY requirements-essential.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements-essential.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier setup.py pour l'installation en mode développement
COPY setup.py .
RUN pip install -e .

# Phase de runtime avec une image plus légère
FROM python:3.10-slim AS runtime

# Métadonnées
LABEL maintainer="EVIL2ROOT Team <contact@evil2root.ai>"
LABEL description="EVIL2ROOT Trading Bot - Système de trading automatisé avec validation IA"
LABEL version="1.0.0"

# Définir les variables d'environnement pour le runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    TZ=Europe/Paris \
    PATH="/opt/venv/bin:$PATH"

# Installer seulement les dépendances runtime essentielles
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier l'environnement virtuel de l'étape de build
COPY --from=builder /opt/venv /opt/venv

# Créer les répertoires nécessaires avec les bonnes permissions
RUN mkdir -p /app/logs /app/data /app/data/models /app/data/historical /app/config && \
    chown -R nobody:nogroup /app

# Copier le code source
COPY src/ /app/src/
COPY scripts/shell/docker-entrypoint.sh /app/
COPY config/ /app/config/
COPY setup.py .

# Rendre le script d'entrée exécutable
RUN chmod +x /app/docker-entrypoint.sh

# Exposer les ports
EXPOSE 5000 8888

# Utiliser un utilisateur non-root pour la sécurité
USER nobody:nogroup

# Point d'entrée
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Commande par défaut
CMD ["python", "src/main.py", "--mode", "paper"]
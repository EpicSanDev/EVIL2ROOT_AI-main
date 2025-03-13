FROM python:3.10-slim

LABEL maintainer="EVIL2ROOT Team <contact@evil2root.ai>"
LABEL description="EVIL2ROOT Trading Bot - Système de trading automatisé avec validation IA"
LABEL version="1.0.0"

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    TZ=Europe/Paris

# Installer les dépendances système et les outils
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .
COPY requirements-essential.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs /app/data /app/data/models /app/data/historical /app/config

# Copier le code source
COPY src/ /app/src/
COPY scripts/shell/docker-entrypoint.sh /app/
COPY config/ /app/config/
COPY setup.py .

# Rendre le script d'entrée exécutable
RUN chmod +x /app/docker-entrypoint.sh

# Installer le package en mode développement
RUN pip install -e .

# Exposer les ports
EXPOSE 5000 8888

# Point d'entrée
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Commande par défaut
CMD ["python", "src/main.py", "--mode", "paper"]
FROM python:3.10-slim AS builder

# Variables d'environnement pour le build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && python -m venv /opt/venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances - séparés pour un meilleur caching
COPY requirements-essential.txt ./
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements-essential.txt

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-telegram-bot[webhooks]==20.6 python-dotenv psycopg2-binary qrcode[pil]

# Phase de runtime avec une image plus légère
FROM python:3.10-slim AS runtime

# Variables d'environnement pour le runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /app

# Installer uniquement les dépendances runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier l'environnement virtuel de l'étape de build
COPY --from=builder /opt/venv /opt/venv

# Créer l'utilisateur non-root et les répertoires nécessaires
RUN groupadd -r botuser && useradd -r -g botuser botuser \
    && mkdir -p /app/logs /app/data \
    && chown -R botuser:botuser /app

# Copier le code source
COPY . .

# Changer vers l'utilisateur non-root
USER botuser

# Exposer le port (pour le webhook si configuré plus tard)
EXPOSE 8443

# Commande de démarrage
CMD ["python", "app/scripts/run_telegram_bot.py"] 
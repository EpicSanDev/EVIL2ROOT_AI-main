FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt requirements-essential.txt ./

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir python-telegram-bot[webhooks]==20.6 python-dotenv psycopg2-binary qrcode[pil]

# Copier le code source
COPY . .

# Créer l'utilisateur non-root
RUN groupadd -r botuser && useradd -r -g botuser botuser \
    && mkdir -p /app/logs /app/data \
    && chown -R botuser:botuser /app

# Changer vers l'utilisateur non-root
USER botuser

# Environnement
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Exposer le port (pour le webhook si configuré plus tard)
EXPOSE 8443

# Commande de démarrage
CMD ["python", "app/scripts/run_telegram_bot.py"] 
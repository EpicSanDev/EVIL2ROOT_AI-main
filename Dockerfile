FROM python:3.9-slim

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=UTC

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    gcc \
    g++ \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Créer un répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances d'abord (pour tirer parti du cache de Docker)
COPY requirements.txt requirements-essential.txt ./

# Installer les dépendances essentielles en premier
RUN pip install --no-cache-dir -r requirements-essential.txt

# Installer TensorFlow séparément pour une meilleure compatibilité
RUN pip install --no-cache-dir tensorflow==2.11.0

# Installer PyTorch (sans CUDA pour réduire la taille)
RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Installer les dépendances Hugging Face
RUN pip install --no-cache-dir transformers==4.28.1

# Installer les dépendances restantes
RUN pip install --no-cache-dir -r requirements.txt

# Installer les nouvelles dépendances pour l'apprentissage par renforcement et l'analyse de sentiment
RUN pip install --no-cache-dir \
    stable-baselines3==1.7.0 \
    ta==0.10.2 \
    gymnasium==0.28.1 \
    websocket-client==1.5.1 \
    tweepy==4.12.1 \
    vaderSentiment==3.3.2 \
    talib-binary==0.4.19 \
    beautifulsoup4==4.12.2 \
    seaborn==0.12.2 \
    jinja2==3.1.2 \
    scikit-learn==1.2.2

# Copier le code source
COPY . .

# S'assurer que les scripts sont exécutables
RUN chmod +x docker-entrypoint.sh init-secrets.sh

# Créer les répertoires nécessaires
RUN mkdir -p logs data saved_models results config

# Ajouter un utilisateur non-root
RUN adduser --disabled-password --gecos "" trader
RUN chown -R trader:trader /app
USER trader

# Exposer le port pour l'API (si applicable)
EXPOSE 8000

# Point d'entrée et commande par défaut
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "src/main_trading_bot.py"]
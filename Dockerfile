FROM python:3.9-slim

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=UTC \
    PIP_DEFAULT_TIMEOUT=100

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
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installation de TA-Lib depuis les sources
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    rm -rf /tmp/ta-lib-0.4.0-src.tar.gz /tmp/ta-lib

# Créer un répertoire de travail
WORKDIR /app

# Mettre à jour pip
RUN pip install --no-cache-dir --upgrade pip

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

# Installer les dépendances restantes par groupes pour mieux gérer les conflits
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn joblib matplotlib seaborn
RUN pip install --no-cache-dir openpyxl beautifulsoup4 threadpoolctl Pillow requests
RUN pip install --no-cache-dir psycopg2-binary redis psutil
RUN pip install --no-cache-dir flask==2.2.5 werkzeug==2.2.3 flask-sqlalchemy flask-migrate flask-login flask-wtf flask-cors
RUN pip install --no-cache-dir python-dotenv gunicorn alembic yfinance
RUN pip install --no-cache-dir pylint pytest pytest-cov python-telegram-bot asyncio schedule protobuf==3.20.3
RUN pip install --no-cache-dir xgboost lightgbm catboost pydantic dill h5py sqlalchemy
RUN pip install --no-cache-dir shap lime interpret interpret-core alibi dalex bayesian-optimization
RUN pip install --no-cache-dir mlflow wandb mlxtend eli5 graphviz dtreeviz yellowbrick plotly_express
RUN pip install --no-cache-dir ta python-binance ccxt statsmodels alpha_vantage backtrader
RUN pip install --no-cache-dir plotly dash prometheus_client
RUN pip install --no-cache-dir sentence-transformers hnswlib optimum langchain
RUN pip install --no-cache-dir optuna hyperopt vaderSentiment nltk scikit-optimize gpytorch tune-sklearn fastai

# Installer les nouvelles dépendances pour l'apprentissage par renforcement et l'analyse de sentiment
# Nous installons ces dépendances avec des versions spécifiques car elles sont critiques
RUN pip install --no-cache-dir --upgrade setuptools wheel
RUN pip install --no-cache-dir gym==0.21.0
RUN pip install --no-cache-dir \
    stable-baselines3==1.7.0 \
    gymnasium==0.28.1 \
    websocket-client==1.5.1 \
    tweepy==4.12.1 \
    vaderSentiment==3.3.2 \
    jinja2==3.1.2
# Installation séparée de ta-lib car déjà compilé dans une étape précédente
RUN pip install --no-cache-dir ta-lib==0.4.19

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
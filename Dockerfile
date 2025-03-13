FROM python:3.9-slim AS builder

# Arguments de build pour les métadonnées
ARG BUILD_DATE
ARG GIT_COMMIT
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# Create working directory
WORKDIR /app

# Install system dependencies, create virtual environment, and cleanup in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && python -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

# Install essential dependencies first (for better caching)
COPY requirements-essential.txt .
RUN pip install --no-cache-dir -r requirements-essential.txt

# Install Plotly and Dash separately to avoid heavy dependencies
RUN pip install --no-cache-dir plotly==5.14.1 --no-deps \
    && pip install --no-cache-dir dash==2.10.0 --no-deps

# Install protobuf explicitly before tensorflow to avoid conflicts
RUN pip install --no-cache-dir "protobuf>=3.20.3,<5.0.0dev"

# Install remaining dependencies excluding Plotly and Dash
COPY requirements.txt .
RUN grep -v "plotly\|dash" requirements.txt > requirements-filtered.txt \
    && cat requirements-filtered.txt | grep -v "^#" | grep "^numpy\|^pandas\|^scipy\|^scikit-learn\|^joblib\|^matplotlib\|^seaborn" > core-deps.txt \
    && pip install --no-cache-dir -r core-deps.txt \
    && cat requirements-filtered.txt | grep -v "^#" | grep "^requests\|^psycopg2\|^redis\|^psutil\|^flask\|^python-dotenv\|^gunicorn" > web-deps.txt \
    && pip install --no-cache-dir -r web-deps.txt

# Installation des dépendances ML en étapes séparées pour économiser la mémoire
RUN cat requirements-filtered.txt | grep -v "^#" | grep "^torch" > torch-deps.txt \
    && pip install --no-cache-dir -r torch-deps.txt --no-deps \
    && rm -rf /root/.cache/pip \
    && echo "Torch installé, nettoyage des caches pour libérer de la mémoire"

RUN cat requirements-filtered.txt | grep -v "^#" | grep "^transformers\|^xgboost\|^lightgbm" > ml-libs.txt \
    && pip install --no-cache-dir -r ml-libs.txt \
    && rm -rf /root/.cache/pip \
    && echo "Transformers, XGBoost et LightGBM installés"

RUN cat requirements-filtered.txt | grep -v "^#" | grep "^catboost" > catboost.txt \
    && pip install --no-cache-dir -r catboost.txt \
    && rm -rf /root/.cache/pip \
    && echo "CatBoost installé"

RUN cat requirements-filtered.txt | grep -v "^#" | grep "^tensorflow" > tf-core.txt \
    && pip install --no-cache-dir -r tf-core.txt \
    && rm -rf /root/.cache/pip \
    && echo "TensorFlow installé"

RUN cat requirements-filtered.txt | grep -v "^#" | grep "^keras" > keras.txt \
    && pip install --no-cache-dir -r keras.txt \
    && rm -rf /root/.cache/pip \
    && echo "Keras installé"

RUN cat requirements-filtered.txt | grep -v "^#" | grep "^torchvision" > torch-vision.txt \
    && pip install --no-cache-dir -r torch-vision.txt --no-deps \
    && rm -rf /root/.cache/pip \
    && echo "TorchVision installé"

RUN cat requirements-filtered.txt | grep -v "^#" | grep "^pytorch-lightning" > pytorch-lightning.txt \
    && pip install --no-cache-dir -r pytorch-lightning.txt \
    && rm -rf /root/.cache/pip \
    && echo "PyTorch Lightning installé"

RUN cat requirements-filtered.txt | grep -v "^#" | grep "^stable-baselines3\|^fastai\|^optimum" > other-torch.txt \
    && pip install --no-cache-dir -r other-torch.txt --use-pep517 \
    && rm -rf /root/.cache/pip \
    && echo "Autres dépendances PyTorch installées"

# Installation des autres dépendances (non ML) en une seule étape pour optimiser le build
RUN cat requirements-filtered.txt | grep -v "^#" | grep -v "^numpy\|^pandas\|^scipy\|^scikit-learn\|^joblib\|^matplotlib\|^seaborn\|^requests\|^psycopg2\|^redis\|^psutil\|^flask\|^python-dotenv\|^gunicorn\|^tensorflow\|^keras\|^torch\|^transformers\|^xgboost\|^lightgbm\|^catboost\|^causalml\|^econml\|^torchvision\|^pytorch-lightning\|^stable-baselines3\|^fastai\|^optimum" > other-deps.txt \
    && pip install --no-cache-dir -r other-deps.txt

# Installation explicite de prometheus_client
RUN pip install --no-cache-dir prometheus_client>=0.16.0

# Copy and run the causalml fix script
COPY docker/fix-causalml-install.sh /tmp/fix-causalml-install.sh
RUN chmod +x /tmp/fix-causalml-install.sh && /tmp/fix-causalml-install.sh || echo "⚠️ Installation de causalml échouée - un package factice a été créé pour éviter les erreurs"

# Copier le script d'installation de finbert-embedding (optionnel)
COPY docker/fix-finbert-install.sh /tmp/fix-finbert-install.sh
RUN chmod +x /tmp/fix-finbert-install.sh

# Multi-stage build for smaller final image
FROM python:3.9-slim AS runtime

# Métadonnées pour la traçabilité et la gestion des images
LABEL org.opencontainers.image.title="Evil2Root AI" \
      org.opencontainers.image.vendor="Evil2Root" \
      maintainer="Evil2Root Team"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    FLASK_APP=run.py \
    FLASK_ENV=production \
    PATH="/opt/venv/bin:$PATH" \
    # Configuration spécifique pour DigitalOcean App Platform
    PORT=8080 \
    # Plus de journalisation pour faciliter le débogage
    GUNICORN_CMD_ARGS="--access-logfile=- --error-logfile=- --log-level=info"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code first, before creating directories and running scripts
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data logs saved_models && \
    chmod -R 777 data logs saved_models

# Make scripts executable and prepare environment
RUN chmod +x docker/prepare-scripts.sh && \
    ./docker/prepare-scripts.sh && \
    chmod +x docker-entrypoint.sh

# Health check adapté pour DigitalOcean (utilisera la variable PORT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port pour DigitalOcean - utilisera la variable PORT injectée
EXPOSE ${PORT}

# Script de démarrage modifié pour DigitalOcean
RUN echo '#!/bin/sh\n\
# Adapter les chemins et permissions pour DigitalOcean\n\
mkdir -p /app/data /app/logs /app/saved_models\n\
chmod -R 777 /app/data /app/logs /app/saved_models\n\
\n\
# Exécuter le script d'\''entrée original\n\
exec /app/docker-entrypoint.sh "$@"\n\
' > /app/digitalocean-entrypoint.sh && \
    chmod +x /app/digitalocean-entrypoint.sh

# DigitalOcean App Platform préfère exécuter en tant que root
# USER nobody:nogroup

# Set entrypoint for DigitalOcean
ENTRYPOINT ["/app/digitalocean-entrypoint.sh"]

# Default command - utilisera la variable PORT injectée par DigitalOcean
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} run:app"]
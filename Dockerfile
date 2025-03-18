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

# Installer les dépendances en une seule couche pour optimiser la taille de l'image
COPY requirements-essential.txt requirements.txt ./

# Installation des dépendances principales
RUN pip install --no-cache-dir -r requirements-essential.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir prometheus_client>=0.16.0 && \
    rm -rf /root/.cache/pip

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

# Create working directory and application user for security
RUN useradd -m -s /bin/bash -u 1000 appuser && \
    mkdir -p /app/data /app/logs /app/saved_models && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser . .

# Set permissions
RUN chmod -R 755 /app && \
    chmod -R 777 /app/data /app/logs /app/saved_models && \
    chmod +x docker-entrypoint.sh docker/prepare-scripts.sh && \
    ./docker/prepare-scripts.sh

# Change to non-root user
USER appuser

# Health check adapté pour DigitalOcean (utilisera la variable PORT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port pour DigitalOcean - utilisera la variable PORT injectée
EXPOSE ${PORT}

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command - utilisera la variable PORT injectée par DigitalOcean
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} run:app"]
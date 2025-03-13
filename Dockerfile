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

# Install remaining dependencies excluding Plotly and Dash
COPY requirements.txt .
RUN grep -v "plotly\|dash" requirements.txt > requirements-filtered.txt \
    && cat requirements-filtered.txt | grep -v "^#" | grep "^numpy\|^pandas\|^scipy\|^scikit-learn\|^joblib\|^matplotlib\|^seaborn" > core-deps.txt \
    && pip install --no-cache-dir -r core-deps.txt \
    && cat requirements-filtered.txt | grep -v "^#" | grep "^requests\|^psycopg2\|^redis\|^psutil\|^flask\|^python-dotenv\|^gunicorn" > web-deps.txt \
    && pip install --no-cache-dir -r web-deps.txt \
    && cat requirements-filtered.txt | grep -v "^#" | grep "^tensorflow\|^keras\|^torch\|^transformers\|^xgboost\|^lightgbm\|^catboost" > ml-deps.txt \
    && pip install --no-cache-dir -r ml-deps.txt \
    && cat requirements-filtered.txt | grep -v "^#" | grep -v "^numpy\|^pandas\|^scipy\|^scikit-learn\|^joblib\|^matplotlib\|^seaborn\|^requests\|^psycopg2\|^redis\|^psutil\|^flask\|^python-dotenv\|^gunicorn\|^tensorflow\|^keras\|^torch\|^transformers\|^xgboost\|^lightgbm\|^catboost\|^causalml\|^econml" > other-deps.txt \
    && pip install --no-cache-dir -r other-deps.txt \
    && pip install --no-cache-dir prometheus_client>=0.16.0

# Copy and run the causalml fix script
COPY docker/fix-causalml-install.sh /tmp/fix-causalml-install.sh
RUN chmod +x /tmp/fix-causalml-install.sh && /tmp/fix-causalml-install.sh || echo "Installation de causalml échouée - l'application fonctionnera sans ce package"

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
    PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create necessary directories with proper permissions
RUN mkdir -p data logs saved_models && \
    chown -R nobody:nogroup data logs saved_models

# Copy application code
COPY . .

# Make scripts executable and prepare environment
RUN chmod +x docker/prepare-scripts.sh && \
    ./docker/prepare-scripts.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Use non-root user for security
USER nobody:nogroup

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
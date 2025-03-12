FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install essential dependencies
COPY requirements-essential.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements-essential.txt

# Install Plotly and Dash separately to avoid heavy dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir plotly==5.14.1 --no-deps \
    && pip install --no-cache-dir dash==2.10.0 --no-deps

# Install remaining dependencies excluding Plotly and Dash
COPY requirements.txt .
RUN grep -v "plotly\|dash" requirements.txt > requirements-filtered.txt \
    && --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements-filtered.txt

# Ensure prometheus-client is installed
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir prometheus_client>=0.16.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs saved_models

# Set up environment
ENV FLASK_APP=run.py \
    FLASK_ENV=production

# Make scripts executable
RUN chmod +x start_daily_analysis.py \
    docker-entrypoint.sh \
    start_market_scheduler.sh \
    stop_market_scheduler.sh \
    analysis-bot-entrypoint.sh

# Expose port
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
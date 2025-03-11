FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances en plusieurs étapes pour optimiser le cache Docker
# 1. D'abord les dépendances essentielles (qui changent rarement)
COPY requirements-essential.txt .
RUN pip install --no-cache-dir -r requirements-essential.txt

# 2. Installer Plotly et Dash séparément avec l'option --no-deps pour éviter les dépendances lourdes
RUN pip install --no-cache-dir plotly==5.14.1 --no-deps \
    && pip install --no-cache-dir dash==2.10.0 --no-deps

# 3. Installer le reste des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for TensorFlow GPU support (if needed)
# Uncomment if you need GPU support
# RUN pip install tensorflow-gpu

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data logs saved_models

# Set up environment
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"] 
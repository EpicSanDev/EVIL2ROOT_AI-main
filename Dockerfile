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
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 1. D'abord les dépendances essentielles (qui changent rarement)
COPY requirements-essential.txt .
RUN pip install --no-cache-dir -r requirements-essential.txt

# 2. Installer Plotly et Dash séparément avec l'option --no-deps pour éviter les dépendances lourdes
# Notez que l'option --no-deps est appliquée ici directement dans la commande pip, pas dans requirements.txt
RUN pip install --no-cache-dir plotly==5.14.1 --no-deps \
    && pip install --no-cache-dir dash==2.10.0 --no-deps

# 3. Installer le reste des dépendances
# Nous excluons Plotly et Dash qui ont déjà été installés séparément
COPY requirements.txt .
RUN grep -v "plotly\|dash" requirements.txt > requirements-filtered.txt \
    && pip install --no-cache-dir -r requirements-filtered.txt

# Install additional dependencies for analysis bot
RUN pip install --no-cache-dir python-telegram-bot==20.6 asyncio==3.4.3 schedule==1.2.0 openai

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data logs saved_models

# Set up environment
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Make start script executable
RUN chmod +x start_daily_analysis.py

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"] 
#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/logs /app/saved_models

# Function to check if Redis is ready
wait_for_redis() {
  echo "Waiting for Redis to be ready..."
  while ! redis-cli -h "$REDIS_HOST" ping > /dev/null 2>&1; do
    echo "Redis is unavailable - sleeping"
    sleep 2
  done
  echo "Redis is up - continuing"
}

# Function to check if Ollama is ready
wait_for_ollama() {
  echo "Waiting for Ollama to be ready..."
  while ! curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/status" > /dev/null 2>&1; do
    echo "Ollama is unavailable - sleeping"
    sleep 2
  done
  echo "Ollama is up - continuing"
}

# Wait for Redis
if [ ! -z "$REDIS_HOST" ]; then
  apt-get update && apt-get install -y --no-install-recommends redis-tools
  wait_for_redis
fi

# Wait for Ollama container to be ready
if [ ! -z "$OLLAMA_HOST" ]; then
  wait_for_ollama
fi

# Execute the main command
exec "$@"

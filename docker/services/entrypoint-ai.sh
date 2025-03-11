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

# Wait for Redis
if [ ! -z "$REDIS_HOST" ]; then
  apt-get update && apt-get install -y --no-install-recommends redis-tools
  wait_for_redis
fi

# Execute the main command
exec "$@"

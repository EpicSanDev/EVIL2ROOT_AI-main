#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/logs /app/saved_models

# Function to check if Redis is ready
wait_for_redis() {
  echo "Waiting for Redis to be ready..."
  
  local max_retries=20
  local retry_count=0
  local retry_delay=1
  
  while ! redis-cli -h "$REDIS_HOST" ping > /dev/null 2>&1; do
    retry_count=$((retry_count+1))
    if [ $retry_count -ge $max_retries ]; then
      echo "ERROR: Redis is still unavailable after $max_retries attempts. Exiting."
      exit 1
    fi
    
    local wait_time=$((retry_delay * (2 ** (retry_count > 4 ? 4 : retry_count - 1))))
    echo "Redis is unavailable - sleeping for ${wait_time}s (attempt $retry_count/$max_retries)"
    sleep $wait_time
  done
  
  # Vérification supplémentaire que Redis est opérationnel
  echo "Redis service is up, verifying functionality..."
  local test_key="healthcheck_$(date +%s)"
  local test_value="ok"
  
  if ! redis-cli -h "$REDIS_HOST" SET $test_key $test_value EX 10 > /dev/null 2>&1 || \
     [ "$(redis-cli -h "$REDIS_HOST" GET $test_key)" != "$test_value" ]; then
    echo "ERROR: Redis service is up but functionality check failed. Exiting."
    exit 1
  fi
  
  echo "Redis is up and operational - continuing"
}

# Wait for Redis
if [ ! -z "$REDIS_HOST" ]; then
  apt-get update && apt-get install -y --no-install-recommends redis-tools
  wait_for_redis
fi

# Setup AI environment
echo "Setting up AI service environment..."

# Check if required models are available
if [ -d "/app/saved_models" ]; then
  model_count=$(find /app/saved_models -type f -name "*.h5" -o -name "*.pkl" | wc -l)
  echo "Found $model_count model(s) in saved_models directory"
else
  echo "WARNING: No saved models directory found, models will be trained if needed"
fi

# Execute the main command
echo "Starting AI service with command: $@"
exec "$@"

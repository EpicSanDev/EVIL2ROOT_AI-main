#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/logs /app/data /app/saved_models

# Function to check if Postgres is ready
wait_for_postgres() {
  echo "Waiting for PostgreSQL to be ready..."
  
  local max_retries=30
  local retry_count=0
  local retry_delay=2
  
  while ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" > /dev/null 2>&1; do
    retry_count=$((retry_count+1))
    if [ $retry_count -ge $max_retries ]; then
      echo "ERROR: PostgreSQL is still unavailable after $max_retries attempts. Exiting."
      exit 1
    fi
    
    local wait_time=$((retry_delay * (2 ** (retry_count > 4 ? 4 : retry_count - 1))))
    echo "PostgreSQL is unavailable - sleeping for ${wait_time}s (attempt $retry_count/$max_retries)"
    sleep $wait_time
  done
  
  # Vérification supplémentaire que la base est accessible
  echo "PostgreSQL service is up, verifying database connection..."
  local db_check_timeout=30
  local db_check_start=$(date +%s)
  
  while ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; do
    local now=$(date +%s)
    if [ $((now - db_check_start)) -gt $db_check_timeout ]; then
      echo "ERROR: Database is not accessible after $db_check_timeout seconds. Exiting."
      exit 1
    fi
    echo "Database not yet accessible - retrying..."
    sleep 2
  done
  
  echo "PostgreSQL is up and database is accessible - continuing"
}

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

# Wait for dependencies
if [ ! -z "$DB_HOST" ]; then
  apt-get update && apt-get install -y --no-install-recommends postgresql-client redis-tools
  wait_for_postgres
fi

if [ ! -z "$REDIS_HOST" ]; then
  [ -z "$(which redis-cli)" ] && apt-get update && apt-get install -y --no-install-recommends redis-tools
  wait_for_redis
fi

# Setup application
echo "Setting up the bot service..."

# Check environment variables required for the bot
if [ -z "$API_KEY_FILE" ] && [ -z "$API_KEY" ]; then
  echo "WARNING: No API key configuration found. Some features may be limited."
fi

# Execute the main command
echo "Starting bot service with command: $@"
exec "$@"

#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p /app/logs /app/data /app/saved_models

# Function to check if Postgres is ready
wait_for_postgres() {
  echo "Waiting for PostgreSQL to be ready..."
  while ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" > /dev/null 2>&1; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 2
  done
  echo "PostgreSQL is up - continuing"
}

# Function to check if Redis is ready
wait_for_redis() {
  echo "Waiting for Redis to be ready..."
  while ! redis-cli -h "$REDIS_HOST" ping > /dev/null 2>&1; do
    echo "Redis is unavailable - sleeping"
    sleep 2
  done
  echo "Redis is up - continuing"
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

# Execute the main command
exec "$@"

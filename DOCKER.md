# EVIL2ROOT Trading Bot - Docker Guide

This guide explains how to work with the containerized version of the EVIL2ROOT Trading Bot system.

## Container Architecture

The system consists of the following Docker containers:

- **trading-bot**: Core trading logic and model execution
- **ai-validation**: Secondary AI system that validates trading decisions
- **web-ui**: Flask-based dashboard for monitoring
- **database**: PostgreSQL database for storing trading data
- **redis**: Redis instance for inter-service communication

## Quick Start

```bash
# Make setup script executable
chmod +x docker/setup-permissions.sh

# Make entrypoint scripts executable
./docker/setup-permissions.sh

# Build and start all containers
make build
make up

# OR with Docker Compose directly
docker compose up --build -d
```

## Directory Structure

- `/docker/services/` - Contains Dockerfiles for each service
- `/docker/init-db/` - Database initialization scripts
- `/docker/services/entrypoint-*.sh` - Entrypoint scripts for containers

## Container Management

### Starting and Stopping

```bash
# Start all services in the background
make up

# Start with logs visible
make up-log

# Stop all services
make down

# Restart all services
make restart

# Restart a specific service
make restart-trading-bot
```

### Logs and Monitoring

```bash
# View logs from all services
make logs

# View logs for a specific service
make logs-trading-bot
make logs-ai-validation
make logs-web-ui

# Check container health status
./docker/check-health.sh

# List running containers
make ps
```

### Database Operations

```bash
# Create a database backup
make backup

# Access PostgreSQL CLI
make db-cli

# Access Redis CLI
make redis-cli
```

### Container Shell Access

```bash
# Access shell in a container
make shell-trading-bot
make shell-ai-validation
make shell-web-ui
```

## Data Persistence

The following data is persisted through Docker volumes:

- **PostgreSQL data**: Stored in a named volume `postgres_data`
- **Redis data**: Stored in a named volume `redis_data`
- **Market data**: Mounted from host path `./data` to `/app/data` in containers
- **Model files**: Mounted from host path `./saved_models` to `/app/saved_models`
- **Log files**: Mounted from host path `./logs` to `/app/logs`

## Configuration

Environment variables are managed through the `.env` file and the Docker Compose configuration. The system is designed to automatically pick up changes to the `.env` file when containers are restarted.

## Container Dependencies and Health Checks

The system uses entrypoint scripts to ensure services wait for their dependencies to be available:

1. The database and Redis containers start first
2. The AI validation service waits for Redis to be available
3. The trading bot waits for both the database and Redis
4. The web-ui waits for the database, Redis, and trading bot

Each service includes health checks to verify proper operation:

- **Database**: Checks if PostgreSQL is accepting connections
- **Redis**: Verifies Redis is responding to ping commands
- **Trading Bot**: Verifies log file creation
- **AI Validation**: Checks Redis connection
- **Web UI**: Tests HTTP response

## Clean Up

```bash
# Remove containers but keep volumes
make down

# Remove containers and volumes
make clean

# Full cleanup including images
make purge
```

## Development Workflow

For development with Docker:

1. Make code changes on your host machine
2. Use `make update` to rebuild and restart affected services
3. Monitor logs with `make logs` to check for errors
4. Access the web interface at http://localhost:5000/

## Troubleshooting

If containers fail to start:

1. Check logs with `make logs` or `docker compose logs`
2. Verify the `.env` file has correct configuration
3. Run `./docker/check-health.sh` to see container health status
4. Ensure entrypoint scripts are executable (`./docker/setup-permissions.sh`)
5. Check if PostgreSQL and Redis are running (`make ps`)

## Performance Considerations

- The trading bot and AI validation services are set up to use optimized images with numerical computation libraries
- Consider enabling resource limits in docker-compose.yml for production use

## Security Notes

- Database credentials are stored in the `.env` file
- The PostgreSQL database is exposed on port 5432 (secure in production)
- The Redis instance is exposed on port 6379 (secure in production)
- Telegram token is stored in the `.env` file

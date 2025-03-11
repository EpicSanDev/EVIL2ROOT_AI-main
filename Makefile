# EVIL2ROOT Trading Bot - Docker operations
.PHONY: build up down logs ps clean restart purge backup test shell

# Default target when just running 'make'
all: up

# Build or rebuild the Docker containers
build:
	docker compose build

# Start the application with detached mode (running in background)
up:
	docker compose up -d

# Start with logs visible
up-log:
	docker compose up

# Stop the services
down:
	docker compose down

# Show container logs
logs:
	docker compose logs -f

# Show logs for a specific service
# Usage: make logs-SERVICE (e.g. make logs-trading-bot)
logs-%:
	docker compose logs -f $*

# Display the status of containers
ps:
	docker compose ps

# Remove volumes, cached data (full reset)
clean:
	docker compose down -v

# Restart all services
restart:
	docker compose restart

# Restart a specific service
# Usage: make restart-SERVICE (e.g. make restart-web-ui)
restart-%:
	docker compose restart $*

# Warning: Remove all Docker volumes, containers, images related to this project
purge:
	docker compose down -v
	docker volume prune -f

# Create a backup of the database
backup:
	docker compose exec database pg_dump -U $(shell grep DB_USER .env | cut -d '=' -f2) \
		$(shell grep DB_NAME .env | cut -d '=' -f2) > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Run tests inside a container
test:
	docker compose run --rm trading-bot python -m unittest discover

# Open a shell in a container
# Usage: make shell-SERVICE (e.g. make shell-trading-bot)
shell-%:
	docker compose exec $* /bin/bash || docker compose exec $* /bin/sh

# Update application code without rebuilding containers
update:
	docker compose up -d --no-deps --build trading-bot web-ui ai-validation

# Watch logs from all services
watch-all:
	docker compose logs -f --tail=100

# Open database CLI
db-cli:
	docker compose exec database psql -U $(shell grep DB_USER .env | cut -d '=' -f2) \
		-d $(shell grep DB_NAME .env | cut -d '=' -f2)

# Open Redis CLI
redis-cli:
	docker compose exec redis redis-cli

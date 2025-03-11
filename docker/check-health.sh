#!/bin/bash

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "=== EVIL2ROOT Trading Bot - Container Health Status ==="
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Get container status
containers=$(docker compose ps --services 2>/dev/null)

if [ -z "$containers" ]; then
  echo -e "${YELLOW}No containers found. Have you started the services?${NC}"
  echo -e "Run ${GREEN}make up${NC} or ${GREEN}docker compose up -d${NC} to start services."
  exit 1
fi

echo -e "${YELLOW}Container Status:${NC}"
docker compose ps

echo
echo -e "${YELLOW}Health Checks:${NC}"

# Check each service health status
for service in $containers; do
  container_id=$(docker compose ps -q $service)
  
  if [ -z "$container_id" ]; then
    echo -e "$service: ${RED}Not running${NC}"
    continue
  fi

  # Check if container has healthcheck
  if docker inspect --format='{{.Config.Healthcheck}}' $container_id | grep -q "Status"; then
    health=$(docker inspect --format='{{json .State.Health.Status}}' $container_id | tr -d '"')
    
    case $health in
      "healthy")
        echo -e "$service: ${GREEN}Healthy${NC}"
        ;;
      "unhealthy")
        echo -e "$service: ${RED}Unhealthy${NC}"
        ;;
      "starting")
        echo -e "$service: ${YELLOW}Starting${NC}"
        ;;
      *)
        echo -e "$service: ${YELLOW}Status unknown${NC}"
        ;;
    esac
  else
    running=$(docker inspect --format='{{.State.Running}}' $container_id)
    if [ "$running" = "true" ]; then
      echo -e "$service: ${YELLOW}Running (no health check defined)${NC}"
    else
      echo -e "$service: ${RED}Not running${NC}"
    fi
  fi
done

echo
echo "=== Resource Usage ==="
echo -e "${YELLOW}Container Resource Usage:${NC}"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

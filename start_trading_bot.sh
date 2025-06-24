#!/bin/bash

# EVIL2ROOT Trading Bot Unified System Startup Script
# Version: 2.0.0
# Author: EVIL2ROOT Team
# License: MIT

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logo and header
echo "============================================================"
echo "🤖 EVIL2ROOT Trading Bot - Unified System 🤖"
echo "============================================================"
echo "Version: 2.0.0"
echo "Author: EVIL2ROOT Team"
echo "License: MIT"
echo "============================================================"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

print_status "Docker is available and running"

# Check if Docker Compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    print_status "Docker Compose is available (docker-compose)"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    print_status "Docker Compose is available (docker compose)"
else
    print_error "Docker Compose is not available"
    exit 1
fi

# Create the unified docker-compose file if it doesn't exist
if [[ ! -f "docker-compose.unified.yml" ]]; then
    print_info "Creating docker-compose.unified.yml file..."
    cat > docker-compose.unified.yml << 'EOF'
services:
  trading-app:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        USE_TALIB_MOCK: "false"
        INSTALL_FULL_DEPS: "true"
        SKIP_OPTIONAL_DEPS: "false"
    container_name: evil2root-trading-app
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - PYTHONPATH=/app
      - DATABASE_URL=postgresql://trading_user:trading_pass@postgres:5432/trading_db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./logs:/app/logs
      - ./saved_models:/app/saved_models
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15-alpine
    container_name: evil2root-postgres
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: trading_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: evil2root-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    container_name: evil2root-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./frontend/dist:/usr/share/nginx/html
    depends_on:
      - trading-app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
EOF
    print_status "Created docker-compose.unified.yml"
fi

# Set mode (default to full)
MODE=${1:-full}

print_info "🚀 Starting EVIL2ROOT Trading Bot in $MODE mode..."

case $MODE in
    "full")
        print_info "🎯 Starting full system..."
        $COMPOSE_CMD -f docker-compose.unified.yml up --build -d
        ;;
    "dev")
        print_info "🔧 Starting development mode..."
        $COMPOSE_CMD -f docker-compose.unified.yml up --build
        ;;
    "logs")
        print_info "📋 Showing logs..."
        $COMPOSE_CMD -f docker-compose.unified.yml logs -f
        ;;
    "stop")
        print_info "🛑 Stopping services..."
        $COMPOSE_CMD -f docker-compose.unified.yml down
        ;;
    "restart")
        print_info "🔄 Restarting services..."
        $COMPOSE_CMD -f docker-compose.unified.yml down
        $COMPOSE_CMD -f docker-compose.unified.yml up --build -d
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo "Available modes: full, dev, logs, stop, restart"
        exit 1
        ;;
esac

if [[ $MODE == "full" ]]; then
    echo ""
    print_status "🎉 EVIL2ROOT Trading Bot started successfully!"
    echo ""
    print_info "📊 Access the application:"
    print_info "  - Web Interface: http://localhost:80"
    print_info "  - API: http://localhost:8000"
    print_info "  - Health Check: http://localhost:8000/health"
    echo ""
    print_info "📋 Useful commands:"
    print_info "  - View logs: ./start_trading_bot.sh logs"
    print_info "  - Stop services: ./start_trading_bot.sh stop"
    print_info "  - Restart: ./start_trading_bot.sh restart"
    echo ""
    print_info "🔍 Check status with: $COMPOSE_CMD -f docker-compose.unified.yml ps"
fi
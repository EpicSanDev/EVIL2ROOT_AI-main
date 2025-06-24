#!/bin/bash

# EVIL2ROOT Trading Bot - Quick Start Script
# This script provides a simple way to start the entire trading system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/startup.log"

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log "✅ Docker is available and running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        error "Docker Compose is not available. Please install docker-compose."
        exit 1
    fi
    
    log "✅ Docker Compose is available ($COMPOSE_CMD)"
}

# Function to display banner
show_banner() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}🤖 EVIL2ROOT Trading Bot - Unified System 🤖${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${BLUE}Version: 2.0.0${NC}"
    echo -e "${BLUE}Author: EVIL2ROOT Team${NC}"
    echo -e "${BLUE}License: MIT${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start          Start the full trading system (default)"
    echo "  paper          Start in paper trading mode (safe mode)"
    echo "  web            Start only web interface"
    echo "  api            Start only API backend"
    echo "  analysis       Start only analysis bot"
    echo "  stop           Stop all services"
    echo "  restart        Restart all services"
    echo "  logs           Show logs from all services"
    echo "  status         Show status of all services"
    echo "  build          Build Docker images"
    echo "  clean          Clean up containers and volumes"
    echo "  setup          Initial setup and configuration"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start       # Start full system"
    echo "  $0 paper       # Start in safe paper trading mode"
    echo "  $0 logs        # View logs"
    echo "  $0 stop        # Stop everything"
    echo ""
}

# Function to setup initial configuration
setup_system() {
    log "🔧 Setting up EVIL2ROOT Trading Bot..."
    
    # Create necessary directories
    log "📁 Creating necessary directories..."
    mkdir -p logs saved_models data cache/market_data secrets config/environments nginx/ssl
    
    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.production" ]; then
            cp .env.production .env
            log "📄 Copied .env.production to .env"
        elif [ -f ".env.example" ]; then
            cp .env.example .env
            log "📄 Copied .env.example to .env"
            warning "Please edit .env file with your actual API keys and configuration"
        else
            error "No environment file template found"
            exit 1
        fi
    fi
    
    # Set permissions
    chmod +x unified_main.py start_trading_bot.sh 2>/dev/null || true
    
    log "✅ Setup completed!"
    echo ""
    warning "IMPORTANT: Please edit the .env file and update your API keys:"
    echo "  - TELEGRAM_TOKEN=your_telegram_bot_token"
    echo "  - OPENROUTER_API_KEY=your_openrouter_api_key"
    echo "  - FINNHUB_API_KEY=your_finnhub_api_key"
    echo "  - COINBASE_API_KEY=your_coinbase_api_key"
    echo ""
}

# Function to build Docker images
build_images() {
    log "🏗️  Building Docker images..."
    
    # Build the unified trading app
    log "Building unified trading application..."
    docker build -f Dockerfile.unified -t evil2root-trading:latest . || {
        error "Failed to build trading application"
        exit 1
    }
    
    # Build frontend if Dockerfile exists
    if [ -f "frontend/Dockerfile" ]; then
        log "Building frontend application..."
        docker build -t evil2root-frontend:latest frontend/ || {
            warning "Failed to build frontend application"
        }
    fi
    
    log "✅ Docker images built successfully"
}

# Function to start services
start_services() {
    local mode=${1:-"full"}
    
    log "🚀 Starting EVIL2ROOT Trading Bot in $mode mode..."
    
    # Check if services are already running
    if $COMPOSE_CMD -f docker-compose.unified.yml ps | grep -q "Up"; then
        warning "Some services are already running. Use 'restart' to restart them."
    fi
    
    # Start services based on mode
    case $mode in
        "paper"|"safe")
            log "📊 Starting in paper trading mode (safe mode)..."
            ENABLE_LIVE_TRADING=false $COMPOSE_CMD -f docker-compose.unified.yml up -d postgres redis trading-app
            ;;
        "web")
            log "🌐 Starting web interface only..."
            $COMPOSE_CMD -f docker-compose.unified.yml up -d postgres redis trading-app frontend nginx
            ;;
        "api")
            log "🔌 Starting API backend only..."
            $COMPOSE_CMD -f docker-compose.unified.yml up -d postgres redis trading-app
            ;;
        "analysis")
            log "📈 Starting analysis bot only..."
            $COMPOSE_CMD -f docker-compose.unified.yml up -d postgres redis analysis-bot
            ;;
        "full"|*)
            log "🎯 Starting full system..."
            $COMPOSE_CMD -f docker-compose.unified.yml up -d
            ;;
    esac
    
    # Wait for services to be ready
    log "⏳ Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_service_health
    
    log "✅ EVIL2ROOT Trading Bot started successfully!"
    show_access_info
}

# Function to check service health
check_service_health() {
    log "🔍 Checking service health..."
    
    # Check database
    if $COMPOSE_CMD -f docker-compose.unified.yml exec -T postgres pg_isready &>/dev/null; then
        log "✅ Database is healthy"
    else
        warning "❌ Database is not ready"
    fi
    
    # Check Redis
    if $COMPOSE_CMD -f docker-compose.unified.yml exec -T redis redis-cli ping &>/dev/null; then
        log "✅ Redis is healthy"
    else
        warning "❌ Redis is not ready"
    fi
    
    # Check trading app
    if curl -f http://localhost:5000/health &>/dev/null || curl -f http://localhost:8000/health &>/dev/null; then
        log "✅ Trading application is healthy"
    else
        warning "❌ Trading application is not ready"
    fi
}

# Function to show access information
show_access_info() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}🎉 EVIL2ROOT Trading Bot is running! 🎉${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo "  📊 Frontend (React):     http://localhost:3000"
    echo "  🌐 Web Interface:        http://localhost:5000"
    echo "  🔌 API Backend:          http://localhost:8000"
    echo "  📈 API Documentation:    http://localhost:8000/docs"
    echo "  🗄️  Database Admin:       http://localhost:5050"
    echo "  🔄 Nginx Proxy:          http://localhost:80"
    echo ""
    echo -e "${YELLOW}Default Credentials:${NC}"
    echo "  👤 Web Interface:  admin / Evil2Root_Admin_2024!"
    echo "  🗄️  PgAdmin:        admin@evil2root.local / pgadmin2024secure"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  📊 View logs:       $0 logs"
    echo "  ⏹️  Stop system:     $0 stop"
    echo "  🔄 Restart:         $0 restart"
    echo "  📋 Status:          $0 status"
    echo ""
}

# Function to stop services
stop_services() {
    log "⏹️  Stopping EVIL2ROOT Trading Bot..."
    $COMPOSE_CMD -f docker-compose.unified.yml down
    log "✅ All services stopped"
}

# Function to restart services
restart_services() {
    log "🔄 Restarting EVIL2ROOT Trading Bot..."
    stop_services
    sleep 5
    start_services "full"
}

# Function to show logs
show_logs() {
    log "📋 Showing logs from all services..."
    $COMPOSE_CMD -f docker-compose.unified.yml logs -f --tail=100
}

# Function to show status
show_status() {
    log "📊 Service Status:"
    $COMPOSE_CMD -f docker-compose.unified.yml ps
    
    echo ""
    log "🌐 Service Health:"
    
    # Check each service
    services=("http://localhost:5000/health:Web Interface" "http://localhost:8000/health:API Backend" "http://localhost:3000:Frontend")
    
    for service in "${services[@]}"; do
        url=${service%%:*}
        name=${service#*:}
        if curl -f "$url" &>/dev/null; then
            echo -e "  ✅ $name: ${GREEN}Healthy${NC}"
        else
            echo -e "  ❌ $name: ${RED}Unhealthy${NC}"
        fi
    done
}

# Function to clean up
clean_system() {
    log "🧹 Cleaning up EVIL2ROOT Trading Bot..."
    
    read -p "This will remove all containers, volumes, and images. Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $COMPOSE_CMD -f docker-compose.unified.yml down -v --remove-orphans
        docker image prune -f
        docker volume prune -f
        log "✅ Cleanup completed"
    else
        log "Cleanup cancelled"
    fi
}

# Main script logic
main() {
    show_banner
    
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Parse command line arguments
    case ${1:-"start"} in
        "start"|"")
            start_services "full"
            ;;
        "paper"|"safe")
            start_services "paper"
            ;;
        "web")
            start_services "web"
            ;;
        "api")
            start_services "api"
            ;;
        "analysis")
            start_services "analysis"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "build")
            build_images
            ;;
        "clean")
            clean_system
            ;;
        "setup")
            setup_system
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
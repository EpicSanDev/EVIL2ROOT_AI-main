#!/usr/bin/env python3
"""
EVIL2ROOT Trading Bot - Cleanup and Improvement Script
====================================================

This script automatically cleans up and improves the EVIL2ROOT Trading Bot project by:
- Removing duplicate files
- Fixing import inconsistencies
- Consolidating configurations
- Improving code structure
- Creating missing files

Author: EVIL2ROOT Team
Version: 2.0.0
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import json
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectCleaner:
    """Clean up and improve the EVIL2ROOT Trading Bot project"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.issues_found = []
        self.fixes_applied = []
        
        logger.info(f"🧹 Initializing project cleanup for: {self.project_root}")
    
    def find_duplicate_files(self):
        """Find and handle duplicate files"""
        logger.info("🔍 Looking for duplicate files...")
        
        # Known duplicates to handle
        duplicates = {
            "trading.py": [
                "app/trading.py",
                "src/core/trading.py", 
                "app/core/trading.py"
            ],
            "__init__.py": [
                "app/models/__init__.py",
                "app/models/__init__.py.new"
            ]
        }
        
        for file_type, paths in duplicates.items():
            existing_paths = [p for p in paths if (self.project_root / p).exists()]
            
            if len(existing_paths) > 1:
                logger.warning(f"⚠️  Found {len(existing_paths)} copies of {file_type}")
                self.issues_found.append(f"Duplicate {file_type}: {existing_paths}")
                
                # Keep the most complete version and remove others
                self._consolidate_duplicates(file_type, existing_paths)
    
    def _consolidate_duplicates(self, file_type: str, paths: list):
        """Consolidate duplicate files by keeping the best version"""
        if file_type == "trading.py":
            # Keep src/core/trading.py as the main trading module
            main_path = "src/core/trading.py"
            if main_path in paths:
                logger.info(f"✅ Keeping {main_path} as the main trading module")
                
                # Remove other versions
                for path in paths:
                    if path != main_path:
                        backup_path = f"{path}.backup"
                        shutil.move(self.project_root / path, self.project_root / backup_path)
                        logger.info(f"📦 Moved {path} to {backup_path}")
                        self.fixes_applied.append(f"Backed up duplicate: {path}")
        
        elif file_type == "__init__.py":
            # Remove the .new file
            for path in paths:
                if path.endswith(".new"):
                    os.remove(self.project_root / path)
                    logger.info(f"🗑️ Removed {path}")
                    self.fixes_applied.append(f"Removed duplicate: {path}")
    
    def fix_imports(self):
        """Fix import inconsistencies throughout the project"""
        logger.info("🔧 Fixing import inconsistencies...")
        
        # Define the standard import mappings
        import_fixes = {
            "from src.core.trading import": "from src.core.trading import",
            "from src.core.trading import": "from src.core.trading import",
            "import src.core.trading": "import src.core.trading",
            "import src.core.trading": "import src.core.trading"
        }
        
        # Files to fix
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply import fixes
                for old_import, new_import in import_fixes.items():
                    content = content.replace(old_import, new_import)
                
                # Write back if changed
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"🔄 Fixed imports in {file_path.relative_to(self.project_root)}")
                    self.fixes_applied.append(f"Fixed imports: {file_path.relative_to(self.project_root)}")
                    
            except Exception as e:
                logger.error(f"❌ Error fixing imports in {file_path}: {e}")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during processing"""
        skip_patterns = [
            "venv/", ".venv/", "__pycache__/", ".git/", 
            "node_modules/", ".backup", ".tmp"
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def create_missing_configs(self):
        """Create missing configuration files"""
        logger.info("📋 Creating missing configuration files...")
        
        # Create .env.example if missing
        env_example_path = self.project_root / ".env.example"
        if not env_example_path.exists():
            env_content = """# EVIL2ROOT Trading Bot Configuration

# Database Configuration
DATABASE_URL=postgresql://trading_user:trading_pass@localhost:5432/trading_db
REDIS_URL=redis://localhost:6379/0

# API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
TELEGRAM_TOKEN=your_telegram_bot_token_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_WEBHOOK_SECRET=your_coinbase_webhook_secret_here

# Trading Configuration
INITIAL_BALANCE=100000
ENABLE_REAL_TIME_TRADING=false
RISK_PERCENTAGE=0.02
MAX_POSITIONS=5

# Model Configuration
USE_TALIB_MOCK=false
ENABLE_AUTO_LEARNING=true
SCAN_INTERVAL_SECONDS=60

# Logging
LOG_LEVEL=INFO
"""
            with open(env_example_path, 'w') as f:
                f.write(env_content)
            logger.info("✅ Created .env.example")
            self.fixes_applied.append("Created .env.example")
        
        # Create improved docker-compose.unified.yml
        self._create_unified_docker_compose()
        
        # Create configuration directory structure
        self._create_config_structure()
    
    def _create_unified_docker_compose(self):
        """Create an improved unified docker-compose file"""
        compose_path = self.project_root / "docker-compose.unified.yml"
        
        compose_content = """services:
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
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./saved_models:/app/saved_models
      - ./data:/app/data
      - ./config:/app/config
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    image: postgres:15-alpine
    container_name: evil2root-postgres
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: trading_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user -d trading_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: evil2root-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

  nginx:
    image: nginx:alpine
    container_name: evil2root-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - trading-app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: evil2root-scheduler
    environment:
      - PYTHONPATH=/app
      - DATABASE_URL=postgresql://trading_user:trading_pass@postgres:5432/trading_db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    command: python app/analytics/market_analysis_scheduler.py

  monitoring:
    image: prom/prometheus:latest
    container_name: evil2root-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:

networks:
  default:
    name: evil2root-network
    driver: bridge
"""
        
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        logger.info("✅ Created improved docker-compose.unified.yml")
        self.fixes_applied.append("Created improved docker-compose.unified.yml")
    
    def _create_config_structure(self):
        """Create proper configuration directory structure"""
        config_dirs = [
            "config/environments",
            "config/models",
            "config/trading",
            "sql",
            "logs",
            "data/historical",
            "data/processed",
            "saved_models",
            "secrets"
        ]
        
        for dir_path in config_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            gitkeep_path = full_path / ".gitkeep"
            if not gitkeep_path.exists() and not any(full_path.iterdir()):
                gitkeep_path.touch()
        
        logger.info("✅ Created proper directory structure")
        self.fixes_applied.append("Created proper directory structure")
    
    def create_database_init(self):
        """Create database initialization SQL"""
        sql_init_path = self.project_root / "sql" / "init.sql"
        
        if not sql_init_path.exists():
            sql_content = """-- EVIL2ROOT Trading Bot Database Initialization

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create trading data tables
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT NOT NULL,
    sma_20 DECIMAL(12,4),
    sma_50 DECIMAL(12,4),
    rsi DECIMAL(6,3),
    macd DECIMAL(12,4),
    bb_upper DECIMAL(12,4),
    bb_lower DECIMAL(12,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    quantity DECIMAL(12,4) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    strategy VARCHAR(50),
    profit_loss DECIMAL(12,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(12,4) NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4),
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),
    status VARCHAR(20) DEFAULT 'open',
    opened_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP,
    profit_loss DECIMAL(12,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    total_value DECIMAL(15,4) NOT NULL,
    daily_return DECIMAL(8,4),
    volatility DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(6,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(date);

-- Insert some initial data if needed
INSERT INTO performance_metrics (date, total_value, daily_return, volatility, sharpe_ratio, max_drawdown, win_rate)
VALUES (CURRENT_DATE, 100000.00, 0.00, 0.00, 0.00, 0.00, 0.00)
ON CONFLICT DO NOTHING;
"""
            
            with open(sql_init_path, 'w') as f:
                f.write(sql_content)
            logger.info("✅ Created database initialization SQL")
            self.fixes_applied.append("Created database initialization SQL")
    
    def improve_dockerfile(self):
        """Improve the Dockerfile with better error handling"""
        dockerfile_path = self.project_root / "Dockerfile"
        
        if not dockerfile_path.exists():
            logger.warning("⚠️ Dockerfile not found")
            return
        
        # The Dockerfile was already fixed earlier, so we just verify it's correct
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        if "PYTHONDONTWRITEBYTECODE=1" in content:
            logger.info("✅ Dockerfile appears to be correctly formatted")
        else:
            logger.warning("⚠️ Dockerfile may still have issues")
    
    def create_makefile(self):
        """Create a Makefile for easy project management"""
        makefile_path = self.project_root / "Makefile"
        
        makefile_content = """# EVIL2ROOT Trading Bot Makefile

.PHONY: help install test clean start stop logs build deploy

help:
	@echo "EVIL2ROOT Trading Bot - Available commands:"
	@echo "  install    - Install dependencies and setup environment"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean up temporary files"
	@echo "  start      - Start the trading bot"
	@echo "  stop       - Stop the trading bot"
	@echo "  logs       - View logs"
	@echo "  build      - Build Docker images"
	@echo "  deploy     - Deploy to production"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -m nltk.downloader punkt averaged_perceptron_tagger
	mkdir -p logs data saved_models

test:
	python -m pytest tests/ -v --cov=src --cov-report=html

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf build dist *.egg-info

start:
	./start_trading_bot.sh full

start-dev:
	./start_trading_bot.sh dev

stop:
	./start_trading_bot.sh stop

restart:
	./start_trading_bot.sh restart

logs:
	./start_trading_bot.sh logs

build:
	docker build -t evil2root-trading:latest .

build-no-cache:
	docker build --no-cache -t evil2root-trading:latest .

deploy:
	docker-compose -f docker-compose.unified.yml up --build -d

deploy-stop:
	docker-compose -f docker-compose.unified.yml down

deploy-logs:
	docker-compose -f docker-compose.unified.yml logs -f

# Development commands
format:
	black src/ app/ tests/
	isort src/ app/ tests/

lint:
	flake8 src/ app/ tests/
	pylint src/ app/ tests/

check:
	python -m py_compile unified_main.py
	python -c "import sys; sys.path.append('.'); import unified_main"

setup-secrets:
	mkdir -p secrets
	echo "trading_user" > secrets/db_user.txt
	echo "trading_pass" > secrets/db_password.txt
	echo "trading_db" > secrets/postgres_db.txt
	echo "super_secret_key_change_this" > secrets/secret_key.txt
	echo "your_telegram_token_here" > secrets/telegram_token.txt
	echo "your_finnhub_api_key_here" > secrets/finnhub_api_key.txt
	echo "your_openrouter_api_key_here" > secrets/openrouter_api_key.txt
	echo "your_coinbase_api_key_here" > secrets/coinbase_api_key.txt
	echo "your_coinbase_webhook_secret_here" > secrets/coinbase_webhook_secret.txt
	@echo "⚠️  Remember to update the secret files with your actual credentials!"

docs:
	python -m pydoc -w unified_main
	python -m pydoc -w src.core.trading
"""
        
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        logger.info("✅ Created Makefile for project management")
        self.fixes_applied.append("Created Makefile")
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        logger.info("🚀 Starting EVIL2ROOT Trading Bot cleanup and improvement...")
        
        try:
            # Step 1: Find and handle duplicates
            self.find_duplicate_files()
            
            # Step 2: Fix imports
            self.fix_imports()
            
            # Step 3: Create missing configs
            self.create_missing_configs()
            
            # Step 4: Create database init
            self.create_database_init()
            
            # Step 5: Improve Dockerfile
            self.improve_dockerfile()
            
            # Step 6: Create Makefile
            self.create_makefile()
            
            # Step 7: Make scripts executable
            self._make_scripts_executable()
            
            # Report results
            self._print_summary()
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            raise
    
    def _make_scripts_executable(self):
        """Make shell scripts executable"""
        script_files = [
            "start_trading_bot.sh",
            "scripts/shell/*.sh",
            "docker/*.sh"
        ]
        
        for pattern in script_files:
            for script_path in self.project_root.glob(pattern):
                if script_path.is_file():
                    os.chmod(script_path, 0o755)
                    logger.info(f"✅ Made {script_path.relative_to(self.project_root)} executable")
        
        self.fixes_applied.append("Made shell scripts executable")
    
    def _print_summary(self):
        """Print summary of cleanup operations"""
        logger.info("\n" + "="*60)
        logger.info("🎉 CLEANUP AND IMPROVEMENT COMPLETE!")
        logger.info("="*60)
        
        if self.issues_found:
            logger.info(f"\n📋 Issues Found ({len(self.issues_found)}):")
            for issue in self.issues_found:
                logger.info(f"  • {issue}")
        
        if self.fixes_applied:
            logger.info(f"\n✅ Fixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                logger.info(f"  • {fix}")
        
        logger.info("\n🚀 Next Steps:")
        logger.info("  1. Review and update .env file with your credentials")
        logger.info("  2. Run 'make setup-secrets' to create secret files")
        logger.info("  3. Update secret files with your actual API keys")
        logger.info("  4. Run './start_trading_bot.sh full' to start the system")
        logger.info("  5. Check logs with './start_trading_bot.sh logs'")
        logger.info("\n📚 Use 'make help' to see all available commands")

def main():
    """Main entry point"""
    logger.info("EVIL2ROOT Trading Bot - Cleanup and Improvement Tool")
    logger.info("="*60)
    
    # Change to project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Run cleanup
    cleaner = ProjectCleaner(project_root)
    cleaner.run_cleanup()

if __name__ == "__main__":
    main()
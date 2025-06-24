# EVIL2ROOT Trading Bot Makefile

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EVIL2ROOT Trading Bot - Unified Main Entry Point

This unified main script resolves the dual architecture issue by combining
the best elements from both app/core/trading.py and src/core/trading.py
into a single, coherent trading system.

Author: EVIL2ROOT Team
Version: 2.0.0
License: MIT
"""

import argparse
import logging
import os
import sys
import asyncio
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add both directories to Python path to ensure all imports work
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "app"))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Import logging utilities
try:
    from src.utils.log_config import setup_logging
    logger = setup_logging('unified_main', 'unified_trading_bot.log')
except ImportError:
    # Fallback logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/unified_trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('unified_main')

class UnifiedTradingSystem:
    """
    Unified Trading System that combines Flask web interface,
    FastAPI backend, and advanced AI trading models.
    """
    
    def __init__(self, mode: str = 'paper', config_path: str = None):
        self.mode = mode
        self.config_path = config_path or '.env.production'
        self.running = False
        self.web_app = None
        self.api_app = None
        self.trading_bot = None
        self.data_manager = None
        self.analysis_bot = None
        
        # Load configuration
        self._load_configuration()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Unified Trading System initialized in {mode} mode")
    
    def _load_configuration(self):
        """Load and validate configuration"""
        try:
            if os.path.exists(self.config_path):
                load_dotenv(self.config_path)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                
            # Validate critical configuration
            required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {missing_vars}")
                logger.info("Please ensure your .env file is properly configured")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize data manager
            symbols = os.getenv('TRADING_SYMBOLS', 'BTC-USD,ETH-USD,AAPL').split(',')
            from app.core.trading import DataManager
            self.data_manager = DataManager(symbols=symbols)
            logger.info(f"Data manager initialized with {len(symbols)} symbols")
            
            # Initialize trading bot (prefer src version for advanced features)
            try:
                from src.core.trading import TradingBot
                initial_balance = float(os.getenv('INITIAL_BALANCE', '100000.0'))
                self.trading_bot = TradingBot(
                    initial_balance=initial_balance,
                    enable_auto_learning=os.getenv('ENABLE_AUTO_LEARNING', 'true').lower() == 'true'
                )
                logger.info("Advanced trading bot (src) initialized successfully")
            except ImportError:
                # Fallback to app version
                from app.core.trading import TradingBot
                initial_balance = float(os.getenv('INITIAL_BALANCE', '100000.0'))
                self.trading_bot = TradingBot(initial_balance=initial_balance)
                logger.info("Basic trading bot (app) initialized as fallback")
            
            # Initialize web interface
            try:
                from app import create_app
                self.web_app = create_app()
                logger.info("Flask web application initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Flask app: {e}")
            
            # Initialize API backend
            try:
                from src.api.app import create_app as create_api_app
                self.api_app = create_api_app()
                logger.info("FastAPI backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FastAPI: {e}")
            
            # Initialize analysis bot if enabled
            if os.getenv('ENABLE_DAILY_ANALYSIS', 'true').lower() == 'true':
                try:
                    from app.bots.daily_analysis_bot import DailyAnalysisBot
                    self.analysis_bot = DailyAnalysisBot(symbols=symbols)
                    logger.info("Daily analysis bot initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize analysis bot: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def start(self):
        """Start the unified trading system"""
        logger.info("Starting EVIL2ROOT Unified Trading System...")
        self.running = True
        
        try:
            if self.mode == 'web':
                self._start_web_mode()
            elif self.mode == 'api':
                self._start_api_mode()
            elif self.mode == 'trading':
                self._start_trading_mode()
            elif self.mode == 'analysis':
                self._start_analysis_mode()
            elif self.mode == 'full':
                self._start_full_mode()
            else:
                # Default: paper trading mode
                self._start_paper_trading_mode()
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Error during system operation: {e}")
        finally:
            self.stop()
    
    def _start_web_mode(self):
        """Start only the web interface"""
        if self.web_app:
            logger.info("Starting web interface on http://localhost:5000")
            self.web_app.run(host='0.0.0.0', port=5000, debug=False)
        else:
            logger.error("Web application not initialized")
    
    def _start_api_mode(self):
        """Start only the API backend"""
        if self.api_app:
            import uvicorn
            logger.info("Starting API backend on http://localhost:8000")
            uvicorn.run(self.api_app, host='0.0.0.0', port=8000)
        else:
            logger.error("API application not initialized")
    
    def _start_trading_mode(self):
        """Start only the trading bot"""
        if self.trading_bot and self.data_manager:
            logger.info(f"Starting trading bot in {self.mode} mode")
            
            # Train models if needed
            if not os.path.exists('saved_models/global_price_model.h5'):
                logger.info("Training models before starting trading...")
                self.trading_bot.train_all_models(self.data_manager)
            
            # Start real-time trading
            self.trading_bot.start_real_time_scanning(self.data_manager)
        else:
            logger.error("Trading bot or data manager not initialized")
    
    def _start_analysis_mode(self):
        """Start only the analysis bot"""
        if self.analysis_bot:
            logger.info("Starting daily analysis bot")
            self.analysis_bot.start_scheduled_analysis()
        else:
            logger.error("Analysis bot not initialized")
    
    def _start_paper_trading_mode(self):
        """Start paper trading mode (safe default)"""
        logger.info("Starting paper trading mode")
        
        if self.trading_bot and self.data_manager:
            # Ensure we're in paper trading mode
            os.environ['ENABLE_LIVE_TRADING'] = 'false'
            
            # Start trading simulation
            self.trading_bot.start_real_time_scanning(self.data_manager)
        else:
            logger.error("Required components not initialized for paper trading")
    
    def _start_full_mode(self):
        """Start all components (web + api + trading + analysis)"""
        logger.info("Starting full system mode")
        
        threads = []
        
        # Start web interface in a thread
        if self.web_app:
            web_thread = threading.Thread(
                target=lambda: self.web_app.run(host='0.0.0.0', port=5000, debug=False),
                name="WebInterface"
            )
            web_thread.daemon = True
            web_thread.start()
            threads.append(web_thread)
            logger.info("Web interface started on port 5000")
        
        # Start API backend in a thread
        if self.api_app:
            import uvicorn
            api_thread = threading.Thread(
                target=lambda: uvicorn.run(self.api_app, host='0.0.0.0', port=8000),
                name="APIBackend"
            )
            api_thread.daemon = True
            api_thread.start()
            threads.append(api_thread)
            logger.info("API backend started on port 8000")
        
        # Start analysis bot in a thread
        if self.analysis_bot:
            analysis_thread = threading.Thread(
                target=self.analysis_bot.start_scheduled_analysis,
                name="AnalysisBot"
            )
            analysis_thread.daemon = True
            analysis_thread.start()
            threads.append(analysis_thread)
            logger.info("Analysis bot started")
        
        # Run trading bot in main thread
        if self.trading_bot and self.data_manager:
            logger.info("Starting trading bot in main thread")
            self.trading_bot.start_real_time_scanning(self.data_manager)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
    
    def stop(self):
        """Stop the trading system gracefully"""
        if not self.running:
            return
            
        logger.info("Stopping EVIL2ROOT Trading System...")
        self.running = False
        
        # Stop components gracefully
        if self.trading_bot:
            self.trading_bot.status = 'stopped'
        
        logger.info("EVIL2ROOT Trading System stopped successfully")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EVIL2ROOT Unified Trading Bot')
    
    parser.add_argument(
        '--mode', 
        choices=['web', 'api', 'trading', 'analysis', 'paper', 'full'], 
        default='paper',
        help='System operation mode (default: paper)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='.env.production',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--symbols', 
        type=str,
        help='Comma-separated list of symbols to trade'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--force-train', 
        action='store_true',
        help='Force model training before starting'
    )
    
    parser.add_argument(
        '--docker', 
        action='store_true',
        help='Running in Docker environment'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Override symbols if provided
    if args.symbols:
        os.environ['TRADING_SYMBOLS'] = args.symbols
    
    logger.info("="*60)
    logger.info("🤖 EVIL2ROOT Trading Bot - Unified System 🤖")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Docker: {args.docker}")
    logger.info("="*60)
    
    try:
        # Create and start the unified system
        system = UnifiedTradingSystem(mode=args.mode, config_path=args.config)
        system.start()
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
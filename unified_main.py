#!/usr/bin/env python3
"""
EVIL2ROOT Trading Bot - Unified Main Entry Point
================================================

This is the main entry point for the EVIL2ROOT Trading Bot system.
It orchestrates all components including data management, AI models, trading execution, and monitoring.

Author: EVIL2ROOT Team
Version: 2.0.0
License: MIT
"""

import os
import sys
import logging
import asyncio
import signal
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Core imports
from src.core.trading import TradingBot, DataManager
from app.models.ensemble.ensemble_integrator import EnsembleIntegrator
from app.analytics.market_analysis_scheduler import MarketAnalysisScheduler
from app.monitoring.core.monitoring_enhanced import EnhancedMonitoringService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UnifiedTradingSystem:
    """
    Unified Trading System that coordinates all components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the unified trading system"""
        self.config = config
        self.is_running = False
        self.components = {}
        
        # Initialize logging
        os.makedirs('logs', exist_ok=True)
        logger.info("🚀 Initializing EVIL2ROOT Unified Trading System")
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize all system components"""
        try:
            # 1. Data Manager
            symbols = self.config.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
            logger.info(f"📊 Initializing Data Manager with symbols: {symbols}")
            self.data_manager = DataManager(symbols)
            self.components['data_manager'] = self.data_manager
            
            # 2. Trading Bot
            initial_balance = self.config.get('initial_balance', 100000)
            logger.info(f"🤖 Initializing Trading Bot with balance: ${initial_balance:,}")
            self.trading_bot = TradingBot(
                initial_balance=initial_balance,
                enable_auto_learning=self.config.get('enable_auto_learning', True),
                auto_learning_config_path=self.config.get('auto_learning_config_path', 'src/config/auto_learning_config.json')
            )
            self.components['trading_bot'] = self.trading_bot
            
            # 3. Ensemble Model Integrator
            logger.info("🧠 Initializing Ensemble Model Integrator")
            self.ensemble_integrator = EnsembleIntegrator()
            self.components['ensemble_integrator'] = self.ensemble_integrator
            
            # 4. Market Analysis Scheduler
            logger.info("📈 Initializing Market Analysis Scheduler")
            self.market_scheduler = MarketAnalysisScheduler()
            self.components['market_scheduler'] = self.market_scheduler
            
            # 5. Monitoring Service
            logger.info("🔍 Initializing Enhanced Monitoring Service")
            self.monitoring_service = EnhancedMonitoringService()
            self.components['monitoring_service'] = self.monitoring_service
            
            # 6. Telegram Bot (if enabled) - TODO: Implement TelegramBot class
            if self.config.get('enable_telegram_bot', False):
                logger.info("📱 Telegram Bot feature is not yet implemented")
                logger.warning("Telegram bot functionality will be added in future updates")
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def train_models(self):
        """Train all ML models"""
        logger.info("🎓 Starting model training...")
        try:
            # Train trading bot models
            self.trading_bot.train_all_models(self.data_manager)
            
            # Train RL model if data is available
            rl_data_file = self.config.get('rl_training_data', 'market_data_cleaned_auto.csv')
            if os.path.exists(rl_data_file):
                logger.info(f"Training RL model with data from {rl_data_file}")
                self.trading_bot.train_rl_model(rl_data_file)
            else:
                logger.warning(f"RL training data file not found: {rl_data_file}")
            
            logger.info("✅ Model training completed")
            
        except Exception as e:
            logger.error(f"❌ Error during model training: {e}")
            raise
    
    def run_backtest(self):
        """Run backtesting"""
        logger.info("📋 Running backtest...")
        try:
            backtest_data = self.config.get('backtest_data', 'market_data_cleaned_auto.csv')
            if not os.path.exists(backtest_data):
                logger.warning(f"Backtest data file not found: {backtest_data}")
                return None
                
            results = self.trading_bot.run_backtest(
                backtest_data,
                risk_percentage=self.config.get('risk_percentage', 0.02),
                max_positions=self.config.get('max_positions', 5)
            )
            
            logger.info("✅ Backtest completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during backtesting: {e}")
            return None
    
    async def start_async_components(self):
        """Start asynchronous components"""
        logger.info("🔄 Starting async components...")
        
        tasks = []
        
        # Start market analysis scheduler
        if 'market_scheduler' in self.components:
            tasks.append(
                asyncio.create_task(self.market_scheduler.start_async())
            )
        
        # Start monitoring service
        if 'monitoring_service' in self.components:
            tasks.append(
                asyncio.create_task(self.monitoring_service.start_monitoring())
            )
        
        # Start telegram bot - TODO: Implement when TelegramBot is available
        # if 'telegram_bot' in self.components:
        #     tasks.append(
        #         asyncio.create_task(self.telegram_bot.start_async())
        #     )
        
        return tasks
    
    def start_real_time_trading(self):
        """Start real-time trading"""
        logger.info("📈 Starting real-time trading...")
        
        try:
            # Start real-time scanning
            self.trading_bot.start_real_time_scanning(
                self.data_manager,
                interval_seconds=self.config.get('scan_interval_seconds', 60),
                enable_incremental_learning=self.config.get('enable_incremental_learning', True),
                auto_learning_cycle_hours=self.config.get('auto_learning_cycle_hours', 12)
            )
            
        except Exception as e:
            logger.error(f"❌ Error starting real-time trading: {e}")
            raise
    
    async def run_unified_system(self):
        """Run the complete unified system"""
        logger.info("🚀 Starting EVIL2ROOT Unified Trading System")
        
        try:
            self.is_running = True
            
            # 1. Train models if required
            if self.config.get('train_models_on_start', True):
                self.train_models()
            
            # 2. Run backtest if required
            if self.config.get('run_backtest_on_start', True):
                self.run_backtest()
            
            # 3. Start async components
            async_tasks = await self.start_async_components()
            
            # 4. Start real-time trading in a separate task
            if self.config.get('enable_real_time_trading', True):
                trading_task = asyncio.create_task(
                    asyncio.to_thread(self.start_real_time_trading)
                )
                async_tasks.append(trading_task)
            
            logger.info("🎉 All systems operational! Trading bot is now running...")
            
            # Keep the system running
            await asyncio.gather(*async_tasks)
            
        except Exception as e:
            logger.error(f"❌ Error in unified system: {e}")
            raise
        finally:
            self.is_running = False
            logger.info("🛑 EVIL2ROOT Trading System stopped")
    
    def stop(self):
        """Stop the trading system"""
        logger.info("🛑 Stopping EVIL2ROOT Trading System...")
        self.is_running = False
        
        # Stop all components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                logger.info(f"✅ Stopped {name}")
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")

def load_config(config_file: str = None) -> Dict[str, Any]:
    """Load configuration from file or environment"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
            'initial_balance': 100000,
            'enable_auto_learning': True,
            'enable_real_time_trading': True,
            'enable_telegram_bot': False,
            'train_models_on_start': True,
            'run_backtest_on_start': True,
            'scan_interval_seconds': 60,
            'enable_incremental_learning': True,
            'auto_learning_cycle_hours': 12,
            'risk_percentage': 0.02,
            'max_positions': 5,
            'auto_learning_config_path': 'src/config/auto_learning_config.json',
            'rl_training_data': 'market_data_cleaned_auto.csv',
            'backtest_data': 'market_data_cleaned_auto.csv'
        }
    
    # Override with environment variables
    for key in config:
        env_key = f"TRADING_BOT_{key.upper()}"
        if env_key in os.environ:
            env_value = os.environ[env_key]
            # Try to convert to appropriate type
            if isinstance(config[key], bool):
                config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(config[key], int):
                config[key] = int(env_value)
            elif isinstance(config[key], float):
                config[key] = float(env_value)
            elif isinstance(config[key], list):
                config[key] = env_value.split(',')
            else:
                config[key] = env_value
    
    return config

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    # This will be handled by the main loop
    pass

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='EVIL2ROOT Unified Trading System')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--mode', '-m', choices=['full', 'train', 'backtest', 'trade'], 
                       default='full', help='Execution mode')
    parser.add_argument('--symbols', '-s', nargs='+', help='Trading symbols')
    parser.add_argument('--balance', '-b', type=float, help='Initial balance')
    parser.add_argument('--no-training', action='store_true', help='Skip model training')
    parser.add_argument('--no-backtest', action='store_true', help='Skip backtesting')
    parser.add_argument('--dry-run', action='store_true', help='Run in simulation mode only')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.symbols:
        config['symbols'] = args.symbols
    if args.balance:
        config['initial_balance'] = args.balance
    if args.no_training:
        config['train_models_on_start'] = False
    if args.no_backtest:
        config['run_backtest_on_start'] = False
    if args.dry_run:
        config['enable_real_time_trading'] = False
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run the system
    system = UnifiedTradingSystem(config)
    
    try:
        if args.mode == 'train':
            system.train_models()
        elif args.mode == 'backtest':
            system.run_backtest()
        elif args.mode == 'trade':
            config['train_models_on_start'] = False
            config['run_backtest_on_start'] = False
            await system.run_unified_system()
        else:  # full mode
            await system.run_unified_system()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        system.stop()

if __name__ == "__main__":
    # Ensure proper event loop handling
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System terminated by user")
    except Exception as e:
        logger.error(f"System failed to start: {e}")
        sys.exit(1)
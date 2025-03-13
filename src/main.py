#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the EVIL2ROOT Trading Bot application.

This module initializes all components and starts the trading system.
It handles command line arguments to control the application's behavior.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

def setup_logging():
    """Configure logging for the application."""
    logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/trading_bot.log')
        ]
    )
    return logging.getLogger('main')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EVIL2ROOT Trading Bot')
    parser.add_argument('--mode', choices=['live', 'backtest', 'paper', 'analysis'], 
                        default='paper', help='Trading mode (default: paper)')
    parser.add_argument('--force-train', action='store_true', 
                        help='Force model training before starting')
    parser.add_argument('--symbols', type=str, 
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--config', type=str, default='config/config.py',
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    logger = setup_logging()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting EVIL2ROOT Trading Bot...")
    logger.info(f"Mode: {args.mode}")
    
    try:
        # Import here to avoid circular imports
        from src.core.trading import TradingSystem
        
        # Initialize the trading system
        trading_system = TradingSystem(
            mode=args.mode,
            force_train=args.force_train,
            symbols=args.symbols.split(',') if args.symbols else None,
            config_path=args.config
        )
        
        # Start the trading system
        trading_system.start()
        
    except Exception as e:
        logger.exception(f"Failed to start trading system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
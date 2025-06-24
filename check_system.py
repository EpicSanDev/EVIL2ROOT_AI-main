#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EVIL2ROOT Trading Bot - System Check Script

This script verifies that all components are properly configured
and ready to run.
"""

import os
import sys
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ Missing {description}: {dirpath}")
        return False

def check_secrets():
    """Check if all secret files exist"""
    print("\n🔐 Checking Secrets...")
    
    required_secrets = [
        "postgres_user.txt",
        "postgres_password.txt", 
        "postgres_db.txt",
        "db_user.txt",
        "db_password.txt",
        "secret_key.txt",
        "pgadmin_default_password.txt",
        "telegram_token.txt",
        "finnhub_api_key.txt",
        "openrouter_api_key.txt",
        "coinbase_api_key.txt",
        "coinbase_webhook_secret.txt"
    ]
    
    all_exist = True
    for secret in required_secrets:
        filepath = f"secrets/{secret}"
        if not check_file_exists(filepath, f"Secret {secret}"):
            all_exist = False
    
    return all_exist

def check_configuration():
    """Check configuration files"""
    print("\n⚙️ Checking Configuration...")
    
    all_exist = True
    
    # Check main config files
    configs = [
        (".env.production", "Production environment file"),
        ("config/bot_config.json", "Bot configuration"),
        ("requirements.txt", "Python requirements"),
        ("docker-compose.unified.yml", "Unified Docker Compose"),
        ("Dockerfile.unified", "Unified Dockerfile")
    ]
    
    for filepath, description in configs:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def check_scripts():
    """Check scripts and executables"""
    print("\n📜 Checking Scripts...")
    
    all_exist = True
    
    scripts = [
        ("unified_main.py", "Unified main entry point"),
        ("start_trading_bot.sh", "Startup script"),
        ("scripts/init_db.sql", "Database initialization script"),
        ("nginx/nginx.conf", "Nginx configuration")
    ]
    
    for filepath, description in scripts:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    # Check if scripts are executable
    if os.path.exists("start_trading_bot.sh"):
        if os.access("start_trading_bot.sh", os.X_OK):
            print("✅ start_trading_bot.sh is executable")
        else:
            print("❌ start_trading_bot.sh is not executable")
            all_exist = False
    
    return all_exist

def check_directories():
    """Check required directories"""
    print("\n📁 Checking Directories...")
    
    all_exist = True
    
    directories = [
        ("logs", "Logs directory"),
        ("saved_models", "Models directory"),
        ("data", "Data directory"),
        ("cache/market_data", "Market data cache"),
        ("secrets", "Secrets directory"),
        ("scripts", "Scripts directory"),
        ("nginx", "Nginx configuration directory"),
        ("app", "Main application directory"),
        ("src", "Source code directory"),
        ("frontend", "Frontend directory")
    ]
    
    for dirpath, description in directories:
        if not check_directory_exists(dirpath, description):
            all_exist = False
    
    return all_exist

def check_environment_variables():
    """Check if .env.production has the required variables"""
    print("\n🌍 Checking Environment Variables...")
    
    if not os.path.exists(".env.production"):
        print("❌ .env.production file not found")
        return False
    
    required_vars = [
        "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD",
        "REDIS_HOST", "REDIS_PORT",
        "SECRET_KEY", "FLASK_ENV",
        "TRADING_SYMBOLS", "INITIAL_BALANCE",
        "MODELS_DIR", "LOG_LEVEL"
    ]
    
    with open(".env.production", "r") as f:
        content = f.read()
    
    all_present = True
    for var in required_vars:
        if f"{var}=" in content:
            print(f"✅ {var} is configured")
        else:
            print(f"❌ Missing environment variable: {var}")
            all_present = False
    
    return all_present

def check_docker_files():
    """Check Docker-related files"""
    print("\n🐳 Checking Docker Files...")
    
    all_exist = True
    
    docker_files = [
        ("Dockerfile.unified", "Unified Dockerfile"),
        ("docker-compose.unified.yml", "Unified Docker Compose"),
        ("docker-compose.yml", "Original Docker Compose"),
        ("requirements.txt", "Python requirements"),
        ("requirements-essential.txt", "Essential requirements")
    ]
    
    for filepath, description in docker_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def main():
    """Main check function"""
    print("🤖 EVIL2ROOT Trading Bot - System Check")
    print("=" * 50)
    
    checks = [
        ("Directories", check_directories),
        ("Configuration Files", check_configuration),
        ("Secrets", check_secrets),
        ("Scripts", check_scripts),
        ("Environment Variables", check_environment_variables),
        ("Docker Files", check_docker_files)
    ]
    
    all_passed = True
    
    for check_name, check_function in checks:
        try:
            if not check_function():
                all_passed = False
        except Exception as e:
            print(f"❌ Error checking {check_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED! 🎉")
        print("✅ The EVIL2ROOT Trading Bot is ready to run!")
        print("\n🚀 To start the system:")
        print("   ./start_trading_bot.sh paper")
        print("\n📚 For more information, see PROJET_COMPLETE.md")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("⚠️  Please fix the issues above before running the system.")
        print("\n🔧 Quick fixes:")
        print("   ./start_trading_bot.sh setup  # Run initial setup")
        print("   chmod +x *.sh *.py           # Make scripts executable")
        return 1

if __name__ == "__main__":
    sys.exit(main())
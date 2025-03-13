#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'installation du système d'auto-apprentissage pour les modèles de trading.
Ce script crée les répertoires nécessaires, initialise la base de données du journal de trading,
et configure les planificateurs de tâches pour l'exécution automatique du système.
"""

import os
import sys
import json
import shutil
import sqlite3
import platform
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Ajout du chemin racine au sys.path pour permettre les imports relatifs
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

class SetupError(Exception):
    """Exception personnalisée pour les erreurs d'installation."""
    pass

def print_header(message):
    """Affiche un message d'en-tête formaté."""
    width = 80
    print("\n" + "=" * width)
    print(message.center(width))
    print("=" * width + "\n")

def print_step(message):
    """Affiche un message d'étape formaté."""
    print(f">> {message}...")

def print_success(message):
    """Affiche un message de succès formaté."""
    print(f"✓ {message}")

def print_error(message):
    """Affiche un message d'erreur formaté."""
    print(f"✗ ERROR: {message}")

def create_directories(base_dir, directories):
    """
    Crée les répertoires nécessaires pour le système.
    
    Args:
        base_dir (str): Répertoire de base
        directories (list): Liste des répertoires à créer
        
    Returns:
        bool: True si tous les répertoires ont été créés avec succès
    """
    print_step("Création des répertoires du système")
    
    success = True
    for directory in directories:
        dir_path = Path(base_dir) / directory
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  - Créé: {dir_path}")
        except Exception as e:
            print_error(f"Impossible de créer le répertoire {dir_path}: {e}")
            success = False
    
    if success:
        print_success("Tous les répertoires ont été créés avec succès")
    else:
        print_error("Certains répertoires n'ont pas pu être créés")
    
    return success

def init_database(db_path):
    """
    Initialise la base de données du journal de trading.
    
    Args:
        db_path (str): Chemin vers la base de données
        
    Returns:
        bool: True si la base de données a été initialisée avec succès
    """
    print_step(f"Initialisation de la base de données à {db_path}")
    
    db_path = Path(db_path)
    
    # Création du répertoire parent si nécessaire
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Si la base de données existe déjà, demander confirmation
    if db_path.exists():
        print(f"  - AVERTISSEMENT: La base de données {db_path} existe déjà.")
        response = input("    Voulez-vous la réinitialiser? Toutes les données seront perdues. (o/N): ")
        if response.lower() != 'o':
            print("  - Conservation de la base de données existante.")
            return True
        else:
            try:
                db_path.unlink()
                print("  - Base de données existante supprimée.")
            except Exception as e:
                print_error(f"Impossible de supprimer la base de données existante: {e}")
                return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Création de la table pour le journal de trading
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            position_size REAL NOT NULL,
            direction TEXT NOT NULL,
            pnl REAL NOT NULL,
            pnl_percent REAL NOT NULL,
            fee REAL,
            strategy_name TEXT,
            model_version TEXT,
            entry_signals TEXT,
            exit_signals TEXT,
            market_conditions TEXT,
            tags TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        # Création de la table pour les erreurs détectées
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            error_type TEXT NOT NULL,
            error_description TEXT,
            error_details TEXT,
            detected_at TEXT NOT NULL,
            FOREIGN KEY (trade_id) REFERENCES trades (id)
        )
        ''')
        
        # Création de la table pour les ajustements de modèles
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            old_version TEXT NOT NULL,
            new_version TEXT NOT NULL,
            adjustment_params TEXT NOT NULL,
            performance_before TEXT,
            performance_after TEXT,
            adjustment_reason TEXT,
            adjusted_at TEXT NOT NULL
        )
        ''')
        
        # Création des index pour améliorer les performances des requêtes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_direction ON trades (direction)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_type ON detected_errors (error_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_trade_id ON detected_errors (trade_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_adjustments_model ON model_adjustments (model_name)')
        
        conn.commit()
        conn.close()
        
        print_success(f"Base de données initialisée avec succès à {db_path}")
        return True
        
    except Exception as e:
        print_error(f"Erreur lors de l'initialisation de la base de données: {e}")
        return False

def copy_config_files(config_dir):
    """
    Copie les fichiers de configuration par défaut.
    
    Args:
        config_dir (str): Répertoire de configuration
        
    Returns:
        bool: True si les fichiers ont été copiés avec succès
    """
    print_step("Copie des fichiers de configuration par défaut")
    
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        "learning_frequency_days": 7,
        "analysis_window_days": 30,
        "error_analysis_window_days": 90,
        "min_trades_for_analysis": 10,
        "auto_adjust_enabled": True,
        "visualization_enabled": True,
        "backup_models_before_adjustment": True,
        "max_adjustment_per_cycle": 3,
        "adjustment_sensitivity": {
            "rl_models": 0.7,
            "ensemble_models": 0.5,
            "price_models": 0.8,
            "sentiment_models": 0.6
        },
        "error_thresholds": {
            "timing_error": 0.15,
            "size_error": 0.12,
            "trend_misread": 0.20,
            "signal_conflict": 0.10,
            "overtrading": 0.08,
            "premature_exit": 0.15,
            "delayed_exit": 0.15,
            "ignored_stop": 0.25,
            "market_condition": 0.18,
            "high_volatility": 0.20
        },
        "email_reports": {
            "enabled": False,
            "frequency_days": 7,
            "recipients": [],
            "include_visualizations": True,
            "include_performance_table": True,
            "include_error_analysis": True,
            "include_adjustment_summary": True
        },
        "logging": {
            "file_enabled": True,
            "console_enabled": True,
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_dir": "logs"
        },
        "model_data": {
            "save_format": "pickle",
            "version_control": True,
            "keep_previous_versions": 3,
            "backup_dir": "models/backups"
        },
        "database": {
            "connection_timeout": 30,
            "query_timeout": 60,
            "auto_vacuum": True,
            "optimization_frequency_days": 30
        }
    }
    
    config_file = config_dir / "auto_learning_config.json"
    
    # Vérifier si le fichier existe déjà
    if config_file.exists():
        print(f"  - Le fichier de configuration {config_file} existe déjà.")
        response = input("    Voulez-vous le remplacer par la configuration par défaut? (o/N): ")
        if response.lower() != 'o':
            print("  - Conservation du fichier de configuration existant.")
            return True
    
    try:
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print_success(f"Fichier de configuration créé avec succès: {config_file}")
        return True
        
    except Exception as e:
        print_error(f"Erreur lors de la création du fichier de configuration: {e}")
        return False

def setup_scheduler(script_path, frequency="weekly"):
    """
    Configure le planificateur de tâches du système d'exploitation pour exécuter
    le script d'auto-apprentissage périodiquement.
    
    Args:
        script_path (str): Chemin vers le script d'auto-apprentissage
        frequency (str): Fréquence d'exécution ("daily", "weekly", "monthly")
        
    Returns:
        bool: True si le planificateur a été configuré avec succès
    """
    print_step(f"Configuration du planificateur de tâches ({frequency})")
    
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        print_error(f"Le script {script_path} n'existe pas.")
        return False
    
    # Rendre le script exécutable sur Unix
    if platform.system() != "Windows":
        try:
            script_path.chmod(script_path.stat().st_mode | 0o111)
            print(f"  - Script rendu exécutable: {script_path}")
        except Exception as e:
            print_error(f"Impossible de rendre le script exécutable: {e}")
    
    # Configuration spécifique à la plateforme
    if platform.system() == "Windows":
        try:
            # Création d'un fichier batch pour l'exécution du script
            batch_path = script_path.parent / "run_auto_learning.bat"
            with open(batch_path, 'w') as f:
                f.write(f'@echo off\r\n')
                f.write(f'python "{script_path}" --log-level INFO\r\n')
                f.write(f'pause\r\n')
            
            # Création de la tâche planifiée avec schtasks
            task_name = "AutoLearningTrading"
            cmd = f'schtasks /create /tn "{task_name}" /tr "{batch_path}" /sc {frequency.upper()} /st 01:00'
            
            print(f"  - Commande d'installation de la tâche planifiée: {cmd}")
            print("  - Pour installer la tâche planifiée, exécutez la commande ci-dessus en tant qu'administrateur.")
            print("  - Ou utilisez le Planificateur de tâches Windows pour configurer manuellement la tâche.")
            
            # Ne pas exécuter automatiquement la commande schtasks car elle nécessite des privilèges administrateur
            print_success("Instructions pour la configuration de la tâche planifiée générées avec succès")
            return True
            
        except Exception as e:
            print_error(f"Erreur lors de la configuration du planificateur de tâches Windows: {e}")
            return False
            
    elif platform.system() == "Linux":
        try:
            # Création d'un fichier crontab
            cron_path = script_path.parent / "auto_learning.cron"
            
            # Définition de la programmation cron en fonction de la fréquence
            if frequency == "daily":
                cron_schedule = "0 1 * * *"  # Tous les jours à 01:00
            elif frequency == "weekly":
                cron_schedule = "0 1 * * 0"  # Tous les dimanches à 01:00
            elif frequency == "monthly":
                cron_schedule = "0 1 1 * *"  # Le 1er de chaque mois à 01:00
            else:
                cron_schedule = "0 1 * * 0"  # Par défaut: hebdomadaire
            
            with open(cron_path, 'w') as f:
                f.write(f'# Tâche automatique pour le système d\'auto-apprentissage de trading\n')
                f.write(f'{cron_schedule} cd {ROOT_DIR} && {sys.executable} {script_path} --log-level INFO\n')
            
            print(f"  - Fichier crontab créé: {cron_path}")
            print(f"  - Pour installer la tâche planifiée, exécutez:")
            print(f"    crontab {cron_path}")
            
            print_success("Instructions pour la configuration de cron générées avec succès")
            return True
            
        except Exception as e:
            print_error(f"Erreur lors de la configuration de cron: {e}")
            return False
            
    elif platform.system() == "Darwin":  # MacOS
        try:
            # Création d'un fichier plist pour launchd
            plist_dir = Path.home() / "Library" / "LaunchAgents"
            plist_dir.mkdir(parents=True, exist_ok=True)
            
            plist_path = plist_dir / "com.evil2root.autolearning.plist"
            
            # Définition de l'intervalle en fonction de la fréquence
            if frequency == "daily":
                interval = 86400  # 24 heures en secondes
            elif frequency == "weekly":
                interval = 604800  # 7 jours en secondes
            elif frequency == "monthly":
                interval = 2592000  # 30 jours en secondes
            else:
                interval = 604800  # Par défaut: hebdomadaire
            
            plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.evil2root.autolearning</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{script_path}</string>
        <string>--log-level</string>
        <string>INFO</string>
    </array>
    <key>StartInterval</key>
    <integer>{interval}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>{ROOT_DIR}/logs/auto_learning_error.log</string>
    <key>StandardOutPath</key>
    <string>{ROOT_DIR}/logs/auto_learning_output.log</string>
    <key>WorkingDirectory</key>
    <string>{ROOT_DIR}</string>
</dict>
</plist>
'''
            
            with open(plist_path, 'w') as f:
                f.write(plist_content)
            
            print(f"  - Fichier plist créé: {plist_path}")
            print(f"  - Pour installer la tâche planifiée, exécutez:")
            print(f"    launchctl load {plist_path}")
            
            print_success("Instructions pour la configuration de launchd générées avec succès")
            return True
            
        except Exception as e:
            print_error(f"Erreur lors de la configuration de launchd: {e}")
            return False
    
    else:
        print_error(f"Système d'exploitation non supporté: {platform.system()}")
        return False

def check_dependencies():
    """
    Vérifie que toutes les dépendances Python requises sont installées.
    
    Returns:
        bool: True si toutes les dépendances sont installées
    """
    print_step("Vérification des dépendances Python")
    
    required_packages = [
        "numpy", "pandas", "matplotlib", "seaborn", "scikit-learn",
        "tensorflow", "keras", "torch", "statsmodels", "plotly", "joblib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  - {package}: Installé")
        except ImportError:
            missing_packages.append(package)
            print(f"  - {package}: Non installé")
    
    if missing_packages:
        print_error(f"Les packages suivants sont manquants: {', '.join(missing_packages)}")
        print(f"  Pour les installer, exécutez:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    else:
        print_success("Toutes les dépendances sont installées")
        return True

def create_test_data(db_path, num_trades=50):
    """
    Crée des données de test pour démontrer le fonctionnement du système.
    
    Args:
        db_path (str): Chemin vers la base de données
        num_trades (int): Nombre de transactions à créer
        
    Returns:
        bool: True si les données ont été créées avec succès
    """
    print_step(f"Création de {num_trades} transactions de test")
    
    try:
        import random
        from datetime import datetime, timedelta
        import json
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Paramètres pour la génération de données
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT"]
        directions = ["BUY", "SELL"]
        strategies = ["trend_following_v1", "mean_reversion_v2", "breakout_v1", "sentiment_based_v1"]
        
        # Génération de transactions aléatoires
        for i in range(num_trades):
            # Choisir un symbole aléatoire
            symbol = random.choice(symbols)
            
            # Date d'entrée aléatoire dans les 180 derniers jours
            days_ago = random.randint(1, 180)
            entry_time = (datetime.now() - timedelta(days=days_ago, 
                                                            hours=random.randint(0, 23),
                                                            minutes=random.randint(0, 59))).isoformat()
            
            # Durée de la transaction entre 10 minutes et 5 jours
            trade_duration = random.randint(10, 7200)  # minutes
            exit_time = (datetime.fromisoformat(entry_time) + 
                        timedelta(minutes=trade_duration)).isoformat()
            
            # Direction aléatoire
            direction = random.choice(directions)
            
            # Prix d'entrée et de sortie
            if symbol == "BTC/USDT":
                base_price = random.uniform(25000, 60000)
            elif symbol == "ETH/USDT":
                base_price = random.uniform(1500, 4000)
            elif symbol == "XRP/USDT":
                base_price = random.uniform(0.4, 1.2)
            elif symbol == "SOL/USDT":
                base_price = random.uniform(50, 200)
            else:  # ADA
                base_price = random.uniform(0.3, 1.0)
            
            entry_price = base_price
            
            # Calculer le prix de sortie en fonction de la direction
            # et avec une tendance à avoir plus de transactions gagnantes que perdantes
            win = random.random() < 0.6  # 60% de chance de gagner
            
            if (direction == "BUY" and win) or (direction == "SELL" and not win):
                # Prix de sortie plus élevé pour un achat gagnant ou une vente perdante
                pct_change = random.uniform(0.5, 8.0)
                exit_price = entry_price * (1 + pct_change / 100)
            else:
                # Prix de sortie plus bas pour un achat perdant ou une vente gagnante
                pct_change = random.uniform(0.5, 5.0)
                exit_price = entry_price * (1 - pct_change / 100)
            
            # Taille de position
            position_size = random.uniform(0.1, 2.0)
            
            # Calcul du P&L
            if direction == "BUY":
                pnl = (exit_price - entry_price) * position_size
                pnl_percent = (exit_price / entry_price - 1) * 100
            else:  # SELL
                pnl = (entry_price - exit_price) * position_size
                pnl_percent = (entry_price / exit_price - 1) * 100
            
            # Frais
            fee = abs(entry_price * position_size * 0.001) + abs(exit_price * position_size * 0.001)
            
            # Stratégie aléatoire
            strategy_name = random.choice(strategies)
            model_version = f"1.{random.randint(0, 9)}.{random.randint(0, 9)}"
            
            # Signaux d'entrée/sortie
            entry_signals = json.dumps({
                "trend": random.choice(["UP", "DOWN", "NEUTRAL"]),
                "rsi": random.uniform(10, 90),
                "macd": random.choice(["BULLISH", "BEARISH", "NEUTRAL"]),
                "volume": random.uniform(100, 10000)
            })
            
            exit_signals = json.dumps({
                "take_profit": random.choice([True, False]),
                "stop_loss": random.choice([True, False]),
                "signal_reversal": random.choice([True, False]),
                "time_exit": random.choice([True, False])
            })
            
            # Conditions de marché
            market_conditions = json.dumps({
                "volatility": random.uniform(0.5, 30),
                "trend_strength": random.uniform(0, 1),
                "volume_profile": random.choice(["HIGH", "MEDIUM", "LOW"]),
                "market_regime": random.choice(["BULL", "BEAR", "SIDEWAYS"]),
                "correlation_btc": random.uniform(-1, 1)
            })
            
            # Dates de création et de mise à jour
            created_at = datetime.now().isoformat()
            updated_at = created_at
            
            # Insertion dans la base de données
            cursor.execute('''
            INSERT INTO trades (
                symbol, entry_time, exit_time, entry_price, exit_price, position_size,
                direction, pnl, pnl_percent, fee, strategy_name, model_version,
                entry_signals, exit_signals, market_conditions, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, entry_time, exit_time, entry_price, exit_price, position_size,
                direction, pnl, pnl_percent, fee, strategy_name, model_version,
                entry_signals, exit_signals, market_conditions, created_at, updated_at
            ))
        
        conn.commit()
        conn.close()
        
        print_success(f"{num_trades} transactions de test créées avec succès")
        return True
        
    except Exception as e:
        print_error(f"Erreur lors de la création des données de test: {e}")
        return False

def main():
    print_header("INSTALLATION DU SYSTÈME D'AUTO-APPRENTISSAGE POUR MODÈLES DE TRADING")
    print(f"Date d'installation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Système d'exploitation: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Répertoire racine: {ROOT_DIR}")
    
    # Parsing des arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Installation du système d\'auto-apprentissage')
    parser.add_argument('--config-dir', type=str, default=os.path.join(ROOT_DIR, 'src/config'),
                        help='Répertoire pour les fichiers de configuration')
    parser.add_argument('--db-path', type=str, default=os.path.join(ROOT_DIR, 'data/trade_journal.db'),
                        help='Chemin vers la base de données du journal de trading')
    parser.add_argument('--skip-dependencies', action='store_true',
                        help='Ignorer la vérification des dépendances')
    parser.add_argument('--skip-scheduler', action='store_true',
                        help='Ignorer la configuration du planificateur de tâches')
    parser.add_argument('--frequency', type=str, choices=['daily', 'weekly', 'monthly'],
                        default='weekly', help='Fréquence d\'exécution du planificateur')
    parser.add_argument('--create-test-data', action='store_true',
                        help='Créer des données de test')
    parser.add_argument('--test-data-count', type=int, default=50,
                        help='Nombre de transactions de test à créer')
    
    args = parser.parse_args()
    
    success = True
    
    # Création des répertoires nécessaires
    directories = [
        'data',
        'data/reports',
        'data/learning_history',
        'logs',
        'models',
        'models/backups',
    ]
    
    if not create_directories(ROOT_DIR, directories):
        success = False
    
    # Vérification des dépendances
    if not args.skip_dependencies and not check_dependencies():
        success = False
    
    # Initialisation de la base de données
    if not init_database(args.db_path):
        success = False
    
    # Copie des fichiers de configuration
    if not copy_config_files(args.config_dir):
        success = False
    
    # Configuration du planificateur de tâches
    scheduler_script = os.path.join(ROOT_DIR, 'src/scripts/auto_learning_scheduler.py')
    if not args.skip_scheduler and not setup_scheduler(scheduler_script, args.frequency):
        success = False
    
    # Création de données de test si demandé
    if args.create_test_data and not create_test_data(args.db_path, args.test_data_count):
        success = False
    
    # Résumé de l'installation
    if success:
        print_header("INSTALLATION TERMINÉE AVEC SUCCÈS")
        print("Le système d'auto-apprentissage est prêt à être utilisé.")
        print("\nPour exécuter manuellement un cycle d'apprentissage, utilisez:")
        print(f"python {scheduler_script} --force")
        print("\nPour plus d'informations, consultez:")
        print(f"{ROOT_DIR}/src/models/auto_learning/README.md")
    else:
        print_header("INSTALLATION TERMINÉE AVEC DES ERREURS")
        print("Certaines étapes de l'installation ont échoué. Veuillez vérifier les messages d'erreur ci-dessus.")
        print("Vous pouvez réexécuter l'installation après avoir résolu les problèmes.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 
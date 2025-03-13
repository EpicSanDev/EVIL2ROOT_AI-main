import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .trade_journal import TradeJournal
from .error_detector import ErrorDetector
from .performance_analyzer import PerformanceAnalyzer
from .model_adjuster import ModelAdjuster

class LearningOrchestrator:
    """
    Orchestrateur central du système d'auto-apprentissage des modèles de trading.
    Cette classe intègre tous les composants et gère le flux de travail d'apprentissage automatique.
    """
    
    def __init__(self, 
                config_path: Optional[str] = None,
                db_path: str = "data/trade_journal.db",
                models_dir: str = "saved_models",
                reports_dir: str = "data/performance_reports"):
        """
        Initialise l'orchestrateur d'apprentissage.
        
        Args:
            config_path: Chemin vers le fichier de configuration (facultatif)
            db_path: Chemin vers la base de données du journal de trading
            models_dir: Répertoire où sont stockés les modèles
            reports_dir: Répertoire où stocker les rapports générés
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Créer les répertoires nécessaires
        for directory in [os.path.dirname(db_path), models_dir, reports_dir]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialiser les composants
        self.trade_journal = TradeJournal(db_path=db_path)
        self.error_detector = ErrorDetector(trade_journal=self.trade_journal)
        self.performance_analyzer = PerformanceAnalyzer(trade_journal=self.trade_journal)
        self.model_adjuster = ModelAdjuster(
            trade_journal=self.trade_journal,
            error_detector=self.error_detector,
            performance_analyzer=self.performance_analyzer,
            models_dir=models_dir
        )
        
        self.logger.info("Orchestrateur d'apprentissage initialisé avec succès")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Charge la configuration depuis un fichier JSON.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Returns:
            Dictionnaire de configuration
        """
        default_config = {
            "learning_frequency_days": 7,
            "analysis_window_days": 30,
            "error_analysis_window_days": 90,
            "min_trades_for_analysis": 10,
            "auto_adjust_enabled": True,
            "visualization_enabled": True,
            "email_reports_enabled": False,
            "email_recipients": []
        }
        
        if not config_path or not os.path.exists(config_path):
            self.logger.info("Fichier de configuration non trouvé, utilisation des valeurs par défaut")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Fusionner avec les valeurs par défaut pour les clés manquantes
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                    
            self.logger.info(f"Configuration chargée depuis {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return default_config
    
    def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Exécute un cycle complet d'auto-apprentissage:
        1. Analyse les performances et erreurs
        2. Ajuste les modèles si nécessaire
        3. Génère des rapports
        
        Returns:
            Dictionnaire contenant les résultats du cycle d'apprentissage
        """
        self.logger.info("Démarrage d'un cycle d'auto-apprentissage")
        
        try:
            # 1. Analyser les performances
            analysis_window = self.config.get('analysis_window_days', 30)
            performance_data = self.performance_analyzer.analyze_recent_performance(days=analysis_window)
            
            # 2. Analyser les erreurs
            error_window = self.config.get('error_analysis_window_days', 90)
            error_analysis = self.error_detector.analyze_losing_trades(days=error_window)
            
            # 3. Générer des visualisations si activé
            visualization_paths = {}
            if self.config.get('visualization_enabled', True):
                visualization_paths = self.performance_analyzer.generate_performance_visualizations()
            
            # 4. Ajuster les modèles si nécessaire et si activé
            adjustment_results = {"status": "disabled"}
            if self.config.get('auto_adjust_enabled', True):
                adjustment_results = self.model_adjuster.analyze_and_adjust_models(days=analysis_window)
            
            # 5. Envoyer des rapports par email si configuré
            email_sent = False
            if self.config.get('email_reports_enabled', False) and self.config.get('email_recipients'):
                email_sent = self._send_email_report(
                    performance_data, 
                    error_analysis, 
                    adjustment_results, 
                    visualization_paths
                )
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'performance_analysis': performance_data,
                'error_analysis': {
                    'total_errors': len(error_analysis.get('detected_errors', [])),
                    'error_types': list(error_analysis.get('error_patterns', {}).keys())
                },
                'model_adjustments': {
                    'status': adjustment_results.get('status'),
                    'models_adjusted': len(adjustment_results.get('adjustments_made', []))
                },
                'visualizations_generated': bool(visualization_paths),
                'email_report_sent': email_sent
            }
            
            self.logger.info(f"Cycle d'apprentissage terminé avec succès: {results['model_adjustments']['models_adjusted']} modèles ajustés")
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors du cycle d'apprentissage: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error_message': str(e)
            }
    
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Enregistre une transaction dans le journal de trading.
        
        Args:
            trade_data: Données de la transaction
            
        Returns:
            ID de la transaction
        """
        return self.trade_journal.log_trade(trade_data)
    
    def analyze_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyse les performances de trading récentes.
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Résultats de l'analyse
        """
        return self.performance_analyzer.analyze_recent_performance(days=days)
    
    def detect_errors(self, days: int = 90) -> Dict[str, Any]:
        """
        Détecte les erreurs dans les transactions récentes.
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Résultats de l'analyse d'erreurs
        """
        return self.error_detector.analyze_losing_trades(days=days)
    
    def adjust_models(self, days: int = 30) -> Dict[str, Any]:
        """
        Ajuste les modèles en fonction des performances et erreurs récentes.
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Résultats des ajustements
        """
        return self.model_adjuster.analyze_and_adjust_models(days=days)
    
    def _send_email_report(self, 
                         performance_data: Dict[str, Any],
                         error_analysis: Dict[str, Any],
                         adjustment_results: Dict[str, Any],
                         visualization_paths: Dict[str, str]) -> bool:
        """
        Envoie un rapport par email aux destinataires configurés.
        
        Args:
            performance_data: Données d'analyse des performances
            error_analysis: Données d'analyse des erreurs
            adjustment_results: Résultats des ajustements des modèles
            visualization_paths: Chemins des fichiers de visualisation
            
        Returns:
            True si l'email a été envoyé avec succès, False sinon
        """
        try:
            # Cette implémentation est un placeholder
            # Dans une implémentation réelle, il faudrait utiliser une bibliothèque comme smtplib
            self.logger.info("Envoi d'un rapport par email (fonctionnalité non implémentée)")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi de l'email: {e}")
            return False 
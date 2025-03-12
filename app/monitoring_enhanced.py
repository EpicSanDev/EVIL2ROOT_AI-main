import time
import logging
import threading
import psutil
import os
import json
from datetime import datetime, timedelta
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from prometheus_client.registry import CollectorRegistry
import numpy as np

# Configuration du logger
logger = logging.getLogger(__name__)

class EnhancedMonitoringService:
    """
    Service de monitoring amélioré pour le trading bot EVIL2ROOT.
    
    Fournit:
    1. Métriques détaillées sur les performances des modèles d'IA
    2. Métriques systèmes (CPU, mémoire, disque, réseau)
    3. Métriques métier avancées (PnL, drawdown, ratio de Sharpe, etc.)
    4. Métriques de calibration et de qualité de prédiction
    5. Export des données de monitoring vers JSON pour analyses externes
    """
    
    def __init__(self, port=8000, export_directory="logs/metrics", model_metrics_interval=300):
        """
        Initialise le service de monitoring amélioré.
        
        Args:
            port: Port sur lequel exposer les métriques Prometheus
            export_directory: Répertoire pour exporter les métriques en JSON
            model_metrics_interval: Intervalle en secondes pour collecter les métriques des modèles
        """
        self.port = port
        self.export_directory = export_directory
        self.model_metrics_interval = model_metrics_interval
        self.server_started = False
        self.server_thread = None
        self.running = False
        self.metrics_collection_thread = None
        self.last_metrics_export = datetime.now()
        
        # Création du répertoire d'export si nécessaire
        os.makedirs(export_directory, exist_ok=True)
        
        # Registre Prometheus dédié
        self.registry = CollectorRegistry()
        
        # Initialisation des différentes catégories de métriques
        self._init_trading_metrics()
        self._init_model_performance_metrics()
        self._init_system_metrics()
        self._init_api_metrics()
        self._init_database_metrics()
        
        logger.info(f"Service de monitoring amélioré initialisé (port: {port})")
    
    def _init_trading_metrics(self):
        """Initialise les métriques liées au trading."""
        # Compteurs de trading
        self.trading_signals = Counter(
            'trading_signals_total', 
            'Nombre total de signaux de trading générés',
            ['symbol', 'direction', 'confidence_level', 'model_type'],
            registry=self.registry
        )
        
        self.executed_trades = Counter(
            'executed_trades_total', 
            'Nombre total de trades exécutés',
            ['symbol', 'direction', 'success'],
            registry=self.registry
        )
        
        # Jauges de performance
        self.open_positions = Gauge(
            'open_positions', 
            'Nombre de positions ouvertes',
            ['symbol', 'direction'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value', 
            'Valeur actuelle du portefeuille',
            registry=self.registry
        )
        
        self.balance = Gauge(
            'balance', 
            'Solde du compte',
            registry=self.registry
        )
        
        # Métriques de performance avancées
        self.win_rate = Gauge(
            'win_rate', 
            'Taux de trades gagnants',
            ['timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.profit_factor = Gauge(
            'profit_factor', 
            'Facteur de profit (gains/pertes)',
            ['timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.average_win = Gauge(
            'average_win', 
            'Gain moyen par trade gagnant',
            ['timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.average_loss = Gauge(
            'average_loss', 
            'Perte moyenne par trade perdant',
            ['timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'max_drawdown', 
            'Drawdown maximum',
            ['timeframe'],
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'sharpe_ratio', 
            'Ratio de Sharpe',
            ['timeframe', 'window'],
            registry=self.registry
        )
        
        self.sortino_ratio = Gauge(
            'sortino_ratio', 
            'Ratio de Sortino',
            ['timeframe', 'window'],
            registry=self.registry
        )
        
        self.calmar_ratio = Gauge(
            'calmar_ratio', 
            'Ratio de Calmar',
            ['timeframe'],
            registry=self.registry
        )
    
    def _init_model_performance_metrics(self):
        """Initialise les métriques liées aux performances des modèles d'IA."""
        # Précision et rappel des modèles
        self.model_accuracy = Gauge(
            'model_accuracy', 
            'Précision du modèle',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'model_precision', 
            'Précision des prédictions positives',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'model_recall', 
            'Rappel des prédictions positives',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'model_f1_score', 
            'Score F1 du modèle',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        # Métriques de régression
        self.model_rmse = Gauge(
            'model_rmse', 
            'Erreur quadratique moyenne',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.model_mae = Gauge(
            'model_mae', 
            'Erreur absolue moyenne',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        # Métriques de calibration
        self.calibration_error = Gauge(
            'calibration_error', 
            'Erreur de calibration (ECE)',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.calibration_reliability = Gauge(
            'calibration_reliability', 
            'Fiabilité de la calibration',
            ['model_name', 'confidence_bin', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        # Métriques d'apprentissage en ligne
        self.drift_detection = Gauge(
            'drift_detection', 
            'Détection de drift conceptuel',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
        
        self.memory_buffer_size = Gauge(
            'memory_buffer_size', 
            'Taille du buffer de mémoire pour l\'apprentissage continu',
            ['model_name', 'symbol'],
            registry=self.registry
        )
        
        self.online_learning_loss = Gauge(
            'online_learning_loss', 
            'Perte lors de l\'apprentissage en ligne',
            ['model_name', 'timeframe', 'symbol'],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialise les métriques système."""
        # CPU & Mémoire
        self.cpu_usage = Gauge(
            'cpu_usage_percent', 
            'Utilisation CPU en pourcentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes', 
            'Utilisation de la mémoire en bytes',
            registry=self.registry
        )
        
        self.memory_percent = Gauge(
            'memory_usage_percent', 
            'Utilisation de la mémoire en pourcentage',
            registry=self.registry
        )
        
        # Disque
        self.disk_usage = Gauge(
            'disk_usage_bytes', 
            'Utilisation du disque en bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        self.disk_percent = Gauge(
            'disk_usage_percent', 
            'Utilisation du disque en pourcentage',
            ['mount_point'],
            registry=self.registry
        )
        
        # Réseau
        self.network_sent = Gauge(
            'network_sent_bytes', 
            'Données réseau envoyées en bytes',
            registry=self.registry
        )
        
        self.network_received = Gauge(
            'network_received_bytes', 
            'Données réseau reçues en bytes',
            registry=self.registry
        )
        
        # Performance du système
        self.process_count = Gauge(
            'process_count', 
            'Nombre de processus en cours d\'exécution',
            registry=self.registry
        )
        
        self.system_load = Gauge(
            'system_load', 
            'Charge système (1, 5, 15 minutes)',
            ['interval'],
            registry=self.registry
        )
        
        # Métriques temporelles
        self.uptime = Gauge(
            'uptime_seconds', 
            'Temps d\'exécution en secondes',
            registry=self.registry
        )
        
        self.start_time = time.time()
    
    def _init_api_metrics(self):
        """Initialise les métriques liées aux API."""
        # Compteurs API
        self.api_requests = Counter(
            'api_requests_total', 
            'Nombre total de requêtes API',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        # Histogrammes pour les temps de réponse
        self.api_latency = Histogram(
            'api_latency_seconds', 
            'Latence des requêtes API en secondes',
            ['endpoint', 'method'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30),
            registry=self.registry
        )
        
        # Quota d'API
        self.api_quota_remaining = Gauge(
            'api_quota_remaining', 
            'Quota d\'API restant',
            ['provider'],
            registry=self.registry
        )
        
        self.api_errors = Counter(
            'api_errors_total', 
            'Nombre total d\'erreurs API',
            ['endpoint', 'error_type'],
            registry=self.registry
        )
    
    def _init_database_metrics(self):
        """Initialise les métriques liées à la base de données."""
        # Métriques de base de données
        self.db_query_duration = Histogram(
            'db_query_duration_seconds', 
            'Durée des requêtes de base de données en secondes',
            ['query_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5),
            registry=self.registry
        )
        
        self.db_connection_pool_size = Gauge(
            'db_connection_pool_size', 
            'Taille du pool de connexions à la base de données',
            registry=self.registry
        )
        
        self.db_connections_active = Gauge(
            'db_connections_active', 
            'Nombre de connexions actives à la base de données',
            registry=self.registry
        )
        
        self.db_operations = Counter(
            'db_operations_total', 
            'Nombre total d\'opérations de base de données',
            ['operation_type'],
            registry=self.registry
        )

    def start_server(self):
        """Démarre le serveur Prometheus sur le port configuré."""
        if self.server_started:
            logger.warning("Le serveur de métriques est déjà démarré")
            return
        
        def server_thread_func():
            logger.info(f"Démarrage du serveur de métriques sur le port {self.port}")
            start_http_server(self.port, registry=self.registry)
            
        self.server_thread = threading.Thread(target=server_thread_func, daemon=True)
        self.server_thread.start()
        self.server_started = True
        
        # Démarrer la collecte périodique des métriques
        self.start_metrics_collection()
        
        logger.info(f"Serveur de métriques démarré sur le port {self.port}")
    
    def stop_server(self):
        """Arrête le serveur de métriques et la collecte périodique."""
        if not self.server_started:
            logger.warning("Le serveur de métriques n'est pas démarré")
            return
        
        # Arrêter la collecte de métriques
        self.running = False
        if self.metrics_collection_thread:
            self.metrics_collection_thread.join(timeout=5)
            
        # Note: Prometheus ne fournit pas de méthode pour arrêter proprement le serveur
        # Nous arrêtons seulement la collecte de métriques
        
        self.server_started = False
        logger.info("Serveur de métriques arrêté")
    
    def start_metrics_collection(self):
        """Démarre la collecte périodique des métriques système."""
        if self.running:
            logger.warning("La collecte de métriques est déjà active")
            return
            
        self.running = True
        
        def metrics_collection_loop():
            last_model_metrics_time = time.time() - self.model_metrics_interval  # Pour forcer une première collecte
            
            while self.running:
                try:
                    # Mise à jour des métriques système
                    self._update_system_metrics()
                    
                    # Mise à jour des métriques des modèles (moins fréquente)
                    current_time = time.time()
                    if current_time - last_model_metrics_time >= self.model_metrics_interval:
                        self._update_model_metrics()
                        last_model_metrics_time = current_time
                    
                    # Export périodique des métriques en JSON
                    if (datetime.now() - self.last_metrics_export).total_seconds() >= 3600:  # Toutes les heures
                        self.export_metrics_to_json()
                        self.last_metrics_export = datetime.now()
                    
                    # Attendre avant la prochaine mise à jour
                    time.sleep(10)  # Collecter toutes les 10 secondes
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la collecte des métriques: {str(e)}")
                    time.sleep(30)  # Attendre plus longtemps en cas d'erreur
        
        self.metrics_collection_thread = threading.Thread(target=metrics_collection_loop, daemon=True)
        self.metrics_collection_thread.start()
        
        logger.info("Collecte périodique des métriques démarrée")
    
    def _update_system_metrics(self):
        """Met à jour les métriques système."""
        try:
            # CPU usage
            self.cpu_usage.set(psutil.cpu_percent(interval=None))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_percent.set(memory.percent)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.disk_usage.labels(mount_point=partition.mountpoint).set(usage.used)
                    self.disk_percent.labels(mount_point=partition.mountpoint).set(usage.percent)
                except (PermissionError, FileNotFoundError):
                    # Ignorer les partitions non accessibles
                    pass
            
            # Network usage
            network = psutil.net_io_counters()
            self.network_sent.set(network.bytes_sent)
            self.network_received.set(network.bytes_recv)
            
            # System load
            load1, load5, load15 = psutil.getloadavg()
            self.system_load.labels(interval="1min").set(load1)
            self.system_load.labels(interval="5min").set(load5)
            self.system_load.labels(interval="15min").set(load15)
            
            # Process count
            self.process_count.set(len(psutil.pids()))
            
            # Uptime
            self.uptime.set(time.time() - self.start_time)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques système: {str(e)}")
    
    def _update_model_metrics(self):
        """
        Met à jour les métriques des modèles.
        Cette méthode devrait être implémentée pour collecter les métriques spécifiques
        aux différents modèles d'IA en fonction de la structure du projet.
        """
        try:
            # Placeholder pour l'implémentation réelle
            # Il faudrait ici récupérer les métriques des différents modèles
            # et mettre à jour les jauges correspondantes
            
            # Par exemple, pour simuler des métriques de test:
            """
            self.model_accuracy.labels(model_name="price_prediction", timeframe="1h", symbol="BTC/USD").set(0.82)
            self.model_precision.labels(model_name="price_prediction", timeframe="1h", symbol="BTC/USD").set(0.78)
            self.model_recall.labels(model_name="price_prediction", timeframe="1h", symbol="BTC/USD").set(0.85)
            """
            
            logger.debug("Mise à jour des métriques des modèles effectuée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques des modèles: {str(e)}")
    
    def export_metrics_to_json(self):
        """Exporte les métriques collectées au format JSON."""
        try:
            # Créer une structure de données pour les métriques
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "disk_usage": {
                        partition.mountpoint: psutil.disk_usage(partition.mountpoint).percent
                        for partition in psutil.disk_partitions()
                        if not partition.mountpoint.startswith('/dev') and not partition.mountpoint.startswith('/sys')
                    }
                },
                "trading": {
                    # Ces données devraient venir du système de trading réel
                    "portfolio_value": 0,  # placeholder
                    "open_positions": 0,   # placeholder
                    "win_rate": 0          # placeholder
                },
                "models": {
                    # Ces données devraient venir des évaluations des modèles
                    "accuracy": {},        # placeholder
                    "drift_detection": {}  # placeholder
                }
            }
            
            # Créer le nom de fichier basé sur la date
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.export_directory, filename)
            
            # Écrire dans le fichier
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"Métriques exportées vers {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export des métriques en JSON: {str(e)}")

    # Méthodes pour enregistrer des événements de trading
    def record_trading_signal(self, symbol, direction, confidence_level, model_type="default"):
        """
        Enregistre un signal de trading généré.
        
        Args:
            symbol: Symbole de trading (ex: BTC/USD)
            direction: Direction du trade ('buy' ou 'sell')
            confidence_level: Niveau de confiance du signal (0.0-1.0)
            model_type: Type de modèle qui a généré le signal
        """
        try:
            self.trading_signals.labels(
                symbol=symbol,
                direction=direction,
                confidence_level=str(round(confidence_level, 2)),
                model_type=model_type
            ).inc()
            
            logger.debug(f"Signal de trading enregistré: {symbol} {direction} (conf: {confidence_level})")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du signal de trading: {str(e)}")
    
    def record_executed_trade(self, symbol, direction, success=True):
        """
        Enregistre un trade exécuté.
        
        Args:
            symbol: Symbole de trading (ex: BTC/USD)
            direction: Direction du trade ('buy' ou 'sell')
            success: Si le trade a été exécuté avec succès
        """
        try:
            self.executed_trades.labels(
                symbol=symbol,
                direction=direction,
                success=str(success)
            ).inc()
            
            logger.debug(f"Trade exécuté enregistré: {symbol} {direction} (succès: {success})")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du trade exécuté: {str(e)}")
    
    def update_portfolio_metrics(self, portfolio_value, balance, open_positions):
        """
        Met à jour les métriques du portefeuille.
        
        Args:
            portfolio_value: Valeur totale du portefeuille
            balance: Solde disponible
            open_positions: Dictionnaire de positions ouvertes {symbol: {direction: count}}
        """
        try:
            self.portfolio_value.set(portfolio_value)
            self.balance.set(balance)
            
            # Réinitialiser les positions ouvertes
            for symbol, positions in open_positions.items():
                for direction, count in positions.items():
                    self.open_positions.labels(symbol=symbol, direction=direction).set(count)
            
            logger.debug(f"Métriques de portefeuille mises à jour: val={portfolio_value}, bal={balance}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques de portefeuille: {str(e)}")
    
    def update_performance_metrics(self, symbol, timeframe, win_rate, profit_factor, avg_win, avg_loss):
        """
        Met à jour les métriques de performance de trading.
        
        Args:
            symbol: Symbole de trading (ex: BTC/USD)
            timeframe: Timeframe (ex: '1h', '1d')
            win_rate: Taux de trades gagnants (0.0-1.0)
            profit_factor: Facteur de profit (gains / pertes)
            avg_win: Gain moyen par trade gagnant
            avg_loss: Perte moyenne par trade perdant (valeur positive)
        """
        try:
            self.win_rate.labels(timeframe=timeframe, symbol=symbol).set(win_rate)
            self.profit_factor.labels(timeframe=timeframe, symbol=symbol).set(profit_factor)
            self.average_win.labels(timeframe=timeframe, symbol=symbol).set(avg_win)
            self.average_loss.labels(timeframe=timeframe, symbol=symbol).set(avg_loss)
            
            logger.debug(f"Métriques de performance mises à jour pour {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques de performance: {str(e)}")
    
    def update_risk_metrics(self, timeframe, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, window="30d"):
        """
        Met à jour les métriques de risque.
        
        Args:
            timeframe: Timeframe (ex: '1h', '1d')
            max_drawdown: Drawdown maximum (en pourcentage)
            sharpe_ratio: Ratio de Sharpe
            sortino_ratio: Ratio de Sortino
            calmar_ratio: Ratio de Calmar
            window: Fenêtre temporelle pour les ratios (ex: '30d', '365d')
        """
        try:
            self.max_drawdown.labels(timeframe=timeframe).set(max_drawdown)
            self.sharpe_ratio.labels(timeframe=timeframe, window=window).set(sharpe_ratio)
            self.sortino_ratio.labels(timeframe=timeframe, window=window).set(sortino_ratio)
            self.calmar_ratio.labels(timeframe=timeframe).set(calmar_ratio)
            
            logger.debug(f"Métriques de risque mises à jour pour {timeframe}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques de risque: {str(e)}")
    
    # Méthodes pour enregistrer des métriques de modèle
    def update_model_performance(self, model_name, timeframe, symbol, metrics):
        """
        Met à jour les métriques de performance d'un modèle.
        
        Args:
            model_name: Nom du modèle (ex: 'price_prediction', 'risk_management')
            timeframe: Timeframe (ex: '1h', '1d')
            symbol: Symbole de trading (ex: BTC/USD)
            metrics: Dictionnaire de métriques {metric_name: value}
                Métriques supportées: 'accuracy', 'precision', 'recall', 'f1', 'rmse', 'mae'
        """
        try:
            # Classification metrics
            if 'accuracy' in metrics:
                self.model_accuracy.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(metrics['accuracy'])
            
            if 'precision' in metrics:
                self.model_precision.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(metrics['precision'])
            
            if 'recall' in metrics:
                self.model_recall.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(metrics['recall'])
            
            if 'f1' in metrics:
                self.model_f1_score.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(metrics['f1'])
            
            # Regression metrics
            if 'rmse' in metrics:
                self.model_rmse.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(metrics['rmse'])
            
            if 'mae' in metrics:
                self.model_mae.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(metrics['mae'])
                
            logger.debug(f"Métriques de modèle {model_name} mises à jour pour {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques de modèle: {str(e)}")
    
    def update_calibration_metrics(self, model_name, timeframe, symbol, ece, reliability_bins=None):
        """
        Met à jour les métriques de calibration d'un modèle.
        
        Args:
            model_name: Nom du modèle (ex: 'price_prediction', 'risk_management')
            timeframe: Timeframe (ex: '1h', '1d')
            symbol: Symbole de trading (ex: BTC/USD)
            ece: Expected Calibration Error
            reliability_bins: Dictionnaire de fiabilité par bin {bin: reliability}
        """
        try:
            self.calibration_error.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(ece)
            
            if reliability_bins:
                for confidence_bin, reliability in reliability_bins.items():
                    self.calibration_reliability.labels(
                        model_name=model_name, 
                        confidence_bin=str(confidence_bin),
                        timeframe=timeframe, 
                        symbol=symbol
                    ).set(reliability)
                    
            logger.debug(f"Métriques de calibration du modèle {model_name} mises à jour pour {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques de calibration: {str(e)}")
    
    def update_online_learning_metrics(self, model_name, timeframe, symbol, drift_score, memory_size, loss):
        """
        Met à jour les métriques d'apprentissage en ligne d'un modèle.
        
        Args:
            model_name: Nom du modèle (ex: 'price_prediction', 'risk_management')
            timeframe: Timeframe (ex: '1h', '1d')
            symbol: Symbole de trading (ex: BTC/USD)
            drift_score: Score de détection de drift (0.0-1.0)
            memory_size: Taille du buffer de mémoire pour l'apprentissage continu
            loss: Perte lors du dernier apprentissage
        """
        try:
            self.drift_detection.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(drift_score)
            self.memory_buffer_size.labels(model_name=model_name, symbol=symbol).set(memory_size)
            self.online_learning_loss.labels(model_name=model_name, timeframe=timeframe, symbol=symbol).set(loss)
            
            logger.debug(f"Métriques d'apprentissage en ligne pour {model_name} mises à jour")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques d'apprentissage en ligne: {str(e)}")
    
    # Méthodes pour enregistrer des métriques d'API
    def record_api_request(self, endpoint, method, status):
        """
        Enregistre une requête API.
        
        Args:
            endpoint: Endpoint de l'API
            method: Méthode HTTP
            status: Code de statut HTTP
        """
        try:
            self.api_requests.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la requête API: {str(e)}")
    
    def record_api_latency(self, endpoint, method, latency_seconds):
        """
        Enregistre la latence d'une requête API.
        
        Args:
            endpoint: Endpoint de l'API
            method: Méthode HTTP
            latency_seconds: Latence en secondes
        """
        try:
            self.api_latency.labels(endpoint=endpoint, method=method).observe(latency_seconds)
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la latence API: {str(e)}")
    
    def update_api_quota(self, provider, remaining):
        """
        Met à jour le quota d'API restant.
        
        Args:
            provider: Fournisseur d'API
            remaining: Quota restant
        """
        try:
            self.api_quota_remaining.labels(provider=provider).set(remaining)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du quota d'API: {str(e)}")
    
    def record_api_error(self, endpoint, error_type):
        """
        Enregistre une erreur d'API.
        
        Args:
            endpoint: Endpoint de l'API
            error_type: Type d'erreur
        """
        try:
            self.api_errors.labels(endpoint=endpoint, error_type=error_type).inc()
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'erreur API: {str(e)}")

    # Méthodes pour enregistrer des métriques de base de données
    def record_db_query(self, query_type, duration_seconds):
        """
        Enregistre la durée d'une requête de base de données.
        
        Args:
            query_type: Type de requête (ex: 'select', 'insert', 'update')
            duration_seconds: Durée en secondes
        """
        try:
            self.db_query_duration.labels(query_type=query_type).observe(duration_seconds)
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la durée de requête DB: {str(e)}")
    
    def update_db_connection_stats(self, pool_size, active_connections):
        """
        Met à jour les statistiques de connexion à la base de données.
        
        Args:
            pool_size: Taille du pool de connexions
            active_connections: Nombre de connexions actives
        """
        try:
            self.db_connection_pool_size.set(pool_size)
            self.db_connections_active.set(active_connections)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des statistiques de connexion DB: {str(e)}")
    
    def record_db_operation(self, operation_type):
        """
        Enregistre une opération de base de données.
        
        Args:
            operation_type: Type d'opération (ex: 'select', 'insert', 'update', 'delete')
        """
        try:
            self.db_operations.labels(operation_type=operation_type).inc()
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'opération DB: {str(e)}")
    
    # Contextes pour mesurer les performances
    def api_request_context(self, endpoint, method):
        """
        Contexte pour mesurer la latence d'une requête API.
        
        Exemple d'utilisation:
        ```
        with monitoring_service.api_request_context('/api/data', 'GET') as status_code:
            status_code.value = 200  # Mettre à jour le code de statut
        ```
        
        Args:
            endpoint: Endpoint de l'API
            method: Méthode HTTP
            
        Returns:
            Un objet de contexte avec un attribut value pour le code de statut
        """
        class StatusCode:
            def __init__(self):
                self.value = 200
        
        status = StatusCode()
        start_time = time.time()
        
        class APIRequestContext:
            def __enter__(self):
                return status
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - start_time
                self.record_api_latency(endpoint, method, duration)
                self.record_api_request(endpoint, method, status.value)
                
                if exc_type is not None:
                    # Une exception s'est produite
                    error_type = exc_type.__name__
                    self.record_api_error(endpoint, error_type)
                    logger.error(f"Exception pendant la requête API {endpoint}: {error_type}")
        
        return APIRequestContext()
    
    def db_query_context(self, query_type):
        """
        Contexte pour mesurer la durée d'une requête de base de données.
        
        Exemple d'utilisation:
        ```
        with monitoring_service.db_query_context('select'):
            # Exécuter la requête
        ```
        
        Args:
            query_type: Type de requête
            
        Returns:
            Un objet de contexte
        """
        start_time = time.time()
        
        class DBQueryContext:
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - start_time
                self.record_db_query(query_type, duration)
                self.record_db_operation(query_type)
                
                if exc_type is not None:
                    # Une exception s'est produite
                    logger.error(f"Exception pendant la requête DB {query_type}: {exc_type.__name__}")
        
        return DBQueryContext()

# Singleton pour le service de monitoring
_monitoring_service_instance = None

def init_enhanced_monitoring(port=8000, export_directory="logs/metrics", model_metrics_interval=300, 
                           with_flask=False, flask_app=None):
    """
    Initialise le service de monitoring amélioré.
    
    Args:
        port: Port sur lequel exposer les métriques Prometheus
        export_directory: Répertoire pour exporter les métriques
        model_metrics_interval: Intervalle pour la collecte des métriques des modèles
        with_flask: Si True, intègre le monitoring à une application Flask
        flask_app: Application Flask à intégrer
        
    Returns:
        Instance du service de monitoring
    """
    global _monitoring_service_instance
    
    if _monitoring_service_instance is None:
        _monitoring_service_instance = EnhancedMonitoringService(
            port=port,
            export_directory=export_directory,
            model_metrics_interval=model_metrics_interval
        )
        
        # Démarrer le serveur de métriques
        _monitoring_service_instance.start_server()
        
        # Intégration Flask si nécessaire
        if with_flask and flask_app:
            from flask import request, g
            import time
            
            @flask_app.before_request
            def before_request():
                g.start_time = time.time()
                
            @flask_app.after_request
            def after_request(response):
                if hasattr(g, 'start_time'):
                    duration = time.time() - g.start_time
                    endpoint = request.endpoint or 'unknown'
                    method = request.method
                    status = response.status_code
                    
                    _monitoring_service_instance.record_api_latency(endpoint, method, duration)
                    _monitoring_service_instance.record_api_request(endpoint, method, status)
                    
                return response
            
        logger.info("Service de monitoring amélioré initialisé et démarré")
    
    return _monitoring_service_instance

def get_monitoring_service():
    """
    Récupère l'instance du service de monitoring amélioré.
    
    Returns:
        Instance du service de monitoring, ou None si non initialisé
    """
    return _monitoring_service_instance

# Supprimer le commentaire obsolète
"""
Module de monitoring amélioré pour le trading bot EVIL2ROOT.
Ce module offre des métriques détaillées sur les performances des modèles,
l'utilisation des ressources système, et les métriques métier essentielles.
""" 
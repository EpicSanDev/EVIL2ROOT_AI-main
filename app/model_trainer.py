import logging
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
import joblib  # For saving and loading models
from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import concurrent.futures

# Fonction de vérification du GPU améliorée
def check_gpu_availability():
    """
    Vérifie si le GPU est disponible et renvoie les informations de configuration.
    Cette fonction implémente plusieurs méthodes de détection pour plus de robustesse.
    """
    gpu_info = {
        'available': False,
        'devices': [],
        'memory_details': [],
        'detection_method': None
    }
    
    # Méthode 1: Vérification via TensorFlow
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info['available'] = True
            gpu_info['detection_method'] = 'tensorflow'
            
            for gpu in gpus:
                gpu_info['devices'].append(str(gpu))
                
                # Configurer la croissance mémoire pour éviter de monopoliser toute la VRAM
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    gpu_info['memory_growth_enabled'] = True
                except RuntimeError as e:
                    gpu_info['memory_growth_enabled'] = False
                    gpu_info['memory_growth_error'] = str(e)
    except Exception as e:
        gpu_info['tf_detection_error'] = str(e)
    
    # Méthode 2: Vérification via nvidia-smi (si TensorFlow n'a pas détecté de GPU)
    if not gpu_info['available']:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if result.returncode == 0 and result.stdout:
                gpu_info['available'] = True
                gpu_info['detection_method'] = 'nvidia-smi'
                gpus_detected = result.stdout.decode('utf-8').strip().split('\n')
                gpu_info['devices'] = gpus_detected
        except Exception as e:
            gpu_info['smi_detection_error'] = str(e)
    
    # Méthode 3: Vérification via CUDA_VISIBLE_DEVICES
    if not gpu_info['available'] and os.environ.get('CUDA_VISIBLE_DEVICES', None) not in [None, '', '-1']:
        gpu_info['available'] = True
        gpu_info['detection_method'] = 'environment_variable'
        gpu_info['devices'] = [f"GPU from CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"]
    
    # Si GPU détecté, essayer d'obtenir plus d'informations via nvidia-smi
    if gpu_info['available'] and gpu_info['detection_method'] in ['tensorflow', 'nvidia-smi']:
        try:
            import subprocess
            gpu_memory_info = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,nounits,noheader'],
                timeout=5
            ).decode('utf-8').strip().split('\n')
            
            for info in gpu_memory_info:
                if ',' in info:  # S'assurer que le format est correct
                    total, free = info.split(',')
                    gpu_info['memory_details'].append({
                        'total_mb': int(total.strip()),
                        'free_mb': int(free.strip()),
                        'used_mb': int(total.strip()) - int(free.strip())
                    })
        except Exception:
            # C'est OK si cette commande échoue, elle est juste informative
            pass
    
    return gpu_info

def configure_gpu_environment(gpu_info):
    """
    Configure l'environnement d'exécution en fonction de la disponibilité du GPU.
    Retourne True si le GPU est correctement configuré et utilisable, False sinon.
    """
    if not gpu_info['available']:
        # Désactiver l'utilisation du GPU en définissant CUDA_VISIBLE_DEVICES à -1
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        return False
    
    # GPU disponible, configurer pour une utilisation optimale
    
    # Activer la croissance mémoire dynamique pour éviter d'allouer toute la VRAM immédiatement
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Configuration spécifique pour TensorFlow
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass  # Ignorer les erreurs car cette étape peut être redondante avec check_gpu_availability
    
    # Configuration pour limiter la mémoire utilisée si spécifiée
    memory_limit_mb = os.environ.get('GPU_MEMORY_LIMIT_MB', None)
    if memory_limit_mb and memory_limit_mb.isdigit():
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(memory_limit_mb))]
                )
        except Exception:
            pass
    
    return True

# Try to import TFKerasPruningCallback, but provide a fallback if it's not available
try:
    from optuna.integration import TFKerasPruningCallback
    OPTUNA_TFKERAS_AVAILABLE = True
except ImportError:
    logging.warning("optuna-integration[tfkeras] not found. TFKerasPruningCallback will not be available.")
    # Define a dummy callback that does nothing as fallback
    class DummyTFKerasPruningCallback(tf.keras.callbacks.Callback):
        def __init__(self, *args, **kwargs):
            super().__init__()
            logging.warning("Using dummy TFKerasPruningCallback. Install optuna-integration[tfkeras] for proper functionality.")
        def on_epoch_end(self, epoch, logs=None):
            pass
    TFKerasPruningCallback = DummyTFKerasPruningCallback
    OPTUNA_TFKERAS_AVAILABLE = False

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import gc
import copy
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from datetime import datetime

from app.models.price_prediction import PricePredictionModel
from app.models.risk_management import RiskManagementModel
from app.models.tp_sl_management import TpSlManagementModel
from app.models.indicator_management import IndicatorManagementModel
from app.models.sentiment_analysis import SentimentAnalyzer
from app.models.rl_trading import train_rl_agent
from app.models.transformer_model import FinancialTransformer  # Import the new model

# Configure logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Enhanced model trainer with advanced optimization techniques
    for improving prediction accuracy and model performance.
    """
    
    def __init__(self, trading_bot=None):
        self.trading_bot = trading_bot
        self.models_dir = os.environ.get('MODELS_DIR', 'saved_models')
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Vérification et configuration du GPU
        gpu_info = check_gpu_availability()
        self.gpu_available = gpu_info['available']
        
        # Configurer l'environnement en fonction de la disponibilité du GPU
        self.gpu_usable = configure_gpu_environment(gpu_info)
        
        if self.gpu_available:
            logger.info(f"GPU détecté via {gpu_info.get('detection_method', 'inconnu')}! Nombre de GPUs: {len(gpu_info['devices'])}")
            for i, device in enumerate(gpu_info['devices']):
                logger.info(f"  GPU #{i}: {device}")
            
            # Si les détails de mémoire sont disponibles
            if 'memory_details' in gpu_info and gpu_info['memory_details']:
                for i, mem_info in enumerate(gpu_info['memory_details']):
                    logger.info(f"  GPU #{i} VRAM: {mem_info['total_mb']}MB total, {mem_info['free_mb']}MB libre")
            
            # État de la croissance mémoire (memory growth)
            if gpu_info.get('memory_growth_enabled', False):
                logger.info("  GPU memory growth activé: la mémoire VRAM sera allouée progressivement")
            else:
                logger.warning("  GPU memory growth non activé. Raison: " + 
                              gpu_info.get('memory_growth_error', 'inconnu'))
        else:
            logger.warning("Aucun GPU détecté! L'entraînement sera plus lent.")
            if gpu_info.get('tf_detection_error'):
                logger.warning(f"Erreur TensorFlow: {gpu_info.get('tf_detection_error')}")
            if gpu_info.get('smi_detection_error'):
                logger.warning(f"Erreur nvidia-smi: {gpu_info.get('smi_detection_error')}")
            logger.warning("Si vous avez bien une carte compatible CUDA, vérifiez l'installation des pilotes NVIDIA")
            logger.warning("et exécutez le script install_nvidia_docker.sh")
        
        # Optimization settings
        self.max_optimization_trials = int(os.environ.get('MAX_OPTIMIZATION_TRIALS', 25))
        self.optimization_timeout = int(os.environ.get('OPTIMIZATION_TIMEOUT', 3600))  # 1 hour default
        # Utiliser le GPU seulement si disponible, correctement configuré, et activé par l'utilisateur
        self.use_gpu = self.gpu_usable and os.environ.get('USE_GPU', 'true').lower() == 'true'
        
        # Performance tracking
        self.model_metrics = {}
        self.best_params = {}
        
        # New model types
        self.use_transformer = os.environ.get('TRANSFORMER_MODEL_ENABLED', 'true').lower() == 'true'
        
        # Online learning settings
        self.enable_online_learning = os.environ.get('ENABLE_ONLINE_LEARNING', 'true').lower() == 'true'
        self.online_learning_epochs = int(os.environ.get('ONLINE_LEARNING_EPOCHS', 5))
        self.online_learning_batch_size = int(os.environ.get('ONLINE_LEARNING_BATCH_SIZE', 32))
        self.min_data_points_for_update = int(os.environ.get('MIN_DATA_POINTS_FOR_UPDATE', 30))
        self.online_learning_memory = {}  # Store recent data points for incremental learning
        
        # Configuration pour la carte RTX 2070 SUPER
        lstm_units = int(os.environ.get('LSTM_UNITS', 128))
        
        logger.info(f"ModelTrainer initialisé avec paramètres optimisés pour RTX 2070 SUPER")
        logger.info(f"Trials: {self.max_optimization_trials}, Timeout: {self.optimization_timeout}s")
        logger.info(f"Accélération GPU: {'activée' if self.use_gpu else 'désactivée'}")
        logger.info(f"Modèle Transformer: {'activé' if self.use_transformer else 'désactivé'}")
        logger.info(f"Apprentissage en ligne: {'activé' if self.enable_online_learning else 'désactivé'}")
        logger.info(f"Unités LSTM: {lstm_units}")
    
    async def train_all_models(self, data_manager):
        """
        Train all models with automated hyperparameter optimization
        """
        logger.info("Starting comprehensive model training...")
        
        trained_models = {}
        
        # Vérifier si on doit limiter les symboles pour l'entrainement
        train_only_essential = os.environ.get('TRAIN_ONLY_ESSENTIAL_SYMBOLS', 'false').lower() == 'true'
        essential_symbols = os.environ.get('ESSENTIAL_SYMBOLS', '').split(',')
        
        # Déterminer les symboles à entraîner
        symbols_to_train = essential_symbols if train_only_essential else data_manager.symbols
        logger.info(f"Symbols to train: {symbols_to_train} (out of {len(data_manager.symbols)} available)")
        
        # Déterminer si on utilise l'entraînement parallèle
        enable_parallel = os.environ.get('ENABLE_PARALLEL_TRAINING', 'false').lower() == 'true'
        max_parallel = int(os.environ.get('MAX_PARALLEL_MODELS', 2))
        
        if enable_parallel:
            logger.info(f"Using parallel training with max {max_parallel} models at once")
            # Entraînement parallèle avec limitation du nombre de modèles
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {}
                for symbol in symbols_to_train:
                    if symbol not in data_manager.symbols:
                        logger.warning(f"Symbol {symbol} not found in data_manager, skipping")
                        continue
                        
                    try:
                        data = data_manager.data.get(symbol)
                        
                        if data is None or data.empty:
                            logger.warning(f"No data available for {symbol}, skipping training")
                            continue
                        
                        future = executor.submit(self._train_symbol_models_sync, data, symbol)
                        futures[future] = symbol
                        
                    except Exception as e:
                        logger.error(f"Error preparing training for {symbol}: {e}")
                
                # Collecter les résultats au fur et à mesure qu'ils se terminent
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        symbol_models = future.result()
                        trained_models[symbol] = symbol_models
                        logger.info(f"All models for {symbol} completed training")
                        # Libérer la mémoire après chaque modèle
                        gc.collect()
                    except Exception as e:
                        logger.error(f"Error during parallel training for {symbol}: {e}")
        else:
            logger.info("Using sequential training (one model at a time)")
            # Entraînement séquentiel (un symbole à la fois)
            for symbol in symbols_to_train:
                if symbol not in data_manager.symbols:
                    logger.warning(f"Symbol {symbol} not found in data_manager, skipping")
                    continue
                    
                try:
                    logger.info(f"Training models for {symbol}...")
                    data = data_manager.data.get(symbol)
                    
                    if data is None or data.empty:
                        logger.warning(f"No data available for {symbol}, skipping training")
                        continue
                    
                    # Train symbol-specific models
                    symbol_models = await self.train_symbol_models(data, symbol)
                    trained_models[symbol] = symbol_models
                    
                    # Clean up memory
                    gc.collect()
                    tf.keras.backend.clear_session()  # Clear TensorFlow session
                    
                except Exception as e:
                    logger.error(f"Error training models for {symbol}: {e}")
        
        # Train global models (if any)
        try:
            # Global risk model using data from all symbols
            global_data = pd.concat([
                data_manager.data[symbol].copy().assign(symbol=symbol) 
                for symbol in data_manager.symbols 
                if symbol in data_manager.data and not data_manager.data[symbol].empty
            ])
            
            if not global_data.empty:
                await self.train_global_models(global_data)
        
        except Exception as e:
            logger.error(f"Error training global models: {e}")
        
        logger.info("All model training completed")
        return trained_models
    
    def _train_symbol_models_sync(self, data, symbol):
        """Version synchrone de train_symbol_models pour l'exécution parallèle"""
        # Convertir la fonction asynchrone en synchrone
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.train_symbol_models(data, symbol))
            return result
        finally:
            loop.close()
            
    async def train_symbol_models(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Train and optimize all models for a specific symbol
        """
        models = {}
        
        # Limiter la taille des données pour réduire la consommation de mémoire
        if len(data) > 5000:
            logger.info(f"Reducing data size for {symbol}: {len(data)} → 5000 data points")
            data = data.iloc[-5000:]
            
        # Prepare data
        train_data, test_data = self._prepare_data_splits(data)
        
        # Supprimer les données originales pour libérer de la mémoire
        del data
        gc.collect()
        
        # 1. Train price prediction model
        logger.info(f"Optimizing price prediction model for {symbol}...")
        try:
            price_model = await self._optimize_price_model(train_data, test_data, symbol)
            models['price'] = price_model
            logger.info(f"Price prediction model for {symbol} trained successfully")
            # Libérer la mémoire après chaque modèle
            gc.collect()
            tf.keras.backend.clear_session()
        except Exception as e:
            logger.error(f"Error training price prediction model for {symbol}: {e}")
        
        # Vérifier si on utilise les Transformers
        use_transformer = os.environ.get('USE_TRANSFORMER_MODEL', 'true').lower() == 'true'
        
        # Déterminer si on utilise le modèle de risque
        model_complexity = os.environ.get('MODEL_COMPLEXITY', 'medium')
        if model_complexity != 'low':
            # 2. Train risk management model
            logger.info(f"Optimizing risk management model for {symbol}...")
            try:
                risk_model = await self._optimize_risk_model(train_data, test_data, symbol)
                models['risk'] = risk_model
                logger.info(f"Risk management model for {symbol} trained successfully")
                # Libérer la mémoire après chaque modèle
                gc.collect()
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.error(f"Error training risk management model for {symbol}: {e}")
            
            # 3. Train take profit / stop loss model
            logger.info(f"Optimizing TP/SL model for {symbol}...")
            try:
                tpsl_model = await self._optimize_tpsl_model(train_data, test_data, symbol)
                models['tpsl'] = tpsl_model
                logger.info(f"TP/SL model for {symbol} trained successfully")
                # Libérer la mémoire après chaque modèle
                gc.collect()
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.error(f"Error training TP/SL model for {symbol}: {e}")
            
            # 4. Train indicator model
            logger.info(f"Optimizing indicator model for {symbol}...")
            try:
                indicator_model = await self._optimize_indicator_model(train_data, test_data, symbol)
                models['indicator'] = indicator_model
                logger.info(f"Indicator model for {symbol} trained successfully")
                # Libérer la mémoire après chaque modèle
                gc.collect()
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.error(f"Error training indicator model for {symbol}: {e}")
        
        # Train transformer model if enabled
        if use_transformer:
            logger.info(f"Training transformer model for {symbol}...")
            try:
                transformer_model = await self.train_transformer_model(train_data, symbol)
                models['transformer'] = transformer_model
                logger.info(f"Transformer model for {symbol} trained successfully")
                # Libérer la mémoire après chaque modèle
                gc.collect()
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.error(f"Error training transformer model for {symbol}: {e}")
        
        # Libérer la mémoire
        del train_data, test_data
        gc.collect()
        
        # Save all model metrics for this symbol
        metrics_file = os.path.join(self.models_dir, f"{symbol}_metrics.json")
        try:
            # Utiliser to_json au lieu de DataFrame pour éviter de créer un objet volumineux
            with open(metrics_file, 'w') as f:
                json.dump(self.model_metrics.get(symbol, {}), f)
            logger.info(f"Model metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
        
        return models
    
    async def train_global_models(self, data: pd.DataFrame):
        """Train models that use data from all symbols"""
        # Global sentiment analysis model training
        try:
            # Train sentiment model on all available data
            logger.info("Training global sentiment analysis model...")
            # Implementation depends on how sentiment data is collected
            # Not fully implemented as it requires text data
        except Exception as e:
            logger.error(f"Error training global sentiment model: {e}")
    
    def _prepare_data_splits(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare time-series aware train/test split with proper feature scaling
        """
        # Ensure data is sorted by time
        data = data.sort_index()
        
        # Use time-series split (train on earlier data, test on later data)
        split_idx = int(len(data) * 0.8)  # 80% train, 20% test
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        logger.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
        
        return train_data, test_data
    
    async def _optimize_price_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame, symbol: str) -> PricePredictionModel:
        """
        Optimize hyperparameters for price prediction model
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'lookback': trial.suggest_int('lookback', 10, 60),
                'units': trial.suggest_int('units', 32, 256),
                'layers': trial.suggest_int('layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            }
            
            # Create and train model with these parameters
            model = PricePredictionModel(
                lookback=params['lookback'],
                units=params['units'],
                num_layers=params['layers'],
                dropout_rate=params['dropout']
            )
            
            # Prepare data
            X, y = model.prepare_data(train_data)
            X_val, y_val = model.prepare_data(test_data)
            
            # Configure callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                TFKerasPruningCallback(trial, 'val_loss')
            ]
            
            # Train model
            model.model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='mean_squared_error'
            )
            
            history = model.model.fit(
                X, y,
                epochs=100,
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Return best validation loss
            val_loss = min(history.history['val_loss'])
            return val_loss
        
        # Run optimization
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.max_optimization_trials, timeout=self.optimization_timeout)
        
        # Get best parameters
        best_params = study.best_params
        self.best_params[f"{symbol}_price"] = best_params
        logger.info(f"Best price prediction params for {symbol}: {best_params}")
        
        # Train final model with best parameters
        final_model = PricePredictionModel(
            lookback=best_params['lookback'],
            units=best_params['units'],
            num_layers=best_params['layers'],
            dropout_rate=best_params['dropout']
        )
        
        # Train on all data
        X, y = final_model.prepare_data(pd.concat([train_data, test_data]))
        
        # Train with best params
        final_model.model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss='mean_squared_error'
        )
        
        final_model.model.fit(
            X, y,
            epochs=100,
            batch_size=best_params['batch_size'],
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0
        )
        
        # Evaluate model
        X_test, y_test = final_model.prepare_data(test_data)
        predictions = final_model.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        self.model_metrics[f"{symbol}_price"] = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'params': best_params
        }
        
        logger.info(f"Price prediction model for {symbol} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{symbol}_price_model")
        final_model.save(model_path)
        
        return final_model
    
    async def _optimize_risk_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame, symbol: str) -> RiskManagementModel:
        """
        Optimize hyperparameters for risk management model
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'lookback': trial.suggest_int('lookback', 10, 50),
                'units': trial.suggest_int('units', 32, 128),
                'layers': trial.suggest_int('layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
            }
            
            # Create model
            model = RiskManagementModel(
                lookback=params['lookback'],
                units=params['units'],
                num_layers=params['layers'],
                dropout_rate=params['dropout']
            )
            
            # Prepare data
            X, y = model.prepare_data(train_data)
            X_val, y_val = model.prepare_data(test_data)
            
            # Configure callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                TFKerasPruningCallback(trial, 'val_loss')
            ]
            
            # Train model
            model.model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='mean_squared_error'  # For risk prediction
            )
            
            history = model.model.fit(
                X, y,
                epochs=100,
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Return best validation loss
            val_loss = min(history.history['val_loss'])
            return val_loss
        
        # Run optimization
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.max_optimization_trials, timeout=self.optimization_timeout)
        
        # Get best parameters
        best_params = study.best_params
        self.best_params[f"{symbol}_risk"] = best_params
        logger.info(f"Best risk management params for {symbol}: {best_params}")
        
        # Train final model with best parameters
        final_model = RiskManagementModel(
            lookback=best_params['lookback'],
            units=best_params['units'],
            num_layers=best_params['layers'],
            dropout_rate=best_params['dropout']
        )
        
        # Prepare all data
        X, y = final_model.prepare_data(pd.concat([train_data, test_data]))
        
        # Train with best params
        final_model.model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss='mean_squared_error'
        )
        
        final_model.model.fit(
            X, y,
            epochs=100,
            batch_size=best_params['batch_size'],
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0
        )
        
        # Evaluate model
        X_test, y_test = final_model.prepare_data(test_data)
        predictions = final_model.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        self.model_metrics[f"{symbol}_risk"] = {
            'mse': float(mse),
            'mae': float(mae),
            'params': best_params
        }
        
        logger.info(f"Risk management model for {symbol} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{symbol}_risk_model")
        final_model.save(model_path)
        
        return final_model
    
    async def _optimize_tpsl_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame, symbol: str) -> TpSlManagementModel:
        """
        Optimize hyperparameters for take-profit/stop-loss model
        """
        # Placeholder implementation
        model = TpSlManagementModel()
        model.train(train_data, symbol)
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{symbol}_tpsl_model")
        model.save(model_path)
        
        return model
    
    async def _optimize_indicator_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame, symbol: str) -> IndicatorManagementModel:
        """
        Optimize hyperparameters for indicator model
        """
        # Placeholder implementation
        model = IndicatorManagementModel()
        model.train(train_data, symbol)
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{symbol}_indicator_model")
        model.save(model_path)
        
        return model
    
    async def fine_tune_model(self, model_type: str, symbol: str, new_data: pd.DataFrame):
        """
        Fine-tune an existing model with new data
        """
        logger.info(f"Fine-tuning {model_type} model for {symbol}")
        
        # Load existing model
        model_path = os.path.join(self.models_dir, f"{symbol}_{model_type}_model")
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return None
        
        try:
            if model_type == 'price':
                model = PricePredictionModel.load(model_path)
                
                # Split data for fine-tuning
                train_size = int(len(new_data) * 0.8)
                train_data = new_data.iloc[:train_size]
                val_data = new_data.iloc[train_size:]
                
                # Prepare data
                X_train, y_train = model.prepare_data(train_data)
                X_val, y_val = model.prepare_data(val_data)
                
                # Fine-tune with a lower learning rate
                model.model.compile(
                    optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
                    loss='mean_squared_error'
                )
                
                model.model.fit(
                    X_train, y_train,
                    epochs=50,  # Fewer epochs for fine-tuning
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=0
                )
                
                # Evaluate fine-tuned model
                predictions = model.model.predict(X_val)
                mse = mean_squared_error(y_val, predictions)
                logger.info(f"Fine-tuned {model_type} model for {symbol} - MSE: {mse:.4f}")
                
                # Save fine-tuned model
                model.save(model_path)
                return model
                
            # Similar implementations for other model types
            else:
                logger.warning(f"Fine-tuning not implemented for model type {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error fine-tuning {model_type} model for {symbol}: {e}")
            return None

    async def optimize_rl_model(self, data_path: str, symbol: str = None):
        """
        Train and optimize a reinforcement learning model
        """
        logger.info(f"Starting RL model optimization for {'all symbols' if symbol is None else symbol}")
        
        # Define optimization parameters
        def objective(params):
            try:
                learning_rate = params['learning_rate']
                gamma = params['gamma']
                batch_size = int(params['batch_size'])
                
                # Train RL agent with these hyperparameters
                model, metrics = train_rl_agent(
                    data_path=data_path,
                    symbol=symbol,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    batch_size=batch_size,
                    episodes=50,  # Reduced episodes for optimization
                    verbose=0
                )
                
                # Use final reward as optimization target
                return {'loss': -metrics['final_reward'], 'status': STATUS_OK}
                
            except Exception as e:
                logger.error(f"Error in RL optimization trial: {e}")
                return {'loss': 0, 'status': STATUS_OK}  # Return a neutral value on error
        
        # Define the search space
        space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'gamma': hp.uniform('gamma', 0.9, 0.999),
            'batch_size': hp.choice('batch_size', [16, 32, 64, 128])
        }
        
        # Run the optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_optimization_trials,
            trials=trials
        )
        
        # Get best parameters
        best_params = {
            'learning_rate': best['learning_rate'],
            'gamma': best['gamma'],
            'batch_size': [16, 32, 64, 128][best['batch_size']]
        }
        
        logger.info(f"Best RL parameters: {best_params}")
        
        # Train final model with best parameters
        final_model, metrics = train_rl_agent(
            data_path=data_path,
            symbol=symbol,
            learning_rate=best_params['learning_rate'],
            gamma=best_params['gamma'],
            batch_size=best_params['batch_size'],
            episodes=100,  # More episodes for final training
            verbose=1
        )
        
        # Save best parameters
        model_tag = f"{symbol}_rl" if symbol else "global_rl"
        self.best_params[model_tag] = best_params
        self.model_metrics[model_tag] = metrics
        
        # Save hyperparameters
        params_path = os.path.join(self.models_dir, f"{model_tag}_params.joblib")
        joblib.dump(best_params, params_path)
        
        return final_model, metrics

    async def train_transformer_model(self, data, symbol):
        """Train a transformer model for time series prediction."""
        logger.info(f"Training transformer model for {symbol}")
        
        try:
            # Create model with default hyperparameters
            # In a real scenario, these could be optimized
            model = FinancialTransformer(
                input_sequence_length=30,
                forecast_horizon=5,
                d_model=64,
                num_heads=4,
                dropout_rate=0.1,
                num_transformer_blocks=2
            )
            
            # Train the model
            history = model.train(
                data=data,
                target_column='Close',
                epochs=50,
                batch_size=32,
                validation_split=0.2
            )
            
            # Save the model
            model_path = os.path.join(self.models_dir, f"{symbol}_transformer_model")
            scaler_path = os.path.join(self.models_dir, f"{symbol}_transformer_scalers.pkl")
            model.save(model_path, scaler_path)
            
            # Log training metrics
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            logger.info(f"Transformer model for {symbol} - Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training transformer model for {symbol}: {e}")
            return None

    async def incremental_update_models(self, symbol: str, new_data: pd.DataFrame):
        """
        Update all models incrementally with new market data
        
        Args:
            symbol: The trading symbol
            new_data: New market data to use for incremental learning
        
        Returns:
            Dict of updated model metrics
        """
        if not self.enable_online_learning:
            logger.info(f"Online learning disabled, skipping incremental update for {symbol}")
            return None
            
        if len(new_data) < self.min_data_points_for_update:
            logger.info(f"Not enough new data points for {symbol} (got {len(new_data)}, need {self.min_data_points_for_update})")
            return None
            
        logger.info(f"Performing incremental model update for {symbol} with {len(new_data)} new data points")
        
        # Store data in memory buffer for this symbol
        if symbol not in self.online_learning_memory:
            self.online_learning_memory[symbol] = new_data
        else:
            # Append new data to existing buffer and keep only the most recent data
            max_memory_size = 1000  # Maximum data points to keep in memory
            self.online_learning_memory[symbol] = pd.concat([
                self.online_learning_memory[symbol], 
                new_data
            ]).drop_duplicates().tail(max_memory_size)
        
        update_metrics = {}
        try:
            # Update price prediction model
            price_model_path = os.path.join(self.models_dir, f"{symbol}_price_model")
            if os.path.exists(price_model_path):
                logger.info(f"Updating price prediction model for {symbol}")
                model = PricePredictionModel.load(price_model_path)
                
                # Prepare data for training
                X, y = model.prepare_data(self.online_learning_memory[symbol])
                
                # Update model with new data
                model.model.fit(
                    X, y,
                    epochs=self.online_learning_epochs,
                    batch_size=self.online_learning_batch_size,
                    verbose=0
                )
                
                # Save updated model
                model.save(price_model_path)
                logger.info(f"Price prediction model for {symbol} updated successfully")
                
            # Update risk management model
            risk_model_path = os.path.join(self.models_dir, f"{symbol}_risk_model")
            if os.path.exists(risk_model_path):
                logger.info(f"Updating risk management model for {symbol}")
                model = RiskManagementModel.load(risk_model_path)
                
                # Prepare data for training
                X, y = model.prepare_data(self.online_learning_memory[symbol])
                
                # Update model with new data
                model.model.fit(
                    X, y,
                    epochs=self.online_learning_epochs,
                    batch_size=self.online_learning_batch_size,
                    verbose=0
                )
                
                # Save updated model
                model.save(risk_model_path)
                logger.info(f"Risk management model for {symbol} updated successfully")
            
            # Update TP/SL model
            tpsl_model_path = os.path.join(self.models_dir, f"{symbol}_tpsl_model")
            if os.path.exists(tpsl_model_path):
                logger.info(f"Updating TP/SL model for {symbol}")
                model = TpSlManagementModel.load(tpsl_model_path)
                
                # Prepare data for training
                X, y = model.prepare_data(self.online_learning_memory[symbol])
                
                # Update model with new data
                model.model.fit(
                    X, y,
                    epochs=self.online_learning_epochs,
                    batch_size=self.online_learning_batch_size,
                    verbose=0
                )
                
                # Save updated model
                model.save(tpsl_model_path)
                logger.info(f"TP/SL model for {symbol} updated successfully")
            
            # Update indicator model
            indicator_model_path = os.path.join(self.models_dir, f"{symbol}_indicator_model")
            if os.path.exists(indicator_model_path):
                logger.info(f"Updating indicator model for {symbol}")
                model = IndicatorManagementModel.load(indicator_model_path)
                
                # Prepare data for training
                X, y = model.prepare_data(self.online_learning_memory[symbol])
                
                # Update model with new data
                model.model.fit(
                    X, y,
                    epochs=self.online_learning_epochs,
                    batch_size=self.online_learning_batch_size,
                    verbose=0
                )
                
                # Save updated model
                model.save(indicator_model_path)
                logger.info(f"Indicator model for {symbol} updated successfully")
                
            # Update transformer model if enabled
            if self.use_transformer:
                transformer_model_path = os.path.join(self.models_dir, f"{symbol}_transformer_model")
                scaler_path = os.path.join(self.models_dir, f"{symbol}_transformer_scalers.pkl")
                
                if os.path.exists(transformer_model_path) and os.path.exists(scaler_path):
                    logger.info(f"Updating transformer model for {symbol}")
                    model = FinancialTransformer.load(transformer_model_path, scaler_path)
                    
                    # Update model with new data
                    history = model.train(
                        data=self.online_learning_memory[symbol],
                        target_column='Close',
                        epochs=self.online_learning_epochs,
                        batch_size=self.online_learning_batch_size,
                        validation_split=0.2
                    )
                    
                    # Save updated model
                    model.save(transformer_model_path, scaler_path)
                    
                    # Store metrics
                    update_metrics['transformer'] = {
                        'loss': history.history['loss'][-1],
                        'val_loss': history.history.get('val_loss', [0])[-1]
                    }
                    
                    logger.info(f"Transformer model for {symbol} updated successfully")
            
            # Save update timestamp to track when models were last updated
            update_info = {
                'last_update': datetime.now().isoformat(),
                'data_points': len(new_data),
                'metrics': update_metrics
            }
            
            update_info_path = os.path.join(self.models_dir, f"{symbol}_update_info.json")
            with open(update_info_path, 'w') as f:
                json.dump(update_info, f)
                
            logger.info(f"Incremental update for {symbol} completed successfully")
            return update_metrics
            
        except Exception as e:
            logger.error(f"Error during incremental update for {symbol}: {e}")
            return None

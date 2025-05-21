import numpy as np
import pandas as pd
import tensorflow as tf
import os
import logging
import joblib
import time
from collections import deque
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc

logger = logging.getLogger(__name__)

class ContinualLearningManager:
    """
    Advanced continual learning manager for financial models.
    
    This class implements several strategies to prevent catastrophic forgetting and
    ensure effective adaptation to changing market conditions, including:
    
    1. Experience Replay: Storing and replaying important past examples
    2. Elastic Weight Consolidation (EWC): Preserving important weights
    3. Concept Drift Detection: Detecting when market conditions change
    4. Sample Weighting: Weighing samples based on recency and importance
    5. Regularization: Dynamic regularization based on data characteristics
    """
    
    def __init__(
        self,
        memory_size=5000,
        buffer_strategy='diverse',  # 'recent', 'diverse', 'prototype', 'uncertainty'
        ewc_lambda=1.0,
        drift_detection_window=30,
        drift_threshold=0.05,
        lr_decay_factor=0.1,
        enable_ewc=True,
        enable_drift_detection=True,
        models_dir='saved_models/online_learning'
    ):
        """
        Initialize the continual learning manager.
        
        Args:
            memory_size: Maximum number of samples to keep in memory buffer
            buffer_strategy: Strategy for memory buffer management
            ewc_lambda: Importance weight for EWC regularization
            drift_detection_window: Window size for drift detection
            drift_threshold: Threshold for concept drift detection
            lr_decay_factor: Learning rate decay factor for fine-tuning
            enable_ewc: Whether to use Elastic Weight Consolidation
            enable_drift_detection: Whether to enable concept drift detection
            models_dir: Directory for saving model checkpoints
        """
        self.memory_size = memory_size
        self.buffer_strategy = buffer_strategy
        self.ewc_lambda = ewc_lambda
        self.drift_detection_window = drift_detection_window
        self.drift_threshold = drift_threshold
        self.lr_decay_factor = lr_decay_factor
        self.enable_ewc = enable_ewc
        self.enable_drift_detection = enable_drift_detection
        self.models_dir = models_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize memory buffers for different symbols
        self.memory_buffers = {}
        self.fisher_diagonals = {}  # For EWC
        self.importance_scores = {}
        self.distribution_stats = {}  # For drift detection
        self.last_update_time = {}
        
        # Track model performance for each symbol
        self.model_metrics = {}
        
        logger.info(f"Initialized ContinualLearningManager with memory size: {memory_size}, "
                   f"buffer strategy: {buffer_strategy}, EWC: {enable_ewc}, "
                   f"drift detection: {enable_drift_detection}")
    
    def initialize_memory(self, symbol, input_shape):
        """Initialize memory buffer for a new symbol."""
        if symbol not in self.memory_buffers:
            self.memory_buffers[symbol] = {
                'X': deque(maxlen=self.memory_size),
                'y': deque(maxlen=self.memory_size),
                'importance': deque(maxlen=self.memory_size)
            }
            self.distribution_stats[symbol] = {
                'mean': None,
                'std': None,
                'history': deque(maxlen=100)
            }
            self.last_update_time[symbol] = time.time()
            logger.info(f"Initialized memory buffer for {symbol} with input shape {input_shape}")
    
    def add_samples(self, symbol, X, y, importance=None):
        """
        Add new samples to the memory buffer.
        
        Args:
            symbol: Trading symbol
            X: Input features
            y: Target values
            importance: Optional importance scores for each sample
        """
        if symbol not in self.memory_buffers:
            self.initialize_memory(symbol, X.shape[1:])
        
        # If importance is not provided, compute it based on recency
        if importance is None:
            # Assign higher importance to more recent samples
            n_samples = len(X)
            importance = np.linspace(0.5, 1.0, n_samples)
        
        # Add samples to memory buffer
        for i in range(len(X)):
            self.memory_buffers[symbol]['X'].append(X[i])
            self.memory_buffers[symbol]['y'].append(y[i])
            self.memory_buffers[symbol]['importance'].append(importance[i])
        
        # Update distribution statistics for drift detection
        if self.enable_drift_detection:
            self._update_distribution_stats(symbol, X)
        
        logger.info(f"Added {len(X)} samples to memory buffer for {symbol}")
        self.last_update_time[symbol] = time.time()
    
    def _update_distribution_stats(self, symbol, X):
        """Update distribution statistics for concept drift detection."""
        # Calculate mean and std of the features
        X_flat = X.reshape(X.shape[0], -1)
        current_mean = np.mean(X_flat, axis=0)
        current_std = np.std(X_flat, axis=0)
        
        # Store current stats
        self.distribution_stats[symbol]['history'].append((current_mean, current_std))
        
        # Update overall stats with exponential moving average
        if self.distribution_stats[symbol]['mean'] is None:
            self.distribution_stats[symbol]['mean'] = current_mean
            self.distribution_stats[symbol]['std'] = current_std
        else:
            alpha = 0.1  # Weight for new observations
            self.distribution_stats[symbol]['mean'] = (1 - alpha) * self.distribution_stats[symbol]['mean'] + alpha * current_mean
            self.distribution_stats[symbol]['std'] = (1 - alpha) * self.distribution_stats[symbol]['std'] + alpha * current_std
    
    def detect_drift(self, symbol, X):
        """
        Detect concept drift in the data.
        
        Args:
            symbol: Trading symbol
            X: New data samples
            
        Returns:
            drift_detected: Whether drift was detected
            drift_score: Magnitude of the drift
        """
        if not self.enable_drift_detection or symbol not in self.distribution_stats:
            return False, 0.0
        
        # Flatten the input
        X_flat = X.reshape(X.shape[0], -1)
        
        # Calculate current distribution statistics
        current_mean = np.mean(X_flat, axis=0)
        current_std = np.std(X_flat, axis=0)
        
        # Compare with stored distribution statistics
        if self.distribution_stats[symbol]['mean'] is None:
            return False, 0.0
        
        # Calculate normalized distance between distributions
        mean_dist = np.linalg.norm(current_mean - self.distribution_stats[symbol]['mean'])
        std_dist = np.linalg.norm(current_std - self.distribution_stats[symbol]['std'])
        
        # Normalize by the magnitude of the stored statistics
        mean_magnitude = np.linalg.norm(self.distribution_stats[symbol]['mean'])
        std_magnitude = np.linalg.norm(self.distribution_stats[symbol]['std'])
        
        norm_mean_dist = mean_dist / (mean_magnitude + 1e-10)
        norm_std_dist = std_dist / (std_magnitude + 1e-10)
        
        # Combined drift score
        drift_score = 0.7 * norm_mean_dist + 0.3 * norm_std_dist
        
        # Detect drift if score exceeds threshold
        drift_detected = drift_score > self.drift_threshold
        
        if drift_detected:
            logger.warning(f"Concept drift detected for {symbol} with score {drift_score:.4f} (threshold: {self.drift_threshold})")
        
        return drift_detected, drift_score
    
    def select_replay_samples(self, symbol, current_X, current_y, max_samples=1000):
        """
        Select samples for replay based on the chosen buffer strategy.
        
        Args:
            symbol: Trading symbol
            current_X: Current batch of input features
            current_y: Current batch of target values
            max_samples: Maximum number of samples to select from memory
            
        Returns:
            X_replay: Selected input features for replay
            y_replay: Selected target values for replay
            sample_weights: Importance weights for selected samples
        """
        if symbol not in self.memory_buffers or len(self.memory_buffers[symbol]['X']) == 0:
            return None, None, None
        
        # Convert deques to numpy arrays
        memory_X = np.array(list(self.memory_buffers[symbol]['X']))
        memory_y = np.array(list(self.memory_buffers[symbol]['y']))
        memory_importance = np.array(list(self.memory_buffers[symbol]['importance']))
        
        # Different selection strategies
        if self.buffer_strategy == 'recent':
            # Select the most recent samples
            indices = np.arange(len(memory_X))
            indices = indices[-max_samples:] if len(indices) > max_samples else indices
            
        elif self.buffer_strategy == 'diverse':
            # Select diverse samples using k-means clustering
            if len(memory_X) > max_samples:
                # Reshape to 2D if needed
                X_2d = memory_X.reshape(memory_X.shape[0], -1)
                
                # Use MiniBatchKMeans for efficiency
                n_clusters = min(max_samples, len(memory_X) // 5)
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(X_2d)
                
                # Select samples closest to centroids
                centroids = kmeans.cluster_centers_
                indices = []
                
                for centroid in centroids:
                    # Find sample closest to this centroid
                    distances = cdist([centroid], X_2d, 'euclidean')[0]
                    closest_idx = np.argmin(distances)
                    indices.append(closest_idx)
                
                # Add more samples to reach max_samples, prioritizing by importance
                remaining = max_samples - len(indices)
                if remaining > 0:
                    # Exclude already selected indices
                    mask = np.ones(len(memory_X), dtype=bool)
                    mask[indices] = False
                    
                    # Sort by importance and add top remaining
                    importance_order = np.argsort(-memory_importance[mask])[:remaining]
                    additional_indices = np.arange(len(memory_X))[mask][importance_order]
                    indices.extend(additional_indices)
            else:
                indices = np.arange(len(memory_X))
                
        elif self.buffer_strategy == 'uncertainty':
            # Select samples with highest importance scores (e.g., uncertain predictions)
            indices = np.argsort(-memory_importance)
            indices = indices[:max_samples] if len(indices) > max_samples else indices
            
        else:  # default to 'prototype'
            # Select prototypical examples representing the distribution
            if len(memory_X) > max_samples:
                # Reshape for distance calculation
                X_2d = memory_X.reshape(memory_X.shape[0], -1)
                
                # Calculate pairwise distances
                distances = cdist(X_2d, X_2d, 'euclidean')
                
                # Initialize with the most central example
                centrality = np.sum(distances, axis=1)
                indices = [np.argmin(centrality)]
                
                # Greedily add diverse examples
                for _ in range(min(max_samples - 1, len(memory_X) - 1)):
                    # Find example that is most distant from already selected examples
                    min_distances = np.min(distances[indices, :], axis=0)
                    next_idx = np.argmax(min_distances)
                    indices.append(next_idx)
            else:
                indices = np.arange(len(memory_X))
        
        # Get selected samples
        X_replay = memory_X[indices]
        y_replay = memory_y[indices]
        
        # Prepare sample weights based on importance
        sample_weights = memory_importance[indices]
        
        logger.info(f"Selected {len(indices)} replay samples for {symbol} using strategy: {self.buffer_strategy}")
        
        return X_replay, y_replay, sample_weights
    
    def compute_fisher_diagonal(self, model, X, y, num_samples=100):
        """
        Compute the diagonal of the Fisher Information Matrix for EWC.
        
        Args:
            model: Keras model
            X: Input features
            y: Target values
            num_samples: Number of samples to use for Fisher computation
            
        Returns:
            Dictionary mapping parameter names to their Fisher values
        """
        # Use a subset of samples for efficiency
        if len(X) > num_samples:
            indices = np.random.choice(len(X), num_samples, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
        else:
            X_subset = X
            y_subset = y
        
        # Get trainable parameters
        params = model.trainable_variables
        fisher_diagonals = {p.name: tf.zeros_like(p) for p in params}
        
        # Compute fisher diagonal for each sample
        for i in range(len(X_subset)):
            x_sample = X_subset[i:i+1]
            y_sample = y_subset[i:i+1]
            
            with tf.GradientTape() as tape:
                predictions = model(x_sample, training=True)
                loss = tf.keras.losses.mean_squared_error(y_sample, predictions)
            
            # Get gradients
            grads = tape.gradient(loss, params)
            
            # Square gradients to approximate Fisher
            for param, grad in zip(params, grads):
                if grad is not None:
                    fisher_diagonals[param.name] += tf.square(grad)
        
        # Average over samples
        for name in fisher_diagonals:
            fisher_diagonals[name] /= len(X_subset)
        
        return fisher_diagonals
    
    def create_ewc_loss_function(self, model, base_loss_fn, fisher_diagonals, old_params):
        """
        Create a customized loss function with EWC regularization.
        
        Args:
            model: Keras model
            base_loss_fn: Base loss function
            fisher_diagonals: Dictionary of fisher diagonal values
            old_params: Dictionary of old parameter values
            
        Returns:
            EWC-regularized loss function
        """
        def ewc_loss(y_true, y_pred):
            # Base loss
            base_loss = base_loss_fn(y_true, y_pred)
            
            # EWC regularization
            ewc_reg = 0
            for i, param in enumerate(model.trainable_variables):
                if param.name in fisher_diagonals and param.name in old_params:
                    fisher = fisher_diagonals[param.name]
                    old_param = old_params[param.name]
                    ewc_reg += tf.reduce_sum(fisher * tf.square(param - old_param))
            
            return base_loss + (self.ewc_lambda * ewc_reg)
        
        return ewc_loss
    
    def update_model(self, symbol, model, new_X, new_y, epochs=10, batch_size=32, validation_split=0.2):
        """
        Update a model with new data using continual learning techniques.
        
        Args:
            symbol: Trading symbol
            model: Keras model to update
            new_X: New input features
            new_y: New target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Updated model and training history
        """
        # Check if we need to initialize memory for this symbol
        if symbol not in self.memory_buffers:
            self.initialize_memory(symbol, new_X.shape[1:])
        
        # Detect concept drift
        drift_detected, drift_score = self.detect_drift(symbol, new_X)
        
        # Store current model parameters for EWC
        old_params = {}
        if self.enable_ewc:
            for param in model.trainable_variables:
                old_params[param.name] = tf.identity(param)
        
        # Save current learning rate
        original_lr = float(model.optimizer.learning_rate.numpy()) if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate
        
        # Adjust learning rate based on drift
        if drift_detected:
            # Higher learning rate for significant drift
            new_lr = original_lr
            logger.info(f"Keeping original learning rate {new_lr} due to detected drift")
        else:
            # Lower learning rate for fine-tuning
            new_lr = original_lr * self.lr_decay_factor
            logger.info(f"Reducing learning rate from {original_lr} to {new_lr} for fine-tuning")
        
        model.optimizer.learning_rate = new_lr
        
        # Select replay samples from memory buffer
        X_replay, y_replay, sample_weights = self.select_replay_samples(symbol, new_X, new_y)
        
        # Prepare combined dataset for training
        if X_replay is not None and len(X_replay) > 0:
            # Combine new data with replay samples
            X_combined = np.concatenate([new_X, X_replay])
            y_combined = np.concatenate([new_y, y_replay])
            
            # Create sample weights: higher for new data, use stored importance for replay
            weights_new = np.ones(len(new_X)) * 1.2  # Slightly higher weight for new data
            weights_combined = np.concatenate([weights_new, sample_weights])
            
            logger.info(f"Training with {len(new_X)} new samples and {len(X_replay)} replay samples")
        else:
            # Use only new data
            X_combined = new_X
            y_combined = new_y
            weights_combined = np.ones(len(new_X))
            
            logger.info(f"Training with {len(new_X)} new samples (no replay samples available)")
        
        # Apply EWC if enabled and we have previous fisher values
        if self.enable_ewc and symbol in self.fisher_diagonals:
            # Create custom loss function with EWC regularization
            original_loss = model.loss
            ewc_loss = self.create_ewc_loss_function(
                model, original_loss, self.fisher_diagonals[symbol], old_params
            )
            
            # Temporarily set the model's loss function
            model.compile(
                optimizer=model.optimizer,
                loss=ewc_loss,
                metrics=model.metrics
            )
            
            logger.info(f"Applied EWC regularization for {symbol}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Train the model
        history = model.fit(
            X_combined, y_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            sample_weight=weights_combined,
            callbacks=callbacks,
            verbose=1
        )
        
        # Restore original loss function if EWC was applied
        if self.enable_ewc and symbol in self.fisher_diagonals:
            model.compile(
                optimizer=model.optimizer,
                loss=original_loss,
                metrics=model.metrics
            )
        
        # Compute fisher diagonal for the updated model
        if self.enable_ewc:
            logger.info(f"Computing Fisher diagonal for {symbol}")
            self.fisher_diagonals[symbol] = self.compute_fisher_diagonal(model, X_combined, y_combined)
        
        # Add new samples to memory buffer with importance scores based on prediction error
        with tf.GradientTape() as tape:
            predictions = model(new_X, training=False)
            errors = tf.abs(predictions - new_y)
        
        # Normalize errors to importance scores [0.5, 1.0]
        max_error = tf.reduce_max(errors)
        min_error = tf.reduce_min(errors)
        importance = 0.5 + 0.5 * (errors - min_error) / (max_error - min_error + 1e-10)
        importance = importance.numpy()
        
        # Add samples to memory buffer
        self.add_samples(symbol, new_X, new_y, importance)
        
        # Update model metrics
        val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
        self.model_metrics[symbol] = {
            'last_update_time': time.time(),
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'val_loss': val_loss,
            'memory_size': len(self.memory_buffers[symbol]['X'])
        }
        
        # Reset learning rate to original
        model.optimizer.learning_rate = original_lr
        
        # Free memory
        gc.collect()
        
        logger.info(f"Model update completed for {symbol}")
        return model, history
    
    def save_state(self, filename=None):
        """
        Save the state of the continual learning manager.
        
        Args:
            filename: Custom filename, defaults to continual_learning_state.pkl
        """
        if filename is None:
            filename = os.path.join(self.models_dir, 'continual_learning_state.pkl')
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare state to save (excluding tensorflow objects)
        save_state = {
            'memory_buffers': self.memory_buffers,
            'distribution_stats': self.distribution_stats,
            'last_update_time': self.last_update_time,
            'model_metrics': self.model_metrics,
            'config': {
                'memory_size': self.memory_size,
                'buffer_strategy': self.buffer_strategy,
                'ewc_lambda': self.ewc_lambda,
                'drift_detection_window': self.drift_detection_window,
                'drift_threshold': self.drift_threshold,
                'enable_ewc': self.enable_ewc,
                'enable_drift_detection': self.enable_drift_detection
            }
        }
        
        # Save state
        joblib.dump(save_state, filename)
        logger.info(f"Saved continual learning state to {filename}")
    
    def load_state(self, filename=None):
        """
        Load the state of the continual learning manager.
        
        Args:
            filename: Custom filename, defaults to continual_learning_state.pkl
        """
        if filename is None:
            filename = os.path.join(self.models_dir, 'continual_learning_state.pkl')
            
        if not os.path.exists(filename):
            logger.warning(f"No saved state found at {filename}")
            return False
        
        # Load state
        save_state = joblib.load(filename)
        
        # Restore state
        self.memory_buffers = save_state['memory_buffers']
        self.distribution_stats = save_state['distribution_stats']
        self.last_update_time = save_state['last_update_time']
        self.model_metrics = save_state['model_metrics']
        
        # Restore configuration
        config = save_state['config']
        self.memory_size = config['memory_size']
        self.buffer_strategy = config['buffer_strategy']
        self.ewc_lambda = config['ewc_lambda']
        self.drift_detection_window = config['drift_detection_window']
        self.drift_threshold = config['drift_threshold']
        self.enable_ewc = config['enable_ewc']
        self.enable_drift_detection = config['enable_drift_detection']
        
        logger.info(f"Loaded continual learning state from {filename}")
        return True
    
    def get_memory_stats(self, symbol=None):
        """
        Get statistics about the memory buffer.
        
        Args:
            symbol: Trading symbol, if None returns stats for all symbols
            
        Returns:
            Dictionary with memory buffer statistics
        """
        if symbol is not None:
            if symbol not in self.memory_buffers:
                return {'error': f'No memory buffer for symbol {symbol}'}
            
            return {
                'memory_size': len(self.memory_buffers[symbol]['X']),
                'last_update': self.last_update_time.get(symbol),
                'drift_score': self.model_metrics.get(symbol, {}).get('drift_score'),
                'metrics': self.model_metrics.get(symbol, {})
            }
        else:
            # Stats for all symbols
            return {
                symbol: {
                    'memory_size': len(buf['X']),
                    'last_update': self.last_update_time.get(symbol),
                    'metrics': self.model_metrics.get(symbol, {})
                }
                for symbol, buf in self.memory_buffers.items()
            }

class OnlineLearningModel:
    """
    Interface pour l'apprentissage en ligne des modèles de trading.
    
    Cette classe encapsule le gestionnaire d'apprentissage continu (ContinualLearningManager)
    et fournit une interface simplifiée pour l'intégration avec le système de trading.
    """
    
    def __init__(
        self,
        memory_size=5000,
        buffer_strategy='diverse',
        enable_ewc=True,
        enable_drift_detection=True,
        models_dir='saved_models/online_learning'
    ):
        """
        Initialiser le modèle d'apprentissage en ligne.
        
        Args:
            memory_size: Taille maximale de la mémoire tampon
            buffer_strategy: Stratégie de gestion de la mémoire ('recent', 'diverse', 'prototype', 'uncertainty')
            enable_ewc: Activer la consolidation élastique des poids (Elastic Weight Consolidation)
            enable_drift_detection: Activer la détection de dérive conceptuelle
            models_dir: Répertoire pour sauvegarder les points de contrôle des modèles
        """
        self.cl_manager = ContinualLearningManager(
            memory_size=memory_size,
            buffer_strategy=buffer_strategy,
            enable_ewc=enable_ewc,
            enable_drift_detection=enable_drift_detection,
            models_dir=models_dir
        )
        
        # Dictionnaire pour stocker les modèles par symbole
        self.models = {}
        
        # Statistiques d'apprentissage
        self.update_stats = {}
        
        logger.info(f"Initialized OnlineLearningModel with memory size: {memory_size}, "
                   f"buffer strategy: {buffer_strategy}, EWC: {enable_ewc}, "
                   f"drift detection: {enable_drift_detection}")
    
    def initialize_for_symbol(self, symbol, model, input_shape):
        """
        Initialiser l'apprentissage en ligne pour un symbole spécifique.
        
        Args:
            symbol: Symbole de trading
            model: Modèle TensorFlow/Keras initial
            input_shape: Forme des données d'entrée
        """
        self.models[symbol] = model
        self.cl_manager.initialize_memory(symbol, input_shape)
        self.update_stats[symbol] = {
            'updates': 0,
            'drift_detected': 0,
            'samples_processed': 0,
            'last_loss': None,
            'last_update_time': time.time()
        }
        logger.info(f"Initialized online learning for {symbol}")
    
    def update(self, symbol, X, y, epochs=5, batch_size=32, validation_split=0.2):
        """
        Mettre à jour le modèle avec de nouvelles données.
        
        Args:
            symbol: Symbole de trading
            X: Données d'entrée (features)
            y: Données cibles
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du lot
            validation_split: Fraction des données pour la validation
            
        Returns:
            Dictionnaire contenant les métriques d'entraînement
        """
        if symbol not in self.models:
            logger.warning(f"Model for {symbol} not initialized. Skipping update.")
            return None
        
        # Vérifier la dérive conceptuelle
        if self.cl_manager.enable_drift_detection:
            drift_detected = self.cl_manager.detect_drift(symbol, X)
            if drift_detected:
                logger.info(f"Concept drift detected for {symbol}. Adjusting learning parameters.")
                self.update_stats[symbol]['drift_detected'] += 1
                # Augmenter le nombre d'époques en cas de dérive
                epochs = min(epochs * 2, 20)
        
        # Ajouter les échantillons à la mémoire
        importance = np.ones(len(X))  # Importance uniforme par défaut
        self.cl_manager.add_samples(symbol, X, y, importance)
        
        # Mettre à jour le modèle
        update_result = self.cl_manager.update_model(
            symbol, 
            self.models[symbol], 
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split
        )
        
        # Mettre à jour les statistiques
        self.update_stats[symbol]['updates'] += 1
        self.update_stats[symbol]['samples_processed'] += len(X)
        self.update_stats[symbol]['last_loss'] = update_result.get('val_loss', update_result.get('loss'))
        self.update_stats[symbol]['last_update_time'] = time.time()
        
        logger.info(f"Updated model for {symbol} with {len(X)} samples. "
                   f"Loss: {self.update_stats[symbol]['last_loss']}")
        
        return update_result
    
    def predict(self, symbol, X):
        """
        Faire des prédictions avec le modèle mis à jour.
        
        Args:
            symbol: Symbole de trading
            X: Données d'entrée
            
        Returns:
            Prédictions du modèle
        """
        if symbol not in self.models:
            logger.warning(f"Model for {symbol} not initialized. Cannot predict.")
            return None
        
        return self.models[symbol].predict(X)
    
    def get_stats(self, symbol=None):
        """
        Obtenir les statistiques d'apprentissage en ligne.
        
        Args:
            symbol: Symbole spécifique (ou None pour tous les symboles)
            
        Returns:
            Statistiques d'apprentissage
        """
        if symbol:
            if symbol in self.update_stats:
                stats = self.update_stats[symbol].copy()
                # Ajouter les statistiques de mémoire
                memory_stats = self.cl_manager.get_memory_stats(symbol)
                stats.update(memory_stats)
                return stats
            return None
        
        # Retourner les statistiques pour tous les symboles
        all_stats = {}
        for sym in self.update_stats:
            all_stats[sym] = self.get_stats(sym)
        return all_stats
    
    def save(self, directory=None):
        """
        Sauvegarder l'état du modèle d'apprentissage en ligne.
        
        Args:
            directory: Répertoire de sauvegarde (utilise models_dir par défaut si None)
        """
        if directory is None:
            directory = self.cl_manager.models_dir
        
        os.makedirs(directory, exist_ok=True)
        
        # Sauvegarder l'état du gestionnaire d'apprentissage continu
        self.cl_manager.save_state(os.path.join(directory, 'cl_manager_state.pkl'))
        
        # Sauvegarder les modèles
        for symbol, model in self.models.items():
            model_path = os.path.join(directory, f'{symbol}_online_model.h5')
            model.save(model_path)
        
        # Sauvegarder les statistiques
        joblib.dump(self.update_stats, os.path.join(directory, 'update_stats.pkl'))
        
        logger.info(f"Saved online learning state to {directory}")
    
    def load(self, directory=None):
        """
        Charger l'état du modèle d'apprentissage en ligne.
        
        Args:
            directory: Répertoire de chargement (utilise models_dir par défaut si None)
        """
        if directory is None:
            directory = self.cl_manager.models_dir
        
        # Charger l'état du gestionnaire d'apprentissage continu
        cl_state_path = os.path.join(directory, 'cl_manager_state.pkl')
        if os.path.exists(cl_state_path):
            self.cl_manager.load_state(cl_state_path)
        
        # Charger les modèles
        for file in os.listdir(directory):
            if file.endswith('_online_model.h5'):
                symbol = file.replace('_online_model.h5', '')
                model_path = os.path.join(directory, file)
                self.models[symbol] = tf.keras.models.load_model(model_path)
        
        # Charger les statistiques
        stats_path = os.path.join(directory, 'update_stats.pkl')
        if os.path.exists(stats_path):
            self.update_stats = joblib.load(stats_path)
        
        logger.info(f"Loaded online learning state from {directory}") 
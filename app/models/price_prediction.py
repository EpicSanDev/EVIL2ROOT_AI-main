from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, TimeDistributed, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import traceback
import inspect
import gc
import psutil
import time

# Configuration optimisée pour TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Autoriser la croissance mémoire GPU
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info(f"GPU disponible et configurée: {physical_devices}")
    except Exception as e:
        logging.warning(f"Impossible de configurer la croissance mémoire GPU: {e}")
else:
    logging.info("Aucun GPU détecté, utilisation du CPU")
    
# Configuration de la mémoire pour réduire les fuites mémoires
tf.config.optimizer.set_jit(True)  # Activer XLA pour améliorer les performances

class PricePredictionModel:
    def __init__(self, sequence_length=60, future_periods=1, model_dir='models'):
        """
        Initialize the price prediction model with advanced configuration.
        
        Args:
            sequence_length: Number of time steps to look back
            future_periods: Number of time steps to predict ahead
            model_dir: Directory to save trained models
        """
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.sequence_length = sequence_length
        self.future_periods = future_periods
        self.model_dir = model_dir
        self.version = '2.1.0'  # Version du modèle pour le suivi des améliorations
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(model_dir, 'model_training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Moniteur de performances
        self.performance_metrics = {}

    def _prepare_features(self, data):
        """
        Create advanced features for better prediction accuracy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators and derived features
            
        Raises:
            ValueError: If critical columns are missing from the data
        """
        df = data.copy()
        
        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Séparer les colonnes en critiques et non-critiques
        critical_columns = ['Open', 'High', 'Low', 'Close']
        critical_missing = [col for col in critical_columns if col not in df.columns]
        
        if critical_missing:
            error_msg = f"Colonnes critiques manquantes: {critical_missing}. Impossible de continuer le traitement."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if missing_columns:
            self.logger.warning(f"Colonnes manquantes: {missing_columns}. Certaines fonctionnalités ne pourront pas être calculées.")
        
        # Basic price features
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Price differences
            df['PriceDiff'] = df['Close'] - df['Open']
            df['HighLowDiff'] = df['High'] - df['Low']
            df['HighCloseDiff'] = df['High'] - df['Close']
            df['LowCloseDiff'] = df['Close'] - df['Low']
            
            # Percentage changes
            df['Returns'] = df['Close'].pct_change()
            df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'MA_diff_{window}'] = df['Close'] - df[f'MA_{window}']
                df[f'MA_ratio_{window}'] = df['Close'] / df[f'MA_{window}']
            
            # Volatility features
            for window in [5, 10, 20, 50]:
                df[f'Volatility_{window}'] = df['LogReturns'].rolling(window=window).std()
            
            # Price momentum
            for window in [5, 10, 20]:
                df[f'Momentum_{window}'] = df['Close'].pct_change(window)
            
            # Bollinger Bands
            for window in [20]:
                df[f'BB_middle_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'BB_std_{window}'] = df['Close'].rolling(window=window).std()
                df[f'BB_upper_{window}'] = df[f'BB_middle_{window}'] + 2 * df[f'BB_std_{window}']
                df[f'BB_lower_{window}'] = df[f'BB_middle_{window}'] - 2 * df[f'BB_std_{window}']
                df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / df[f'BB_middle_{window}']
                df[f'BB_relative_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}'])
        
        # Volume-based features
        if 'Volume' in df.columns:
            df['VolumeChange'] = df['Volume'].pct_change()
            df['VolumeMA_5'] = df['Volume'].rolling(5).mean()
            df['VolumeMA_10'] = df['Volume'].rolling(10).mean()
            df['VolumeRatio_5'] = df['Volume'] / df['VolumeMA_5']
            df['VolumePriceRatio'] = df['Volume'] / df['Close']
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['Year'] = df.index.year
            df['IsMonthStart'] = df.index.is_month_start.astype(int)
            df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        
        # Drop NaN values that occur due to rolling calculations
        df = df.dropna()
        
        return df

    def _scale_data(self, data, symbol, train_mode=True):
        """
        Scale the data using robust scaling methods.
        
        Args:
            data: DataFrame with features
            symbol: Trading symbol
            train_mode: Whether we're training (fit_transform) or predicting (transform)
            
        Returns:
            Scaled data as numpy array
        """
        if train_mode:
            # Initialize scalers for each feature
            self.feature_scalers[symbol] = {}
            
            # Target variable scaler (separate from features)
            self.scalers[symbol] = RobustScaler()
            y_scaled = self.scalers[symbol].fit_transform(data['Close'].values.reshape(-1, 1))
            
            # Feature scalers
            X_scaled = np.zeros((len(data), len(data.columns) - 1))
            feature_idx = 0
            
            for col in data.columns:
                if col != 'Close':
                    scaler = RobustScaler()
                    X_scaled[:, feature_idx] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                    self.feature_scalers[symbol][col] = scaler
                    feature_idx += 1
                    
            return X_scaled, y_scaled
        else:
            # For prediction mode
            if symbol not in self.feature_scalers or symbol not in self.scalers:
                raise ValueError(f"Scalers for {symbol} not found. Train the model first.")
                
            X_scaled = np.zeros((len(data), len(data.columns) - 1))
            feature_idx = 0
            
            for col in data.columns:
                if col != 'Close':
                    scaler = self.feature_scalers[symbol].get(col)
                    if scaler:
                        X_scaled[:, feature_idx] = scaler.transform(data[col].values.reshape(-1, 1)).flatten()
                        feature_idx += 1
            
            y_scaled = self.scalers[symbol].transform(data['Close'].values.reshape(-1, 1))
            return X_scaled, y_scaled

    def _create_sequences(self, X, y, sequence_length, future_periods=1):
        """
        Create input sequences and target values.
        
        Args:
            X: Feature data
            y: Target data
            sequence_length: Length of input sequence
            future_periods: How many steps into the future to predict
            
        Returns:
            X_seq: Sequences of input features
            y_seq: Target values
        """
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X) - future_periods + 1):
            X_seq.append(X[i - sequence_length:i])
            y_seq.append(y[i + future_periods - 1])
            
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, params, input_shape, output_shape=1):
        """
        Build the neural network model for price prediction.
        
        Args:
            params: Dictionary of hyperparameters
            input_shape: Shape of input features
            output_shape: Number of outputs (default: 1 for single step prediction)
            
        Returns:
            Compiled Keras model
        """
        # Extraction des hyperparamètres
        model_type = params.get('model_type', 'hybrid')
        lstm_units = params.get('lstm_units', 64)
        dropout_rate = params.get('dropout_rate', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        l1_reg = params.get('l1_reg', 0.0)
        l2_reg = params.get('l2_reg', 0.0)
        use_batch_norm = params.get('use_batch_norm', True)
        use_bidirectional = params.get('use_bidirectional', True)
        use_attention = params.get('use_attention', True)
        num_attention_heads = params.get('num_attention_heads', 2)
        key_dim = params.get('key_dim', 32)
        
        # Surveillance de la mémoire avant construction du modèle
        self._log_memory_usage("Avant construction du modèle")
        
        # Sélecteur de modèle
        if model_type == 'simple_lstm':
            model = self._build_simple_lstm_model(input_shape, lstm_units, dropout_rate, 
                                                 l1_reg, l2_reg, use_batch_norm, output_shape)
        elif model_type == 'bidirectional_lstm':
            model = self._build_bidirectional_lstm_model(input_shape, lstm_units, dropout_rate, 
                                                         l1_reg, l2_reg, use_batch_norm, output_shape)
        elif model_type == 'cnn_lstm':
            model = self._build_cnn_lstm_model(input_shape, lstm_units, dropout_rate, 
                                              l1_reg, l2_reg, use_batch_norm, output_shape)
        elif model_type == 'attention_lstm':
            model = self._build_attention_lstm_model(input_shape, lstm_units, dropout_rate,
                                                   l1_reg, l2_reg, use_batch_norm, 
                                                   num_attention_heads, key_dim, output_shape)
        elif model_type == 'hybrid':
            model = self._build_hybrid_model(input_shape, lstm_units, dropout_rate,
                                           l1_reg, l2_reg, use_batch_norm, 
                                           use_bidirectional, use_attention,
                                           num_attention_heads, key_dim, output_shape)
        else:
            self.logger.warning(f"Type de modèle inconnu '{model_type}', utilisation du modèle hybride par défaut")
            model = self._build_hybrid_model(input_shape, lstm_units, dropout_rate,
                                           l1_reg, l2_reg, use_batch_norm, 
                                           use_bidirectional, use_attention,
                                           num_attention_heads, key_dim, output_shape)
        
        # Compilation du modèle
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mape']
        )
        
        # Surveillance de la mémoire après construction du modèle
        self._log_memory_usage("Après construction du modèle")
        
        return model
    
    def _build_simple_lstm_model(self, input_shape, lstm_units, dropout_rate, 
                              l1_reg, l2_reg, use_batch_norm, output_shape):
        """Simple LSTM model"""
        model = Sequential()
        model.add(LSTM(
            lstm_units, 
            input_shape=input_shape,
            return_sequences=False,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape))
        return model
    
    def _build_bidirectional_lstm_model(self, input_shape, lstm_units, dropout_rate, 
                                     l1_reg, l2_reg, use_batch_norm, output_shape):
        """Bidirectional LSTM model for better capturing of trends"""
        model = Sequential()
        model.add(Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ),
            input_shape=input_shape
        ))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(
            LSTM(
                lstm_units // 2,
                return_sequences=False,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            )
        ))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape))
        return model
    
    def _build_cnn_lstm_model(self, input_shape, lstm_units, dropout_rate, 
                           l1_reg, l2_reg, use_batch_norm, output_shape):
        """CNN-LSTM model to capture both local and temporal patterns"""
        model = Sequential()
        model.add(Conv1D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
            input_shape=input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same'
        ))
        model.add(MaxPooling1D(pool_size=2))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(LSTM(
            lstm_units,
            return_sequences=False,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape))
        return model
    
    def _build_attention_lstm_model(self, input_shape, lstm_units, dropout_rate,
                                  l1_reg, l2_reg, use_batch_norm, 
                                  num_attention_heads, key_dim, output_shape):
        """LSTM model with multi-head attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layer with Layer Normalization
        x = LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        )(inputs)
        
        if use_batch_norm:
            x = LayerNormalization()(x)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=key_dim
        )(x, x)
        
        # Residual connection
        x = Add()([x, attention_output])
        
        # Layer Normalization
        x = LayerNormalization()(x)
        
        # Second LSTM layer
        x = LSTM(
            lstm_units // 2,
            return_sequences=False,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        )(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(output_shape)(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def _build_hybrid_model(self, input_shape, lstm_units, dropout_rate,
                          l1_reg, l2_reg, use_batch_norm, 
                          use_bidirectional, use_attention,
                          num_attention_heads, key_dim, output_shape):
        """
        Hybrid model combining CNN for feature extraction, 
        LSTM (optionally bidirectional) for temporal dynamics,
        and attention mechanism for focusing on important timesteps
        """
        inputs = Input(shape=input_shape)
        
        # CNN feature extraction
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        # Deeper CNN
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        # LSTM layer (bidirectional if specified)
        if use_bidirectional:
            x = Bidirectional(LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))(x)
        else:
            x = LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            )(x)
        
        if use_batch_norm:
            x = LayerNormalization()(x)
        
        # Attention mechanism if specified
        if use_attention:
            attention_output = MultiHeadAttention(
                num_heads=num_attention_heads,
                key_dim=key_dim
            )(x, x)
            
            # Residual connection
            x = Add()([x, attention_output])
            
            # Layer Normalization
            x = LayerNormalization()(x)
        
        # Second LSTM layer
        if use_bidirectional:
            x = Bidirectional(LSTM(
                lstm_units // 2,
                return_sequences=False,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            ))(x)
        else:
            x = LSTM(
                lstm_units // 2,
                return_sequences=False,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
            )(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        x = Dense(
            32,
            activation='relu',
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        )(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        x = Dropout(dropout_rate / 2)(x)
        
        # Output layer
        outputs = Dense(output_shape)(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def _log_memory_usage(self, step_name):
        """Log memory usage at various steps for debugging and optimization"""
        # Récupérer l'utilisation mémoire du processus
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Convertir en MB
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # Nettoyer la mémoire inutilisée si possible
        gc.collect()
        
        self.logger.info(f"Utilisation mémoire à '{step_name}': {memory_mb:.2f} MB")
        
        # Stocker pour analyse ultérieure
        self.performance_metrics[f"mémoire_{step_name}"] = memory_mb
    
    def _optimize_hyperparameters(self, X_train, y_train, symbol, max_trials=25, timeout=3600):
        """
        Perform real hyperparameter optimization using Bayesian optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            symbol: Trading symbol
            max_trials: Maximum number of optimization trials
            timeout: Maximum optimization time in seconds
            
        Returns:
            Dict of best hyperparameters
        """
        self.logger.info(f"Starting hyperparameter optimization for {symbol}")
        
        # Define the search space
        param_space = {
            'model_type': Categorical(['lstm', 'bidirectional', 'gru', 'conv_lstm', 'hybrid']),
            'lstm_units': Integer(50, 200),
            'conv_filters': Integer(32, 128),
            'kernel_size': Integer(2, 5),
            'dense_units': Integer(16, 64),
            'dropout_rate': Real(0.1, 0.5),
            'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
            'batch_size': Integer(16, 128),
            'l1': Real(1e-6, 1e-3, prior='log-uniform'),
            'l2': Real(1e-6, 1e-3, prior='log-uniform'),
            'loss': Categorical(['mean_squared_error', 'mean_absolute_error', 'huber_loss'])
        }
        
        # Define a model creation function for BayesSearchCV
        def create_model_for_bayes(**params):
            model = self.build_model(params, input_shape=(X_train.shape[1], X_train.shape[2]))
            return model
        
        # Use TimeSeriesSplit for time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Set up callbacks for each CV iteration
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Define the scorer - we'll use negative MSE for regression
        def custom_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)
        
        # Configure the search
        search = BayesSearchCV(
            estimator=tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model_for_bayes),
            search_spaces=param_space,
            n_iter=max_trials,  # Number of parameter settings sampled
            cv=tscv,
            n_jobs=1,  # Run on a single core (increase if more hardware is available)
            verbose=1,
            scoring=custom_scorer,
            return_train_score=True,
            fit_params={'callbacks': callbacks, 'epochs': 50, 'validation_split': 0.2},
            timeout=timeout
        )
        
        # Perform the search
        try:
            search.fit(X_train, y_train)
            self.logger.info(f"Best hyperparameters for {symbol}: {search.best_params_}")
            
            # Save the optimization results
            optimization_results = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'all_results': search.cv_results_
            }
            
            joblib.dump(
                optimization_results, 
                os.path.join(self.model_dir, f'{symbol}_hyperopt_results.pkl')
            )
            
            return search.best_params_
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter optimization: {str(e)}")
            # Fallback parameters
            self.logger.info("Using fallback parameters")
            return {
                'model_type': 'lstm',
                'lstm_units': 100,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'dense_units': 32,
                'l1': 1e-5,
                'l2': 1e-5,
                'loss': 'mean_squared_error'
            }

    def train(self, data=None, symbol=None, optimize=True, epochs=100, validation_split=0.2, **kwargs):
        """
        Train or optimize the price prediction model.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            optimize: Whether to perform hyperparameter optimization
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            
        Returns:
            History object from model training
        """
        if data is None or symbol is None:
            self.logger.error("Les données et le symbole sont requis pour l'entraînement")
            return None
        
        self.logger.info(f"Début de l'entraînement pour {symbol}")
        training_start_time = time.time()
        self._log_memory_usage("Début d'entraînement")
            
        # Prepare data
        prepared_data = self._prepare_features(data)
        self.logger.info(f"Nombre de caractéristiques: {prepared_data.shape[1]}")
        
        # Split into features and target
        X = prepared_data.drop('Close', axis=1)
        y = prepared_data['Close']
        
        # Scale data
        X_scaled, _ = self._scale_data(X, symbol, train_mode=True)
        y_scaled, _ = self._scale_data(pd.DataFrame(y), symbol + '_target', train_mode=True)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(
            X_scaled, y_scaled.values, self.sequence_length, self.future_periods
        )
        
        # Vérification des dimensions
        self.logger.info(f"Dimensions des séquences: X={X_seq.shape}, y={y_seq.shape}")
        self._log_memory_usage("Après préparation des données")
        
        # Stratified time series split for validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for i, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            if i == n_splits - 1:  # Use only the last split
                X_train, X_val = X_seq[train_idx], X_seq[val_idx]
                y_train, y_val = y_seq[train_idx], y_seq[val_idx]
        
        self.logger.info(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]}")
        self.logger.info(f"Taille de l'ensemble de validation: {X_val.shape[0]}")
        
        # Hyperparameter optimization if required
        if optimize:
            self.logger.info("Début de l'optimisation des hyperparamètres...")
            self._log_memory_usage("Avant optimisation")
            
            best_params = self._optimize_hyperparameters(
                X_train, y_train, symbol, 
                max_trials=kwargs.get('max_trials', 25),
                timeout=kwargs.get('timeout', 3600)
            )
            
            self.logger.info(f"Meilleurs hyperparamètres: {best_params}")
            self._log_memory_usage("Après optimisation")
        else:
            # Use default parameters
            best_params = {
                'model_type': 'hybrid',
                'lstm_units': 64,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'l1_reg': 0.0001,
                'l2_reg': 0.0001,
                'use_batch_norm': True,
                'use_bidirectional': True,
                'use_attention': True,
                'num_attention_heads': 2,
                'key_dim': 32
            }
            self.logger.info(f"Utilisation des paramètres par défaut: {best_params}")
        
        # Build model with best parameters
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_model(best_params, input_shape)
        
        # Log model summary
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        self.logger.info("Architecture du modèle:\n" + "\n".join(stringlist))
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'{symbol}_best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.logger.info(f"Entraînement du modèle pour {symbol} avec {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=kwargs.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model on validation set
        self._log_memory_usage("Avant évaluation")
        evaluation = self._evaluate_model(model, X_val, y_val, symbol)
        self.logger.info(f"Métriques d'évaluation: {evaluation}")
        
        # Store the model
        self.models[symbol] = model
        
        # Save the model to disk
        self.logger.info(f"Sauvegarde du modèle pour {symbol}")
        self.save(symbol)
        
        # Log training time
        training_time = time.time() - training_start_time
        self.logger.info(f"Temps total d'entraînement pour {symbol}: {training_time:.2f} secondes")
        self._log_memory_usage("Fin d'entraînement")
        
        # Free memory
        gc.collect()
        
        return history

    def _evaluate_model(self, model, X_val, y_val, symbol):
        """
        Evaluate the model and save performance metrics.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            symbol: Trading symbol
        """
        y_pred = model.predict(X_val)
        
        # Convert scaled predictions back to original values
        y_val_orig = self.scalers[symbol].inverse_transform(y_val)
        y_pred_orig = self.scalers[symbol].inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_val_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_val_orig - y_pred_orig) / y_val_orig)) * 100
        
        # Save metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(
            metrics,
            os.path.join(self.model_dir, f'{symbol}_metrics.pkl')
        )
        
        # Log metrics
        self.logger.info(f"Model evaluation for {symbol}:")
        self.logger.info(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
        self.logger.info(f"R²: {r2:.6f}, MAPE: {mape:.2f}%")
        
        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(y_val_orig[:100], label='Actual')
        plt.plot(y_pred_orig[:100], label='Predicted')
        plt.title(f'Price Prediction Evaluation for {symbol}')
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, f'{symbol}_prediction_plot.png'))
        plt.close()

    def predict(self, data, symbol, days_ahead=1):
        """
        Make price predictions for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            days_ahead: Number of days to predict ahead
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if symbol not in self.models:
            self.logger.warning(f"Aucun modèle trouvé pour {symbol}, tentative de chargement...")
            try:
                self.load(symbol)
            except Exception as e:
                self.logger.error(f"Impossible de charger le modèle pour {symbol}: {e}")
                return None
        
        # Mesure du temps de prédiction pour les performances
        prediction_start_time = time.time()
        
        # Prepare data
        prepared_data = self._prepare_features(data)
        
        # Ensure we have enough data
        if len(prepared_data) < self.sequence_length + days_ahead:
            self.logger.error(f"Données insuffisantes pour la prédiction ({len(prepared_data)} < {self.sequence_length + days_ahead})")
            return None
        
        # Split into features and target
        X = prepared_data.drop('Close', axis=1)
        y = prepared_data['Close']
        
        # Scale data
        X_scaled, _ = self._scale_data(X, symbol, train_mode=False)
        
        # For scaling back predictions
        _, target_scaler = self._scale_data(pd.DataFrame(y), symbol + '_target', train_mode=False)
        
        # Create sequences for each day ahead
        all_predictions = []
        
        for i in range(days_ahead):
            # Get the sequence for current prediction day
            if i == 0:
                # For first day, use the latest available data
                X_pred = X_scaled[-self.sequence_length:].values.reshape(1, self.sequence_length, X_scaled.shape[1])
            else:
                # For subsequent days, shift the window by adding the previous prediction
                # and dropping the oldest observation
                X_pred = np.roll(X_pred, -1, axis=1)
                # Here we would need to update the last entry with the previous prediction
                # This is a simplified approach, as ideally we'd reconstruct all features
            
            # Make prediction
            pred_scaled = self.models[symbol].predict(X_pred, verbose=0)
            
            # Add batch dimension if needed
            if len(pred_scaled.shape) == 1:
                pred_scaled = pred_scaled.reshape(-1, 1)
            
            # Inverse scaling
            pred = target_scaler.inverse_transform(pred_scaled)[0, 0]
            
            # Calculate prediction interval (simpler approach for multiple days)
            pred_std = np.std(y[-30:]) * (1 + i * 0.1)  # Increasing uncertainty with time
            lower_bound = pred - 1.96 * pred_std
            upper_bound = pred + 1.96 * pred_std
            
            # Calculate prediction date
            if isinstance(data.index, pd.DatetimeIndex):
                last_date = data.index[-1]
                if i == 0:
                    # For the first prediction, use the next day after the last date
                    pred_date = last_date + pd.Timedelta(days=1)
                else:
                    # For subsequent predictions, add one day to the previous prediction date
                    pred_date = all_predictions[-1]['date'] + pd.Timedelta(days=1)
            else:
                # If index is not datetime, use numeric index
                pred_date = i + 1
            
            # Store prediction
            all_predictions.append({
                'date': pred_date,
                'predicted_price': pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence': 1.0 - (i * 0.05)  # Decreasing confidence with time
            })
        
        # Create DataFrame of predictions
        predictions_df = pd.DataFrame(all_predictions)
        
        # Log prediction time
        prediction_time = time.time() - prediction_start_time
        self.logger.debug(f"Temps de prédiction pour {symbol}: {prediction_time:.4f} secondes")
        
        return predictions_df

    def save(self, symbol=None):
        """
        Save the model and associated data to disk.
        
        Args:
            symbol: Trading symbol or None to save all models
        """
        # If symbol is None, save all models
        if symbol is None:
            symbols = list(self.models.keys())
            self.logger.info(f"Saving all models: {symbols}")
        else:
            symbols = [symbol]
            self.logger.info(f"Saving model for {symbol}")
        
        for sym in symbols:
            if sym in self.models:
                # Save the model
                self.models[sym].save(os.path.join(self.model_dir, f'{sym}_model.h5'))
                
                # Save scalers if available
                if sym in self.scalers:
                    joblib.dump(
                        self.scalers[sym],
                        os.path.join(self.model_dir, f'{sym}_target_scaler.pkl')
                    )
                    
                if sym in self.feature_scalers:
                    joblib.dump(
                        self.feature_scalers[sym],
                        os.path.join(self.model_dir, f'{sym}_feature_scalers.pkl')
                    )
                    
                self.logger.info(f"Model and data for {sym} saved successfully")

    def load(self, symbol):
        """
        Load model and related data from disk.
        
        Args:
            symbol: Trading symbol to load
            
        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(self.model_dir, f'{symbol}_model.h5')
        scaler_path = os.path.join(self.model_dir, f'{symbol}_target_scaler.pkl')
        feature_scalers_path = os.path.join(self.model_dir, f'{symbol}_feature_scalers.pkl')
        
        try:
            if os.path.exists(model_path):
                self.models[symbol] = load_model(model_path)
                self.logger.info(f"Model loaded for {symbol}")
                
                if os.path.exists(scaler_path):
                    self.scalers[symbol] = joblib.load(scaler_path)
                    self.logger.info(f"Target scaler loaded for {symbol}")
                    
                if os.path.exists(feature_scalers_path):
                    self.feature_scalers[symbol] = joblib.load(feature_scalers_path)
                    self.logger.info(f"Feature scalers loaded for {symbol}")
                
                return True
            else:
                self.logger.warning(f"No saved model found for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model for {symbol}: {str(e)}")
            return False
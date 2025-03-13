from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten
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
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Configure TensorFlow for better performance
        tf.config.optimizer.set_jit(True)  # Enable XLA acceleration
        
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
        Build a sophisticated deep learning model architecture.
        
        Args:
            params: Dict of hyperparameters
            input_shape: Shape of input data
            output_shape: Number of output values
            
        Returns:
            Compiled Keras model
        """
        model_type = params.get('model_type', 'hybrid')
        
        if model_type == 'lstm':
            model = Sequential()
            model.add(LSTM(params['lstm_units'], return_sequences=True, 
                          input_shape=input_shape,
                          kernel_regularizer=l1_l2(params.get('l1', 0.0), params.get('l2', 0.0))))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            model.add(LSTM(params['lstm_units'] // 2, return_sequences=True,
                          kernel_regularizer=l1_l2(params.get('l1', 0.0), params.get('l2', 0.0))))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            model.add(LSTM(params['lstm_units'] // 4, return_sequences=False,
                          kernel_regularizer=l1_l2(params.get('l1', 0.0), params.get('l2', 0.0))))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            model.add(Dense(params.get('dense_units', 32), activation='relu',
                           kernel_regularizer=l1_l2(params.get('l1', 0.0), params.get('l2', 0.0))))
            model.add(Dropout(params['dropout_rate'] / 2))
            model.add(Dense(output_shape))
            
        elif model_type == 'bidirectional':
            model = Sequential()
            model.add(Bidirectional(LSTM(params['lstm_units'], return_sequences=True), 
                                    input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            model.add(Bidirectional(LSTM(params['lstm_units'] // 2, return_sequences=False)))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            model.add(Dense(params.get('dense_units', 32), activation='relu'))
            model.add(Dropout(params['dropout_rate'] / 2))
            model.add(Dense(output_shape))
            
        elif model_type == 'gru':
            model = Sequential()
            model.add(GRU(params['lstm_units'], return_sequences=True, 
                         input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            model.add(GRU(params['lstm_units'] // 2, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            model.add(Dense(params.get('dense_units', 32), activation='relu'))
            model.add(Dropout(params['dropout_rate'] / 2))
            model.add(Dense(output_shape))
            
        elif model_type == 'conv_lstm':
            model = Sequential()
            model.add(Conv1D(filters=params.get('conv_filters', 64),
                             kernel_size=params.get('kernel_size', 3),
                             activation='relu',
                             input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(BatchNormalization())
            
            model.add(LSTM(params['lstm_units'], return_sequences=True))
            model.add(Dropout(params['dropout_rate']))
            
            model.add(LSTM(params['lstm_units'] // 2, return_sequences=False))
            model.add(Dropout(params['dropout_rate']))
            
            model.add(Dense(params.get('dense_units', 32), activation='relu'))
            model.add(Dropout(params['dropout_rate'] / 2))
            model.add(Dense(output_shape))
            
        elif model_type == 'hybrid':
            # Hybrid model with parallel paths
            input_layer = Input(shape=input_shape)
            
            # CNN path
            conv = Conv1D(filters=params.get('conv_filters', 64),
                          kernel_size=params.get('kernel_size', 3),
                          activation='relu')(input_layer)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = BatchNormalization()(conv)
            conv = Flatten()(conv)
            
            # LSTM path
            lstm = LSTM(params['lstm_units'], return_sequences=True)(input_layer)
            lstm = Dropout(params['dropout_rate'])(lstm)
            lstm = LSTM(params['lstm_units'] // 2, return_sequences=False)(lstm)
            lstm = Dropout(params['dropout_rate'])(lstm)
            
            # Combine paths
            combined = Concatenate()([conv, lstm])
            combined = Dense(params.get('dense_units', 64), activation='relu')(combined)
            combined = Dropout(params['dropout_rate'] / 2)(combined)
            combined = Dense(params.get('dense_units', 32), activation='relu')(combined)
            output = Dense(output_shape)(combined)
            
            model = Model(inputs=input_layer, outputs=output)
        
        # Compile model with specified optimizer
        optimizer = Adam(learning_rate=params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss=params.get('loss', 'mean_squared_error'))
        
        return model

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
        Train the model with advanced features and techniques.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            optimize: Whether to optimize hyperparameters
            epochs: Number of training epochs if not optimizing
            validation_split: Validation data fraction
            **kwargs: Additional arguments that might be passed
            
        Returns:
            Training history object
        """
        # Handle various parameter combinations, including keyword arguments
        # Extensive parameter handling to avoid the missing symbol issue
        if data is None and 'data' in kwargs:
            data = kwargs.get('data')
        if data is None and 'market_data' in kwargs:
            data = kwargs.get('market_data')
            
        if symbol is None and 'symbol' in kwargs:
            symbol = kwargs.get('symbol')
            
        # More parameters that might be in kwargs
        optimize = kwargs.get('optimize', optimize)
        epochs = kwargs.get('epochs', epochs)
        validation_split = kwargs.get('validation_split', validation_split)
        
        # Ajout d'un log détaillé
        caller = inspect.getouterframes(inspect.currentframe())[1]
        caller_info = f"{caller.filename}:{caller.lineno} in {caller.function}"
        
        # Print extra debugging information
        self.logger.info(f"EXTRA DEBUG - train parameters: data={type(data)}, symbol={symbol}, optimize={optimize}")
        self.logger.info(f"EXTRA DEBUG - caller info: {caller_info}")
        self.logger.info(f"EXTRA DEBUG - arguments: {inspect.signature(self.train)}")
        
        # If symbol is None, raise a detailed error to help debugging
        if symbol is None:
            error_msg = f"Symbol parameter is required but was None. Called from {caller_info}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.logger.info(f"DÉBUT DE TRAIN pour {symbol} - appelé depuis: {caller_info}")
        
        try:
            self.logger.info(f"Training model for symbol: {symbol}")
            
            # Limiter la taille des données pour réduire la consommation de mémoire
            if len(data) > 5000:
                self.logger.info(f"Réduction de la taille des données: {len(data)} → 5000 points")
                data = data.iloc[-5000:]
            
            # Create features
            feature_data = self._prepare_features(data)
            self.logger.info(f"Created {len(feature_data.columns)} features for {symbol}")
            
            # Libérer la mémoire
            del data
            gc.collect()
            
            # Scale data
            X_scaled, y_scaled = self._scale_data(feature_data, symbol, train_mode=True)
            
            # Libérer la mémoire
            del feature_data
            gc.collect()
            
            # Lire la séquence depuis les variables d'environnement
            seq_length = int(os.environ.get('SEQUENCE_LENGTH', self.sequence_length))
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled, seq_length, self.future_periods)
            self.logger.info(f"Created {len(X_seq)} sequences for {symbol}")
            
            # Libérer la mémoire
            del X_scaled, y_scaled
            gc.collect()
            
            # Split data into training and validation sets
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Libérer la mémoire
            del X_seq, y_seq
            gc.collect()
            
            # Get hyperparameters
            if optimize:
                # Limiter le nombre d'essais depuis les variables d'environnement
                max_trials = int(os.environ.get('MAX_OPTIMIZATION_TRIALS', 25))
                optimization_timeout = int(os.environ.get('OPTIMIZATION_TIMEOUT', 3600))
                
                self.logger.info(f"Optimizing hyperparameters with max_trials={max_trials}, timeout={optimization_timeout}s")
                best_params = self._optimize_hyperparameters(X_train, y_train, symbol, 
                                                           max_trials=max_trials, 
                                                           timeout=optimization_timeout)
            else:
                best_params = {
                    'model_type': 'hybrid',
                    'lstm_units': 100,
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': int(os.environ.get('BATCH_SIZE', 32)),
                    'dense_units': 32,
                    'l1': 1e-5,
                    'l2': 1e-5,
                    'loss': 'mean_squared_error'
                }
            
            # Ajuster la complexité du modèle en fonction de la variable MODEL_COMPLEXITY
            model_complexity = os.environ.get('MODEL_COMPLEXITY', 'medium')
            if model_complexity == 'low':
                best_params['lstm_units'] = min(best_params.get('lstm_units', 100), 64)
                best_params['dense_units'] = min(best_params.get('dense_units', 32), 16)
            elif model_complexity == 'high':
                # Garder les valeurs optimisées
                pass
            
            # Build and train the model
            self.logger.info(f"Building model with parameters: {best_params}")
            model = self.build_model(best_params, input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Define callbacks
            model_path = os.path.join(self.model_dir, f'{symbol}_model.h5')
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Ajustement des époques en fonction de MODEL_COMPLEXITY
            if model_complexity == 'low':
                epochs = min(epochs, 50)
            elif model_complexity == 'medium':
                epochs = min(epochs, 100)
            
            # Train the model
            batch_size = best_params.get('batch_size', 32)
            self.logger.info(f"Training model for {epochs} epochs with batch_size={batch_size}")
            
            # Configurer TensorFlow pour gérer l'utilisation du GPU de manière robuste
            use_gpu = os.environ.get('USE_GPU', 'true').lower() == 'true'
            gpu_available = False
            
            if use_gpu:
                try:
                    # Importer la fonction de vérification et configuration du GPU
                    from app.model_trainer import check_gpu_availability, configure_gpu_environment
                    
                    # Vérifier la disponibilité du GPU
                    gpu_info = check_gpu_availability()
                    gpu_available = gpu_info['available']
                    
                    if gpu_available:
                        # Configurer l'environnement GPU
                        gpu_usable = configure_gpu_environment(gpu_info)
                        if not gpu_usable:
                            self.logger.warning("GPU détecté mais ne peut pas être configuré correctement. Utilisation du CPU.")
                            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                            gpu_available = False
                        else:
                            self.logger.info(f"Utilisation du GPU pour l'entraînement: {gpu_info.get('devices', ['GPU'])}")
                    else:
                        self.logger.warning("Aucun GPU détecté. L'entraînement se fera sur CPU.")
                        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                except Exception as e:
                    self.logger.error(f"Erreur lors de la configuration du GPU: {e}. Utilisation du CPU.")
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    gpu_available = False
            else:
                # Forcer l'utilisation du CPU si use_gpu est False
                self.logger.info("Utilisation forcée du CPU (GPU désactivé par configuration)")
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Ajustement des paramètres d'entraînement en fonction de la disponibilité du GPU
            if not gpu_available:
                # Réduire la taille du batch sur CPU pour éviter les problèmes de mémoire
                batch_size = min(batch_size, 16)
                self.logger.info(f"Taille de batch réduite à {batch_size} pour utilisation CPU")
                
                # Réduire le nombre d'époques si nécessaire
                if epochs > 50:
                    self.logger.info(f"Réduction du nombre d'époques de {epochs} à 50 (mode CPU)")
                    epochs = 50
            
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Libérer la mémoire
                del X_train, y_train, X_val, y_val
                gc.collect()
                
                # Load the best model (saved by ModelCheckpoint)
                if os.path.exists(model_path):
                    model = load_model(model_path)
            except Exception as e:
                self.logger.error(f"Erreur lors de la formation du modèle: {e}")
            
            # Store the model
            self.models[symbol] = model
            
            # Save feature column names for prediction
            joblib.dump(
                list(feature_data.columns) if 'feature_data' in locals() else [],
                os.path.join(self.model_dir, f'{symbol}_feature_columns.pkl')
            )
            
            # Save scalers
            joblib.dump(
                self.scalers[symbol],
                os.path.join(self.model_dir, f'{symbol}_target_scaler.pkl')
            )
            joblib.dump(
                self.feature_scalers[symbol],
                os.path.join(self.model_dir, f'{symbol}_feature_scalers.pkl')
            )
            
            # Save hyperparameters
            joblib.dump(
                best_params,
                os.path.join(self.model_dir, f'{symbol}_hyperparams.pkl')
            )
            
            # Libérer la mémoire associée aux modèles pour éviter les fuites
            tf.keras.backend.clear_session()
            gc.collect()
            
            self.logger.info(f"Model trained for symbol: {symbol}")
            return history

        except Exception as e:
            self.logger.error(f"Error during model training for {symbol}: {str(e)}")
            # Libérer la mémoire en cas d'erreur
            tf.keras.backend.clear_session()
            gc.collect()
            raise

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
        Make predictions using the trained model.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            days_ahead: How many days ahead to predict
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if symbol not in self.models:
            # Try to load the model from disk
            model_path = os.path.join(self.model_dir, f'{symbol}_model.h5')
            if os.path.exists(model_path):
                self.models[symbol] = load_model(model_path)
                
                # Load scalers
                scaler_path = os.path.join(self.model_dir, f'{symbol}_target_scaler.pkl')
                feature_scalers_path = os.path.join(self.model_dir, f'{symbol}_feature_scalers.pkl')
                
                if os.path.exists(scaler_path) and os.path.exists(feature_scalers_path):
                    self.scalers[symbol] = joblib.load(scaler_path)
                    self.feature_scalers[symbol] = joblib.load(feature_scalers_path)
                else:
                    raise ValueError(f"Scalers for {symbol} not found. Train the model first.")
            else:
                raise ValueError(f"Model for {symbol} not found. Train the model first.")
        
        # Get feature column names
        feature_columns_path = os.path.join(self.model_dir, f'{symbol}_feature_columns.pkl')
        
        try:
            # Create features from input data
            feature_data = self._prepare_features(data)
            
            # Check if we have enough features
            if len(feature_data.columns) < 5:  # Simple heuristic check
                self.logger.warning(f"Missing column PriceDiff in input data. Creating features...")
                # Try to create more features
                feature_data = self._prepare_features(data)
            
            # Scale the features
            X_scaled, _ = self._scale_data(feature_data, symbol, train_mode=False)
            
            # Create sequences
            X_seq, _ = self._create_sequences(X_scaled, np.zeros((len(X_scaled), 1)), self.sequence_length, self.future_periods)
            
            # Make predictions
            predictions = self.models[symbol].predict(X_seq)
            
            # Inverse transform to get actual prices
            predictions_orig = self.scalers[symbol].inverse_transform(predictions)
            
            # Get the last actual price
            last_price = data['Close'].iloc[-1]
            
            # Calculate predicted price
            predicted_price = predictions_orig[-1][0]
            
            # Calculate confidence intervals (simple approach)
            std_dev = np.std(data['Close'].pct_change().dropna()) * np.sqrt(days_ahead)
            lower_bound = predicted_price * (1 - 1.96 * std_dev)
            upper_bound = predicted_price * (1 + 1.96 * std_dev)
            
            # Calculate percent change
            percent_change = ((predicted_price / last_price) - 1) * 100
            
            # Return prediction with confidence intervals
            return {
                'current_price': float(last_price),
                'predicted_price': float(predicted_price),
                'percent_change': float(percent_change),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence': float(1 - std_dev)  # Simple confidence measure
            }
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            
            # Fallback to a simple prediction based on the last price
            if 'Close' in data.columns:
                last_price = data['Close'].iloc[-1]
                # Simple random walk prediction
                predicted_price = last_price * (1 + np.random.normal(0, 0.01))
                return {
                    'current_price': float(last_price),
                    'predicted_price': float(predicted_price),
                    'percent_change': float(((predicted_price / last_price) - 1) * 100),
                    'lower_bound': float(last_price * 0.95),
                    'upper_bound': float(last_price * 1.05),
                    'confidence': 0.5,
                    'is_fallback': True
                }
            else:
                raise ValueError(f"Cannot make prediction: no Close column in data and model not trained for {symbol}")

    def save(self, symbol=None):
        """
        Save model(s) and related data to disk.
        
        Args:
            symbol: Specific symbol to save, or None to save all
        """
        if symbol:
            symbols = [symbol]
        else:
            symbols = list(self.models.keys())
            
        for sym in symbols:
            if sym in self.models:
                # Save model
                model_path = os.path.join(self.model_dir, f'{sym}_model.h5')
                self.models[sym].save(model_path)
                
                # Save scalers
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
            else:
                self.logger.warning(f"No model found for {sym}")

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
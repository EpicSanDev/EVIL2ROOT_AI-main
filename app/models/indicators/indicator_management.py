import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class IndicatorManagementModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def train(self, data, symbol):
        indicators = self.calculate_indicators(data)
        self.scalers[symbol] = MinMaxScaler()
        indicators_scaled = self.scalers[symbol].fit_transform(indicators)

        model = Ridge(alpha=1.0)
        model.fit(indicators_scaled, data['Close'].values)
        self.models[symbol] = model

    def calculate_indicators(self, data):
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['BB_upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
        data['BB_lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)

        # Moving Average Convergence Divergence (MACD)
        data['MACD'] = data['EMA_20'] - data['EMA_50']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Commodity Channel Index (CCI)
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        data['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

        # Fill NA and compile indicators
        data.fillna(0, inplace=True)
        indicators = data[['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line', 'CCI']].values
        return indicators

    def predict(self, data, symbol):
        indicators = self.calculate_indicators(data)
        indicators_scaled = self.scalers[symbol].transform(indicators)
        predicted_indicator = self.models[symbol].predict(indicators_scaled[-1].reshape(1, -1))
        return predicted_indicator[0]

    def add_technical_indicators(self, data):
        """
        Adds technical indicators directly to the provided DataFrame.
        This method modifies the DataFrame in place.
        
        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data
        """
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['BB_upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
        data['BB_lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)

        # Moving Average Convergence Divergence (MACD)
        data['MACD'] = data['EMA_20'] - data['EMA_50']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Commodity Channel Index (CCI)
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        data['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

        # Fill NA values
        data.fillna(0, inplace=True)

    def build_model(self, sequence_length=60, n_features=1, lstm_units1=128, lstm_units2=64,
                  dense_units1=32, dropout_rate=0.3, learning_rate=0.001, bidirectional=True):
        """
        Builds a more sophisticated LSTM model for time series prediction with configurable parameters.
        
        Args:
            sequence_length (int): Number of timesteps in input sequences
            n_features (int): Number of features in input data
            lstm_units1 (int): Number of units in first LSTM layer
            lstm_units2 (int): Number of units in second LSTM layer
            dense_units1 (int): Number of units in first Dense layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            bidirectional (bool): Whether to use bidirectional LSTM layers
        
        Returns:
            model: Compiled Keras model
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
        from tensorflow.keras.layers import Attention, Concatenate, Input, RepeatVector, TimeDistributed
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.regularizers import l1_l2
        from tensorflow.keras.models import Model
        import tensorflow as tf
        
        # Use functional API for more flexibility
        inputs = Input(shape=(sequence_length, n_features))
        
        # First LSTM layer with batch normalization
        if bidirectional:
            x = Bidirectional(LSTM(lstm_units1, return_sequences=True, 
                                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                  recurrent_regularizer=l1_l2(l1=0, l2=1e-4),
                                  bias_regularizer=l1_l2(l1=0, l2=1e-4)))(inputs)
        else:
            x = LSTM(lstm_units1, return_sequences=True,
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    recurrent_regularizer=l1_l2(l1=0, l2=1e-4),
                    bias_regularizer=l1_l2(l1=0, l2=1e-4))(inputs)
        
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Second LSTM layer
        if bidirectional:
            x = Bidirectional(LSTM(lstm_units2, return_sequences=True))(x)
        else:
            x = LSTM(lstm_units2, return_sequences=True)(x)
        
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Attention mechanism
        # Self-attention layer to focus on most important time steps
        # Simple implementation of attention mechanism
        attention = Dense(1, activation='tanh')(x)
        attention = tf.squeeze(attention, -1)
        attention_weights = tf.nn.softmax(attention)
        context = tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
        
        # Dense layers for final prediction
        x = Dense(dense_units1, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(context)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate/2)(x)
        
        outputs = Dense(1)(x)
        
        # Assemble and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use Adam optimizer with configurable learning rate
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, 
                     loss='mean_squared_error',
                     metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
        
        return model

    def train_lstm(self, data, symbol, epochs=50, batch_size=32, validation_split=0.2, sequence_length=60):
        """
        Train an LSTM model on the time series data with improved training procedure
        
        Args:
            data (pd.DataFrame): DataFrame with at least a 'Close' column
            symbol (str): Trading symbol identifier
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Portion of data to use for validation
            sequence_length (int): Number of previous time steps to use for prediction
        """
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        import os
        
        # Scale the data
        self.scalers[symbol] = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scalers[symbol].fit_transform(data['Close'].values.reshape(-1,1))
        
        # Prepare training data
        X_train, y_train = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Build the model with default or custom parameters
        model = self.build_model(sequence_length=sequence_length)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
        ]
        
        # Create model checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join('models', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Add ModelCheckpoint callback if directory exists
        checkpoint_path = os.path.join(checkpoint_dir, f'{symbol}_lstm_model.h5')
        callbacks.append(ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ))
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            shuffle=True,
            verbose=1
        )
        
        # Store the model and training history
        self.models[symbol] = model
        return history
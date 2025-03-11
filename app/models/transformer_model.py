import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, 
    Concatenate, Conv1D, Embedding, Add
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FinancialTransformer:
    """
    Time series transformer model for financial forecasting.
    Uses attention mechanisms to capture temporal dependencies and patterns in market data.
    
    References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (Lim et al., 2020)
    """
    
    def __init__(
        self,
        input_sequence_length=30,
        forecast_horizon=5,
        d_model=64,
        num_heads=4,
        dropout_rate=0.1,
        dff=256,
        num_transformer_blocks=2,
        learning_rate=0.001
    ):
        """
        Initialize the Financial Transformer model.
        
        Args:
            input_sequence_length: Number of time steps for input sequence
            forecast_horizon: Number of time steps to predict
            d_model: Dimensionality of the transformer model
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
            dff: Dimensionality of the feedforward network inside transformer
            num_transformer_blocks: Number of transformer blocks to stack
            learning_rate: Learning rate for the Adam optimizer
        """
        self.input_sequence_length = input_sequence_length
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dff = dff
        self.num_transformer_blocks = num_transformer_blocks
        self.learning_rate = learning_rate
        
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        logger.info(f"Initialized FinancialTransformer with {num_transformer_blocks} transformer blocks")

    def _build_transformer_block(self, inputs):
        """Build a single transformer block."""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model
        )(inputs, inputs)
        
        # Add & normalize (first residual connection)
        attention_output = Add()([inputs, attention_output])
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        
        # Feed-forward network
        ffn_output = Dense(self.dff, activation='relu')(attention_output)
        ffn_output = Dense(self.d_model)(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        
        # Add & normalize (second residual connection)
        output = Add()([attention_output, ffn_output])
        output = LayerNormalization(epsilon=1e-6)(output)
        
        return output

    def build_model(self, num_features):
        """Build the transformer model architecture."""
        # Input layer
        inputs = Input(shape=(self.input_sequence_length, num_features))
        
        # Initial projection to d_model dimensions
        x = Conv1D(filters=self.d_model, kernel_size=1, activation='relu')(inputs)
        
        # Positional encoding (simple approach - could be enhanced)
        positions = tf.range(start=0, limit=self.input_sequence_length, delta=1)
        position_embedding = Embedding(
            input_dim=self.input_sequence_length, 
            output_dim=self.d_model
        )(positions)
        
        # Add positional encoding
        position_embedding = tf.expand_dims(position_embedding, axis=0)
        position_encoding = tf.tile(position_embedding, [tf.shape(x)[0], 1, 1])
        x = Add()([x, position_encoding])
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self._build_transformer_block(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers - one for each step in the forecast horizon
        outputs = []
        for i in range(self.forecast_horizon):
            output = Dense(32, activation='relu')(x)
            output = Dense(1, name=f'forecast_{i+1}')(output)
            outputs.append(output)
        
        # Combine all outputs
        if len(outputs) > 1:
            final_output = Concatenate(axis=1)(outputs)
        else:
            final_output = outputs[0]
        
        # Build and compile the model
        self.model = Model(inputs=inputs, outputs=final_output)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built transformer model with {self.model.count_params()} parameters")
        return self.model
    
    def _prepare_features(self, df):
        """Prepare features from raw market data."""
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Add technical indicators as features
        # Moving averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Volatility indicators
        data['ATR'] = (
            data['High'] - data['Low']).rolling(window=14).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
        
        # Price momentum
        data['ROC_5'] = (data['Close'] / data['Close'].shift(5) - 1) * 100
        data['ROC_10'] = (data['Close'] / data['Close'].shift(10) - 1) * 100
        
        # Volume indicators
        data['Volume_ROC'] = (data['Volume'] / data['Volume'].shift(1) - 1) * 100
        data['OBV'] = (data['Close'] > data['Close'].shift(1)).astype(int) * data['Volume']
        data['OBV'] = data['OBV'].cumsum()
        
        # Fill missing values created by indicators
        data = data.fillna(method='bfill')
        
        return data
    
    def _create_sequences(self, data, target_column='Close'):
        """Create input sequences and target values."""
        features = data.drop([target_column], axis=1)
        target = data[target_column]
        
        # Scale the features and target
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_target = self.target_scaler.fit_transform(target.values.reshape(-1, 1))
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(data) - self.input_sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(scaled_features[i:i+self.input_sequence_length])
            
            # Target sequence (future values)
            y_seq = scaled_target[i+self.input_sequence_length:i+self.input_sequence_length+self.forecast_horizon]
            y.append(y_seq.reshape(self.forecast_horizon))
        
        return np.array(X), np.array(y)
    
    def train(self, data, target_column='Close', epochs=100, batch_size=32, validation_split=0.2):
        """Train the transformer model on the provided data."""
        # Prepare features
        processed_data = self._prepare_features(data)
        
        # Create sequences
        X, y = self._create_sequences(processed_data, target_column)
        
        if X.shape[0] == 0:
            raise ValueError("Not enough data points to create sequences")
        
        # Build the model if it doesn't exist
        if self.model is None:
            self.build_model(X.shape[2])
        
        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info(f"Trained transformer model for {len(history.epoch)} epochs")
        return history
    
    def predict(self, data, target_column='Close'):
        """Generate predictions for the given data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Prepare features
        processed_data = self._prepare_features(data)
        
        # Get the latest data points for prediction
        latest_data = processed_data.iloc[-self.input_sequence_length:]
        
        # Prepare the input sequence
        features = latest_data.drop([target_column], axis=1)
        scaled_features = self.feature_scaler.transform(features)
        X = np.array([scaled_features])
        
        # Generate predictions
        scaled_predictions = self.model.predict(X)
        
        # Inverse transform to get actual values
        predictions = self.target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
        
        return predictions.flatten()
    
    def save(self, model_path, scaler_path):
        """Save the model and scalers."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save Keras model
        self.model.save(model_path)
        
        # Save scalers
        import joblib
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }, scaler_path)
        
        logger.info(f"Saved transformer model to {model_path}")
    
    def load(self, model_path, scaler_path):
        """Load the model and scalers."""
        # Load Keras model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scalers
        import joblib
        scalers = joblib.load(scaler_path)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        
        logger.info(f"Loaded transformer model from {model_path}")
        return self 
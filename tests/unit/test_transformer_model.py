import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from app.models.transformer_model import FinancialTransformer

class TestFinancialTransformer:
    """Test suite for the FinancialTransformer model"""
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial time series data"""
        # Generate 100 days of sample data
        date_range = pd.date_range(start='2022-01-01', periods=100, freq='D')
        
        # Create a sine wave pattern with some noise for price data
        t = np.linspace(0, 4*np.pi, 100)
        close_prices = 100 + 10 * np.sin(t) + np.random.normal(0, 1, 100)
        
        # Create other price columns based on close with some variations
        high_prices = close_prices + np.random.uniform(0, 2, 100)
        low_prices = close_prices - np.random.uniform(0, 2, 100)
        open_prices = close_prices.copy()
        np.random.shuffle(open_prices)
        
        # Create volume with some correlation to price changes
        volume = 10000 + 5000 * np.abs(np.diff(close_prices, prepend=close_prices[0])) + np.random.normal(0, 1000, 100)
        
        # Create the DataFrame
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=date_range)
        
        return df
    
    def test_transformer_initialization(self):
        """Test that the transformer model initializes correctly"""
        model = FinancialTransformer()
        
        # Check default parameters
        assert model.input_sequence_length == 30
        assert model.forecast_horizon == 5
        assert model.d_model == 64
        assert model.num_heads == 4
        assert model.dropout_rate == 0.1
        assert model.dff == 256
        assert model.num_transformer_blocks == 2
        assert model.learning_rate == 0.001
        
        # Check custom parameters
        custom_model = FinancialTransformer(
            input_sequence_length=20,
            forecast_horizon=3,
            d_model=32,
            num_heads=2
        )
        
        assert custom_model.input_sequence_length == 20
        assert custom_model.forecast_horizon == 3
        assert custom_model.d_model == 32
        assert custom_model.num_heads == 2
    
    def test_feature_engineering(self, sample_financial_data):
        """Test that feature engineering creates expected features"""
        model = FinancialTransformer()
        
        # Apply feature engineering
        features_df = model._prepare_features(sample_financial_data)
        
        # Check that basic features are preserved
        assert 'Open' in features_df.columns
        assert 'High' in features_df.columns
        assert 'Low' in features_df.columns
        assert 'Close' in features_df.columns
        assert 'Volume' in features_df.columns
        
        # Check for engineered features (actual names may vary depending on implementation)
        # We expect at least more columns than the original data
        assert len(features_df.columns) > 5
        
        # Check for specific technical indicators commonly used
        technical_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'ATR']
        indicator_found = False
        
        for indicator in technical_indicators:
            if any(indicator in col for col in features_df.columns):
                indicator_found = True
                break
                
        assert indicator_found, "No technical indicators found in feature engineering output"
    
    def test_sequence_creation(self, sample_financial_data):
        """Test sequence creation for time series data"""
        model = FinancialTransformer(
            input_sequence_length=10,
            forecast_horizon=2
        )
        
        # First prepare features
        features_df = model._prepare_features(sample_financial_data)
        
        # Create sequences
        X, y = model._create_sequences(features_df)
        
        # Check dimensions
        assert len(X.shape) == 3  # (samples, time steps, features)
        assert X.shape[1] == 10  # input_sequence_length
        assert y.shape[1] == 2  # forecast_horizon
        
        # Check that we have the expected number of samples
        expected_samples = len(features_df) - 10 - 2 + 1
        assert X.shape[0] == expected_samples
        assert y.shape[0] == expected_samples
    
    def test_model_building(self):
        """Test model building functionality"""
        model = FinancialTransformer()
        
        # Build model with 10 features
        keras_model = model.build_model(10)
        
        # Check model structure
        assert keras_model is not None
        assert len(keras_model.layers) > 0
        
        # Check input shape
        expected_input_shape = (None, model.input_sequence_length, 10)
        assert keras_model.input_shape == expected_input_shape
        
        # Check output shape
        expected_output_shape = (None, model.forecast_horizon)
        assert keras_model.output_shape == expected_output_shape
    
    def test_predict_functionality(self, sample_financial_data):
        """Test prediction functionality with minimal training"""
        
        # Use small model and data subset for quick testing
        model = FinancialTransformer(
            input_sequence_length=5,
            forecast_horizon=1,
            d_model=8,
            num_heads=1,
            num_transformer_blocks=1
        )
        
        # Use only first 50 rows for quicker testing
        data_subset = sample_financial_data.iloc[:50]
        
        # Train with minimal epochs
        model.train(data_subset, epochs=1, batch_size=16, validation_split=0.1)
        
        # Make predictions
        predictions = model.predict(data_subset)
        
        # Check that predictions are returned
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        
        # Check dimensions: should predict forecast_horizon values for each possible window
        expected_len = len(data_subset) - model.input_sequence_length - model.forecast_horizon + 1
        assert len(predictions) == expected_len
    
    def test_save_load(self, sample_financial_data):
        """Test model saving and loading"""
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "transformer_model.h5")
            scaler_path = os.path.join(temp_dir, "transformer_scaler.pkl")
            
            # Initialize and train a simple model
            model = FinancialTransformer(
                input_sequence_length=5,
                forecast_horizon=1,
                d_model=8,
                num_heads=1,
                num_transformer_blocks=1
            )
            
            # Use only first 30 rows for quicker testing
            data_subset = sample_financial_data.iloc[:30]
            
            # Train with minimal epochs
            model.train(data_subset, epochs=1, batch_size=8, validation_split=0.1)
            
            # Save the model
            model.save(model_path, scaler_path)
            
            # Check that files exist
            assert os.path.exists(model_path)
            assert os.path.exists(scaler_path)
            
            # Create a new model and load the saved weights
            loaded_model = FinancialTransformer(
                input_sequence_length=5,
                forecast_horizon=1,
                d_model=8,
                num_heads=1,
                num_transformer_blocks=1
            )
            loaded_model.load(model_path, scaler_path)
            
            # Check that the model is loaded and can make predictions
            predictions = loaded_model.predict(data_subset)
            assert predictions is not None
            assert isinstance(predictions, np.ndarray) 
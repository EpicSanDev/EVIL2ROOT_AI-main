import pytest
import numpy as np
import pandas as pd
from app.models.price_prediction import PricePredictionModel

class TestPricePrediction:
    """Test the price prediction model functionality."""
    
    def test_model_initialization(self):
        """Test that the model initializes correctly with default parameters."""
        model = PricePredictionModel()
        assert model is not None
        assert hasattr(model, 'build_model')
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering process creates expected features."""
        model = PricePredictionModel()
        features = model._prepare_features(sample_data)
        
        # Check that features were created
        assert features is not None
        assert isinstance(features, pd.DataFrame)
        
        # Check that all original columns are preserved
        for col in sample_data.columns:
            assert col in features.columns
        
        # Check that technical indicators were added
        assert 'SMA_20' in features.columns
        assert 'EMA_12' in features.columns
        assert 'RSI_14' in features.columns
        
    def test_label_creation(self, sample_data):
        """Test that prediction labels are created correctly."""
        model = PricePredictionModel()
        features = model._prepare_features(sample_data)
        X, y = model._create_training_data(features, prediction_horizon=5)
        
        # Check that X and y have correct shapes
        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == 1  # Single target variable
        
        # Ensure y contains future prices (cannot directly check values, but can check type)
        assert isinstance(y, np.ndarray)
    
    def test_predict_returns_valid_predictions(self, sample_data):
        """Test that predict method returns valid predictions."""
        model = PricePredictionModel()
        
        # Mock the model's predict_keras method to avoid actual model training
        def mock_predict(X):
            return np.random.rand(X.shape[0], 1)
        
        model.model = type('obj', (object,), {'predict': mock_predict})
        
        # Test prediction
        predictions = model.predict(sample_data)
        
        # Check that predictions have the expected format
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
        
        # Check that predictions are within a reasonable range
        # For stock price predictions, values should typically be positive
        assert np.all(predictions >= 0) 
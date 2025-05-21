import numpy as np
import pandas as pd
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Explainable AI libraries
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance

# Import other existing models
from app.models.price_prediction import PricePredictionModel
from app.models.indicator_management import IndicatorManagementModel
from app.models.transformer_model import TransformerModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models
    with advanced feature engineering and explainable AI capabilities.
    """
    
    def __init__(self, 
                 model_dir='saved_models',
                 ensemble_type='stacking',  # 'voting', 'stacking', 'bagging', 'boosting'
                 use_shap=True,
                 use_lime=False,
                 model_weights=None, 
                 sequence_length=60):
        """
        Initialize the ensemble model.
        
        Args:
            model_dir: Directory to save model artifacts
            ensemble_type: Type of ensemble ('voting', 'stacking', 'bagging', 'boosting')
            use_shap: Whether to use SHAP for model explanations
            use_lime: Whether to use LIME for model explanations
            model_weights: Optional weights for component models in voting ensemble
            sequence_length: Sequence length for time series data
        """
        self.model_dir = model_dir
        self.ensemble_type = ensemble_type
        self.use_shap = use_shap
        self.use_lime = use_lime
        self.model_weights = model_weights
        self.sequence_length = sequence_length
        
        # Component models
        self.models = {}
        self.base_models = []
        
        # Ensemble model
        self.ensemble = None
        
        # Feature importances and explanations
        self.feature_importances = {}
        self.explainer = None
        
        # Scalers
        self.scalers = {}
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        logger.info(f"Initialized EnsembleModel with {ensemble_type} approach")
    
    def add_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical and fundamental features to the dataset.
        
        Args:
            data: DataFrame with OHLCV and existing features
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Adding advanced feature engineering")
        df = data.copy()
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Required column {col} not in data, some features may not be created")
        
        try:
            # === 1. Technical Indicators ===
            
            # 1.1 Trend Indicators
            # Moving Average Crossover signals
            for fast, slow in [(5, 20), (10, 50), (20, 100)]:
                if 'Close' in df.columns:
                    df[f'SMA_{fast}'] = df['Close'].rolling(window=fast).mean()
                    df[f'SMA_{slow}'] = df['Close'].rolling(window=slow).mean()
                    df[f'SMA_crossover_{fast}_{slow}'] = np.where(df[f'SMA_{fast}'] > df[f'SMA_{slow}'], 1, -1)
            
            # Average Directional Index (ADX) - Trend strength indicator
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                # Calculate +DM and -DM
                high_diff = df['High'].diff()
                low_diff = df['Low'].diff().abs() * -1
                
                plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
                minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
                
                # Calculate ATR as basic implementation
                tr1 = df['High'] - df['Low']
                tr2 = abs(df['High'] - df['Close'].shift(1))
                tr3 = abs(df['Low'] - df['Close'].shift(1))
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr14 = tr.rolling(window=14).mean()
                
                # Calculate +DI and -DI
                plus_di14 = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr14
                minus_di14 = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr14
                
                # Calculate DX
                dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
                
                # Calculate ADX
                df['ADX'] = dx.rolling(window=14).mean()
            
            # 1.2 Momentum Indicators
            if 'Close' in df.columns:
                # Rate of Change (RoC)
                for period in [5, 10, 20]:
                    df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
                
                # Momentum
                for period in [5, 10, 20]:
                    df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
                
                # Stochastic Oscillator
                if 'High' in df.columns and 'Low' in df.columns:
                    for period in [14, 21]:
                        lowest_low = df['Low'].rolling(window=period).min()
                        highest_high = df['High'].rolling(window=period).max()
                        df[f'K_{period}'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
                        df[f'D_{period}'] = df[f'K_{period}'].rolling(window=3).mean()
            
            # 1.3 Volatility Indicators
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                # Average True Range (ATR)
                for period in [7, 14, 21]:
                    df[f'ATR_{period}'] = tr.rolling(window=period).mean()
                
                # Bollinger Band Normalized (BBN) - Distance from price to middle band in stdevs
                for period in [20, 50]:
                    middle_band = df['Close'].rolling(window=period).mean()
                    std_dev = df['Close'].rolling(window=period).std()
                    df[f'BBN_{period}'] = (df['Close'] - middle_band) / (2 * std_dev)
                
                # Bollinger Band Width (BBW) - Indicates volatility
                for period in [20, 50]:
                    middle_band = df['Close'].rolling(window=period).mean()
                    std_dev = df['Close'].rolling(window=period).std()
                    df[f'BBW_{period}'] = (2 * std_dev) / middle_band
            
            # 1.4 Volume-based Indicators
            if all(col in df.columns for col in ['Close', 'Volume']):
                # On-Balance Volume (OBV)
                df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
                
                # Chaikin Money Flow (CMF)
                if all(col in df.columns for col in ['High', 'Low']):
                    for period in [20, 50]:
                        money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
                        money_flow_volume = money_flow_multiplier * df['Volume']
                        df[f'CMF_{period}'] = money_flow_volume.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
                
                # Accumulation/Distribution Line (ADL)
                if all(col in df.columns for col in ['High', 'Low']):
                    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
                    money_flow_volume = money_flow_multiplier * df['Volume']
                    df['ADL'] = money_flow_volume.cumsum()
            
            # 1.5 Cycle Indicators and Pattern Recognition
            if 'Close' in df.columns:
                # Price Patterns
                # Head and Shoulders pattern indicator (simplified)
                df['PriceShift1'] = df['Close'].shift(1)
                df['PriceShift2'] = df['Close'].shift(2)
                df['PriceShift3'] = df['Close'].shift(3)
                df['PriceShift4'] = df['Close'].shift(4)
                df['PriceShift5'] = df['Close'].shift(5)
                
                # Candlestick pattern features
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    # Hammer/Hanging Man pattern (simplified)
                    df['BodySize'] = abs(df['Close'] - df['Open'])
                    df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
                    df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
                    df['Hammer'] = ((df['LowerShadow'] > 2 * df['BodySize']) & (df['UpperShadow'] < 0.2 * df['BodySize'])).astype(int)
                    
                    # Doji pattern (simplified)
                    df['Doji'] = (df['BodySize'] < 0.1 * (df['High'] - df['Low'])).astype(int)
            
            # === 2. Financial and Macroeconomic Features ===
            # These would typically come from external sources
            # For now, we'll add placeholder columns that can be populated later
            # with actual economic data or derived from external APIs
            df['MarketRegime'] = 0  # Placeholder for market regime classification
            df['EconomicSurpriseIndex'] = 0  # Placeholder for economic surprise index
            df['MarketVolatilityIndex'] = 0  # Placeholder for VIX or similar
            
            # === 3. Feature Transformations and Feature Crosses ===
            
            # 3.1 Lagged Features (for time series relationships)
            if 'Close' in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                    df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)
            
            # 3.2 Moving Statistics (beyond simple moving averages)
            if 'Close' in df.columns:
                # Moving standard deviation (volatility)
                for window in [5, 10, 20]:
                    df[f'Close_std_{window}'] = df['Close'].rolling(window=window).std()
                
                # Moving kurtosis and skewness
                for window in [20, 50]:
                    df[f'Close_kurt_{window}'] = df['Close'].rolling(window=window).kurt()
                    df[f'Close_skew_{window}'] = df['Close'].rolling(window=window).skew()
            
            # 3.3 Feature Crosses (interactions between features)
            # Example: RSI × Bollinger Band Width (combines momentum and volatility)
            if all(x in df.columns for x in ['RSI_14', 'BBW_20']):
                df['RSI_BBW_Cross'] = df['RSI_14'] * df['BBW_20']
            
            # 3.4 Fourier Transform Features for cyclical patterns
            if 'Close' in df.columns and len(df) >= 128:  # Need sufficient data
                try:
                    from scipy.fftpack import fft
                    
                    # Apply FFT to close prices
                    close_fft = fft(df['Close'].values)
                    close_fft = np.abs(close_fft[:30])  # Take first 30 components
                    
                    # Add a few dominant frequencies as features
                    for i in range(1, min(5, len(close_fft))):
                        df[f'FFT_{i}'] = close_fft[i]
                except Exception as e:
                    logger.warning(f"Could not calculate FFT features: {e}")
            
            # Fill NaN values (especially important for features with lookback windows)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)
            
            logger.info(f"Added {len(df.columns) - len(data.columns)} new features through engineering")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            # Return original data in case of error
            return data
    
    def build_models(self, data: pd.DataFrame, target_col: str, classification=False):
        """
        Build component models for ensemble.
        
        Args:
            data: Training data with features
            target_col: Column name of the target variable
            classification: Whether this is a classification task
        """
        logger.info(f"Building {'classification' if classification else 'regression'} models for ensemble")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Save feature names for explanations
        self.feature_names = X.columns.tolist()
        
        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize base models based on classification or regression
        if classification:
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
                ('lgbm', LGBMClassifier(n_estimators=100, random_state=42)),
                ('cat', CatBoostClassifier(n_estimators=100, random_state=42, verbose=0))
            ]
            
            # Final model for stacking
            final_model = LogisticRegression()
        else:
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
                ('lgbm', LGBMRegressor(n_estimators=100, random_state=42)),
                ('cat', CatBoostRegressor(n_estimators=100, random_state=42, verbose=0))
            ]
            
            # Final model for stacking
            final_model = Ridge()
        
        # Create the ensemble based on the specified type
        if self.ensemble_type == 'voting':
            if classification:
                self.ensemble = VotingClassifier(
                    estimators=base_models,
                    voting='soft',
                    weights=self.model_weights
                )
            else:
                self.ensemble = VotingRegressor(
                    estimators=base_models,
                    weights=self.model_weights
                )
        elif self.ensemble_type == 'stacking':
            # Create a stacking ensemble
            if classification:
                self.ensemble = StackingClassifier(
                    estimators=base_models,
                    final_estimator=final_model,
                    cv=tscv
                )
            else:
                self.ensemble = StackingRegressor(
                    estimators=base_models,
                    final_estimator=final_model,
                    cv=tscv
                )
        else:
            # Default to using a single best model (RandomForest)
            logger.warning(f"Ensemble type {self.ensemble_type} not recognized, using RandomForest")
            if classification:
                self.ensemble = RandomForestClassifier(n_estimators=200, random_state=42)
            else:
                self.ensemble = RandomForestRegressor(n_estimators=200, random_state=42)
        
        # Store the base models for later analysis
        self.base_models = base_models
                
    def train(self, data: pd.DataFrame, symbol: str, target_col: str, classification=False):
        """
        Train the ensemble model on the provided data.
        
        Args:
            data: Training data with features
            symbol: Trading symbol
            target_col: Column name of the target variable
            classification: Whether this is a classification task
        """
        logger.info(f"Training ensemble model for {symbol}")
        
        try:
            # Add engineered features
            enhanced_data = self.add_feature_engineering(data)
            
            # Scale features
            self.scalers[symbol] = StandardScaler()
            feature_cols = [col for col in enhanced_data.columns if col != target_col]
            
            enhanced_data_scaled = enhanced_data.copy()
            enhanced_data_scaled[feature_cols] = self.scalers[symbol].fit_transform(enhanced_data[feature_cols])
            
            # Build models if not already built
            if self.ensemble is None:
                self.build_models(enhanced_data_scaled, target_col, classification)
            
            # Split data into features and target
            X = enhanced_data_scaled.drop(columns=[target_col])
            y = enhanced_data_scaled[target_col]
            
            # Fit the ensemble model
            self.ensemble.fit(X, y)
            
            # Generate feature importances
            self.calculate_feature_importance(X, y, symbol)
            
            # Initialize SHAP explainer if enabled
            if self.use_shap:
                try:
                    # Choose appropriate explainer based on model type
                    if hasattr(self.ensemble, 'estimators_'):
                        # For voting or stacking ensemble, use the first base model
                        model_to_explain = self.ensemble.estimators_[0]
                    elif hasattr(self.ensemble, 'estimators'):
                        # For some scikit-learn ensembles
                        model_to_explain = self.ensemble.estimators[0]
                    else:
                        # Direct model
                        model_to_explain = self.ensemble
                    
                    # Create SHAP explainer
                    self.explainer = shap.Explainer(model_to_explain, X)
                    logger.info(f"SHAP explainer initialized for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not initialize SHAP explainer: {e}")
                    self.explainer = None
            
            # Save the model
            self.save(symbol)
            logger.info(f"Successfully trained ensemble model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error training ensemble model for {symbol}: {e}")
            return False
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series, symbol: str):
        """
        Calculate feature importance from the trained model.
        
        Args:
            X: Feature matrix
            y: Target variable
            symbol: Trading symbol
        """
        importances = {}
        
        try:
            # 1. Get feature importances from the model if available
            if hasattr(self.ensemble, 'feature_importances_'):
                importances['model'] = dict(zip(X.columns, self.ensemble.feature_importances_))
            
            # 2. Calculate permutation importance
            try:
                perm_importance = permutation_importance(
                    self.ensemble, X, y, n_repeats=10, random_state=42
                )
                importances['permutation'] = dict(zip(X.columns, perm_importance.importances_mean))
            except Exception as e:
                logger.warning(f"Could not calculate permutation importance: {e}")
            
            # Store the feature importances
            self.feature_importances[symbol] = importances
            
            logger.info(f"Calculated feature importance for {symbol}")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
    
    def predict(self, data: pd.DataFrame, symbol: str):
        """
        Make predictions using the ensemble model.
        
        Args:
            data: Data with features
            symbol: Trading symbol
            
        Returns:
            Predictions and explanations
        """
        logger.info(f"Making ensemble predictions for {symbol}")
        
        try:
            # Check if model and scaler exist
            if symbol not in self.scalers or self.ensemble is None:
                logger.error(f"Model or scaler not found for {symbol}")
                return None, None
            
            # Add engineered features
            enhanced_data = self.add_feature_engineering(data)
            
            # Scale features
            feature_cols = [col for col in enhanced_data.columns if col in self.scalers[symbol].feature_names_in_]
            enhanced_data_scaled = enhanced_data.copy()
            enhanced_data_scaled[feature_cols] = self.scalers[symbol].transform(enhanced_data[feature_cols][feature_cols])
            
            # Make predictions
            X = enhanced_data_scaled[feature_cols]
            predictions = self.ensemble.predict(X)
            
            # Generate explanations if SHAP is enabled
            explanations = None
            if self.use_shap and self.explainer is not None:
                try:
                    # Get SHAP values for the latest data point
                    shap_values = self.explainer(X.iloc[-10:])  # Explain last 10 points
                    
                    # Structure explanations
                    explanations = {
                        'shap_values': shap_values.values,
                        'feature_names': X.columns.tolist(),
                        'base_value': shap_values.base_values[0],
                        'shap_plots': self._generate_shap_plots(shap_values, X)
                    }
                except Exception as e:
                    logger.warning(f"Could not generate SHAP explanations: {e}")
            
            return predictions, explanations
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return None, None
    
    def _generate_shap_plots(self, shap_values, X):
        """
        Generate explanation plots using SHAP values.
        
        Args:
            shap_values: SHAP values
            X: Feature matrix
            
        Returns:
            Dictionary of plot data
        """
        plots = {}
        try:
            # Save waterfall plot for the most recent prediction
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[-1], max_display=10, show=False)
            plt.tight_layout()
            waterfall_path = os.path.join(self.model_dir, 'waterfall_plot.png')
            plt.savefig(waterfall_path)
            plt.close()
            plots['waterfall_path'] = waterfall_path
            
            # Save beeswarm plot for feature importance
            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.tight_layout()
            beeswarm_path = os.path.join(self.model_dir, 'beeswarm_plot.png')
            plt.savefig(beeswarm_path)
            plt.close()
            plots['beeswarm_path'] = beeswarm_path
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP plots: {e}")
        
        return plots
    
    def get_top_features(self, symbol: str, n=10):
        """
        Get the top N most important features for a symbol.
        
        Args:
            symbol: Trading symbol
            n: Number of top features to return
            
        Returns:
            Dictionary of top features and their importance
        """
        if symbol not in self.feature_importances:
            logger.warning(f"No feature importance data for {symbol}")
            return {}
        
        try:
            # Get permutation importance if available, else fall back to model importance
            if 'permutation' in self.feature_importances[symbol]:
                importances = self.feature_importances[symbol]['permutation']
            elif 'model' in self.feature_importances[symbol]:
                importances = self.feature_importances[symbol]['model']
            else:
                return {}
            
            # Sort features by importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N
            return dict(sorted_features[:n])
            
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            return {}
    
    def generate_explanation_report(self, data: pd.DataFrame, symbol: str, save_path=None):
        """
        Generate a comprehensive explanation report for the model.
        
        Args:
            data: Latest data with features
            symbol: Trading symbol
            save_path: Path to save the report
            
        Returns:
            Dictionary with explanation data
        """
        if not save_path:
            save_path = os.path.join(self.model_dir, f"{symbol}_explanation_report.txt")
        
        report = {
            'top_features': self.get_top_features(symbol),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol
        }
        
        try:
            # Add engineered features
            enhanced_data = self.add_feature_engineering(data)
            
            # Scale features
            if symbol in self.scalers:
                feature_cols = [col for col in enhanced_data.columns if col in self.scalers[symbol].feature_names_in_]
                enhanced_data_scaled = enhanced_data.copy()
                enhanced_data_scaled[feature_cols] = self.scalers[symbol].transform(enhanced_data[feature_cols][feature_cols])
                X = enhanced_data_scaled[feature_cols]
                
                # Generate predictions and explanations
                predictions, explanations = self.predict(data, symbol)
                
                if predictions is not None:
                    report['predictions'] = predictions[-5:].tolist()  # Last 5 predictions
                
                if explanations is not None:
                    report['explanation_plots'] = explanations.get('shap_plots', {})
                    
                    # Add textual explanation of the top contributing features
                    last_shap = explanations.get('shap_values', [])[-1]
                    features = explanations.get('feature_names', [])
                    
                    if len(last_shap) > 0 and len(features) > 0:
                        # Get top 5 features by absolute SHAP value
                        feature_effects = [(features[i], last_shap[i]) for i in range(len(features))]
                        feature_effects.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        top_effects = feature_effects[:5]
                        report['feature_effects'] = top_effects
                        
                        # Generate natural language explanation
                        explanation_text = f"Prediction for {symbol} is influenced most by:\n"
                        
                        for feature, effect in top_effects:
                            direction = "increased" if effect > 0 else "decreased"
                            explanation_text += f"- {feature}: {direction} prediction by {abs(effect):.4f}\n"
                        
                        report['explanation_text'] = explanation_text
            
            # Write report to file
            with open(save_path, 'w') as f:
                f.write(f"Explanation Report for {symbol}\n")
                f.write(f"Generated at: {report['timestamp']}\n\n")
                
                f.write("Top Important Features:\n")
                for feature, importance in report.get('top_features', {}).items():
                    f.write(f"- {feature}: {importance:.4f}\n")
                
                f.write("\n")
                
                if 'explanation_text' in report:
                    f.write(report['explanation_text'])
                
                f.write("\n\nPrediction plots saved to the model directory.\n")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return {'error': str(e)}
    
    def save(self, symbol):
        """
        Save the ensemble model and related components.
        
        Args:
            symbol: Trading symbol
        """
        try:
            # Create directory if it doesn't exist
            symbol_dir = os.path.join(self.model_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save ensemble model
            model_path = os.path.join(symbol_dir, 'ensemble_model.joblib')
            joblib.dump(self.ensemble, model_path)
            
            # Save scaler
            if symbol in self.scalers:
                scaler_path = os.path.join(symbol_dir, 'scaler.joblib')
                joblib.dump(self.scalers[symbol], scaler_path)
            
            # Save feature importances
            if symbol in self.feature_importances:
                importance_path = os.path.join(symbol_dir, 'feature_importances.joblib')
                joblib.dump(self.feature_importances[symbol], importance_path)
            
            logger.info(f"Saved ensemble model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ensemble model for {symbol}: {e}")
            return False
    
    def load(self, symbol):
        """
        Load the ensemble model and related components.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check if model exists
            symbol_dir = os.path.join(self.model_dir, symbol)
            model_path = os.path.join(symbol_dir, 'ensemble_model.joblib')
            
            if not os.path.exists(model_path):
                logger.warning(f"No ensemble model found for {symbol}")
                return False
            
            # Load ensemble model
            self.ensemble = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(symbol_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scalers[symbol] = joblib.load(scaler_path)
            
            # Load feature importances
            importance_path = os.path.join(symbol_dir, 'feature_importances.joblib')
            if os.path.exists(importance_path):
                self.feature_importances[symbol] = joblib.load(importance_path)
            
            logger.info(f"Loaded ensemble model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble model for {symbol}: {e}")
            return False 
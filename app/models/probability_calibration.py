import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import matplotlib.pyplot as plt
import os
import joblib
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """
    Advanced probability calibration for machine learning models.
    
    This class implements several probability calibration methods to improve
    the reliability of confidence scores from machine learning models:
    
    1. Platt Scaling: A parametric approach using logistic regression
    2. Isotonic Regression: A non-parametric approach using isotonic regression
    3. Temperature Scaling: A simple method that scales the logits
    4. Beta Calibration: A more flexible parametric approach
    5. Ensemble Calibration: Combining multiple calibration methods
    
    These methods transform raw model outputs into well-calibrated probability
    estimates that can be reliably interpreted as confidence levels.
    """
    
    def __init__(self, 
                 method='ensemble',  # 'platt', 'isotonic', 'temperature', 'beta', 'ensemble'
                 models_dir='saved_models/calibration'):
        """
        Initialize the probability calibrator.
        
        Args:
            method: Calibration method to use
            models_dir: Directory for saving calibration models
        """
        self.method = method
        self.models_dir = models_dir
        self.calibrators = {}
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"Initialized ProbabilityCalibrator with method: {method}")
    
    def fit(self, symbol, predictions, true_values, bins=15):
        """
        Fit calibration model for the specified symbol.
        
        Args:
            symbol: Trading symbol
            predictions: Raw model predictions (probabilities or scores)
            true_values: Actual values (0/1 for classification, continuous for regression)
            bins: Number of bins for Beta calibration and diagnostics
            
        Returns:
            Calibration metrics
        """
        # Initialize calibration models for this symbol
        if symbol not in self.calibrators:
            self.calibrators[symbol] = {}
        
        # Store the raw predictions and true values for diagnostics
        self.calibrators[symbol]['raw_predictions'] = predictions
        self.calibrators[symbol]['true_values'] = true_values
        
        # Fit different calibration models
        if self.method in ['platt', 'ensemble']:
            self._fit_platt_scaling(symbol, predictions, true_values)
            
        if self.method in ['isotonic', 'ensemble']:
            self._fit_isotonic_regression(symbol, predictions, true_values)
            
        if self.method in ['temperature', 'ensemble']:
            self._fit_temperature_scaling(symbol, predictions, true_values)
            
        if self.method in ['beta', 'ensemble']:
            self._fit_beta_calibration(symbol, predictions, true_values, bins)
        
        # Calculate calibration metrics
        metrics = self.evaluate_calibration(symbol, predictions, true_values)
        logger.info(f"Calibration model fitted for {symbol} with metrics: {metrics}")
        
        return metrics
    
    def _fit_platt_scaling(self, symbol, predictions, true_values):
        """Fit Platt scaling (logistic regression) calibration."""
        # Reshape predictions if necessary
        if len(predictions.shape) == 1:
            X = predictions.reshape(-1, 1)
        else:
            X = predictions
            
        # Fit logistic regression
        calibrator = LogisticRegression(solver='lbfgs')
        calibrator.fit(X, true_values)
        
        # Store calibrator
        self.calibrators[symbol]['platt'] = calibrator
        logger.info(f"Fitted Platt scaling for {symbol}")
    
    def _fit_isotonic_regression(self, symbol, predictions, true_values):
        """Fit Isotonic Regression calibration."""
        # Reshape predictions if necessary
        if len(predictions.shape) == 1:
            X = predictions
        else:
            X = predictions.ravel()
            
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X, true_values)
        
        # Store calibrator
        self.calibrators[symbol]['isotonic'] = calibrator
        logger.info(f"Fitted Isotonic Regression for {symbol}")
    
    def _fit_temperature_scaling(self, symbol, predictions, true_values):
        """Fit Temperature Scaling calibration."""
        # Initial temperature parameter
        temperature = tf.Variable(1.0, dtype=tf.float32)
        
        # Define loss function
        def temperature_loss(temp):
            # Apply temperature scaling
            scaled_predictions = predictions / temp
            # Negative log likelihood loss
            nll = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    true_values, 
                    tf.sigmoid(scaled_predictions)
                )
            )
            return nll
        
        # Optimize temperature parameter
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        
        for _ in range(100):
            with tf.GradientTape() as tape:
                loss = temperature_loss(temperature)
            
            gradients = tape.gradient(loss, [temperature])
            optimizer.apply_gradients(zip(gradients, [temperature]))
            
            # Ensure temperature is positive
            temperature.assign(tf.maximum(0.1, temperature))
        
        # Store calibrator
        self.calibrators[symbol]['temperature'] = temperature.numpy()
        logger.info(f"Fitted Temperature Scaling for {symbol} with T={temperature.numpy():.4f}")
    
    def _fit_beta_calibration(self, symbol, predictions, true_values, bins=15):
        """
        Fit Beta Calibration model.
        
        This is a more flexible parametric approach that can handle
        various shapes of calibration curves, especially useful for
        extreme probabilities.
        """
        # Reshape predictions if necessary
        if len(predictions.shape) == 1:
            X = predictions.reshape(-1, 1)
        else:
            X = predictions
        
        # Transform predictions to avoid 0 and 1
        epsilon = 1e-9
        X_transformed = np.log(X + epsilon) - np.log(1 - X + epsilon)
        
        # Fit logistic regression with the transformed values
        calibrator = LogisticRegression(solver='lbfgs', C=1.0)
        calibrator.fit(X_transformed, true_values)
        
        # Store calibrator and parameters
        self.calibrators[symbol]['beta'] = calibrator
        self.calibrators[symbol]['beta_params'] = {
            'a': calibrator.coef_[0][0],
            'b': calibrator.intercept_[0]
        }
        
        logger.info(f"Fitted Beta Calibration for {symbol}")
    
    def calibrate(self, symbol, predictions):
        """
        Calibrate raw model predictions into reliable probability estimates.
        
        Args:
            symbol: Trading symbol
            predictions: Raw model predictions to calibrate
            
        Returns:
            Calibrated probability estimates
        """
        if symbol not in self.calibrators:
            logger.warning(f"No calibration model for {symbol}, returning raw predictions")
            return predictions
        
        # Reshape predictions if necessary
        if len(predictions.shape) == 1:
            X = predictions.reshape(-1, 1)
        else:
            X = predictions
        
        # Apply the appropriate calibration method
        if self.method == 'platt':
            calibrated = self._apply_platt_scaling(symbol, X)
        elif self.method == 'isotonic':
            calibrated = self._apply_isotonic_regression(symbol, X)
        elif self.method == 'temperature':
            calibrated = self._apply_temperature_scaling(symbol, X)
        elif self.method == 'beta':
            calibrated = self._apply_beta_calibration(symbol, X)
        elif self.method == 'ensemble':
            # Combine multiple calibration methods
            calibrated_platt = self._apply_platt_scaling(symbol, X)
            calibrated_isotonic = self._apply_isotonic_regression(symbol, X)
            calibrated_temp = self._apply_temperature_scaling(symbol, X)
            calibrated_beta = self._apply_beta_calibration(symbol, X)
            
            # Average the calibrated probabilities
            calibrated = (calibrated_platt + calibrated_isotonic + calibrated_temp + calibrated_beta) / 4.0
        else:
            logger.warning(f"Unknown calibration method: {self.method}, returning raw predictions")
            calibrated = predictions
        
        return calibrated
    
    def _apply_platt_scaling(self, symbol, X):
        """Apply Platt scaling to predictions."""
        if 'platt' not in self.calibrators[symbol]:
            logger.warning(f"No Platt scaling model for {symbol}")
            return X.ravel() if len(X.shape) > 1 else X
        
        calibrator = self.calibrators[symbol]['platt']
        return calibrator.predict_proba(X)[:, 1]
    
    def _apply_isotonic_regression(self, symbol, X):
        """Apply Isotonic regression to predictions."""
        if 'isotonic' not in self.calibrators[symbol]:
            logger.warning(f"No Isotonic regression model for {symbol}")
            return X.ravel() if len(X.shape) > 1 else X
        
        calibrator = self.calibrators[symbol]['isotonic']
        predictions = X.ravel() if len(X.shape) > 1 else X
        return calibrator.transform(predictions)
    
    def _apply_temperature_scaling(self, symbol, X):
        """Apply Temperature scaling to predictions."""
        if 'temperature' not in self.calibrators[symbol]:
            logger.warning(f"No Temperature scaling model for {symbol}")
            return X.ravel() if len(X.shape) > 1 else X
        
        temperature = self.calibrators[symbol]['temperature']
        predictions = X.ravel() if len(X.shape) > 1 else X
        return expit(predictions / temperature)
    
    def _apply_beta_calibration(self, symbol, X):
        """Apply Beta calibration to predictions."""
        if 'beta' not in self.calibrators[symbol]:
            logger.warning(f"No Beta calibration model for {symbol}")
            return X.ravel() if len(X.shape) > 1 else X
        
        # Transform predictions
        epsilon = 1e-9
        predictions = X.ravel() if len(X.shape) > 1 else X
        X_transformed = np.log(predictions + epsilon) - np.log(1 - predictions + epsilon)
        
        # Apply calibration
        calibrator = self.calibrators[symbol]['beta']
        return calibrator.predict_proba(X_transformed.reshape(-1, 1))[:, 1]
    
    def evaluate_calibration(self, symbol, predictions, true_values, n_bins=10):
        """
        Evaluate the calibration quality using various metrics.
        
        Args:
            symbol: Trading symbol
            predictions: Raw model predictions
            true_values: Actual values
            n_bins: Number of bins for binning the predictions
            
        Returns:
            Dictionary with calibration metrics
        """
        # Compute reliability diagram (binned calibration curve)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_sums = np.bincount(bin_indices, weights=true_values, minlength=n_bins)
        bin_means = np.zeros(n_bins)
        
        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_means[i] = bin_sums[i] / bin_counts[i]
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0
        for i in range(n_bins):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / len(predictions)) * abs(bin_means[i] - bin_centers[i])
        
        # Calculate Maximum Calibration Error (MCE)
        mce = 0
        for i in range(n_bins):
            if bin_counts[i] > 0:
                mce = max(mce, abs(bin_means[i] - bin_centers[i]))
        
        # Calculate Brier Score
        brier_score = np.mean((predictions - true_values) ** 2)
        
        # Store reliability diagram data
        reliability_diagram = {
            'bin_centers': bin_centers,
            'bin_means': bin_means,
            'bin_counts': bin_counts
        }
        
        self.calibrators[symbol]['reliability_diagram'] = reliability_diagram
        
        # Return metrics
        return {
            'ECE': ece,
            'MCE': mce,
            'Brier_Score': brier_score,
            'reliability_diagram': reliability_diagram
        }
    
    def plot_calibration_curve(self, symbol, save_path=None):
        """
        Plot calibration curve for a specific symbol.
        
        Args:
            symbol: Trading symbol
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib figure object
        """
        if symbol not in self.calibrators or 'reliability_diagram' not in self.calibrators[symbol]:
            logger.warning(f"No calibration data available for {symbol}")
            return None
        
        # Get reliability diagram data
        reliability_diagram = self.calibrators[symbol]['reliability_diagram']
        bin_centers = reliability_diagram['bin_centers']
        bin_means = reliability_diagram['bin_means']
        bin_counts = reliability_diagram['bin_counts']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot calibration curve
        ax.plot(bin_centers, bin_means, 'o-', markersize=8, label='Calibration curve')
        
        # Plot diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Ideal calibration')
        
        # Plot histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(self.calibrators[symbol]['raw_predictions'], bins=len(bin_centers), 
                 alpha=0.3, color='blue', label='Prediction histogram')
        
        # Set labels and title
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('True fraction of positive class')
        ax2.set_ylabel('Count')
        ax.set_title(f'Calibration Curve for {symbol}')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Add metrics
        metrics = self.evaluate_calibration(symbol, 
                                           self.calibrators[symbol]['raw_predictions'],
                                           self.calibrators[symbol]['true_values'])
        
        metrics_text = f"ECE: {metrics['ECE']:.4f}\nMCE: {metrics['MCE']:.4f}\nBrier Score: {metrics['Brier_Score']:.4f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        plt.tight_layout()
        return fig
    
    def save(self, filename=None):
        """
        Save the calibration models.
        
        Args:
            filename: Custom filename, defaults to probability_calibrator.pkl
        """
        if filename is None:
            filename = os.path.join(self.models_dir, 'probability_calibrator.pkl')
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save calibrators
        joblib.dump({
            'calibrators': self.calibrators,
            'method': self.method
        }, filename)
        
        logger.info(f"Saved probability calibration models to {filename}")
    
    def load(self, filename=None):
        """
        Load calibration models.
        
        Args:
            filename: Custom filename, defaults to probability_calibrator.pkl
        """
        if filename is None:
            filename = os.path.join(self.models_dir, 'probability_calibrator.pkl')
            
        if not os.path.exists(filename):
            logger.warning(f"No saved calibration models found at {filename}")
            return False
        
        # Load calibrators
        data = joblib.load(filename)
        self.calibrators = data['calibrators']
        self.method = data['method']
        
        logger.info(f"Loaded probability calibration models from {filename}")
        return True


class RegressionCalibrator:
    """
    Calibration for regression model outputs to produce reliable prediction intervals.
    
    This class implements methods to calibrate regression predictions to provide
    trustworthy confidence intervals that accurately reflect prediction uncertainty:
    
    1. Quantile Regression: Direct prediction of confidence intervals
    2. Conformal Prediction: Distribution-free method with coverage guarantees
    3. Gaussian Process-based: Bayesian approach for prediction intervals
    """
    
    def __init__(self, 
                 method='conformal',  # 'quantile', 'conformal', 'gaussian'
                 confidence_level=0.95,
                 models_dir='saved_models/calibration'):
        """
        Initialize the regression calibrator.
        
        Args:
            method: Calibration method to use
            confidence_level: Desired confidence level for prediction intervals
            models_dir: Directory for saving calibration models
        """
        self.method = method
        self.confidence_level = confidence_level
        self.models_dir = models_dir
        self.calibrators = {}
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"Initialized RegressionCalibrator with method: {method}, "
                   f"confidence level: {confidence_level}")
    
    def fit(self, symbol, predictions, true_values):
        """
        Fit regression calibration model for the specified symbol.
        
        Args:
            symbol: Trading symbol
            predictions: Model predictions (point estimates)
            true_values: Actual values
            
        Returns:
            Calibration metrics
        """
        # Initialize calibration for this symbol
        if symbol not in self.calibrators:
            self.calibrators[symbol] = {}
        
        # Store the data for diagnostics
        self.calibrators[symbol]['predictions'] = predictions
        self.calibrators[symbol]['true_values'] = true_values
        
        # Calculate residuals
        residuals = true_values - predictions
        
        # Fit different calibration methods
        if self.method == 'conformal':
            self._fit_conformal_prediction(symbol, residuals)
        elif self.method == 'quantile':
            self._fit_quantile_regression(symbol, predictions, true_values)
        elif self.method == 'gaussian':
            self._fit_gaussian_process(symbol, residuals)
        
        # Calculate calibration metrics
        metrics = self.evaluate_calibration(symbol, predictions, true_values)
        logger.info(f"Regression calibration fitted for {symbol} with metrics: {metrics}")
        
        return metrics
    
    def _fit_conformal_prediction(self, symbol, residuals):
        """
        Fit conformal prediction intervals.
        
        This is a distribution-free method that guarantees coverage.
        """
        # Calculate absolute residuals
        abs_residuals = np.abs(residuals)
        
        # Calculate the quantile based on confidence level
        alpha = 1 - self.confidence_level
        quantile = np.quantile(abs_residuals, 1 - alpha)
        
        # Store calibrator
        self.calibrators[symbol]['conformal'] = {
            'quantile': quantile
        }
        
        logger.info(f"Fitted Conformal Prediction for {symbol} with quantile: {quantile:.4f}")
    
    def _fit_quantile_regression(self, symbol, predictions, true_values):
        """
        Fit quantile regression for prediction intervals.
        
        This approach models the lower and upper quantiles directly.
        """
        # Calculate error scaling factor based on predictions
        errors = np.abs(true_values - predictions)
        
        # Fit a simple linear model to model error as a function of prediction magnitude
        X = np.abs(predictions).reshape(-1, 1)
        
        # To avoid division by zero
        X = np.maximum(X, 1e-10)
        
        # Fit lower quantile
        lower_alpha = (1 - self.confidence_level) / 2
        lower_quantile = np.quantile(errors / X.ravel(), lower_alpha)
        
        # Fit upper quantile
        upper_alpha = 1 - lower_alpha
        upper_quantile = np.quantile(errors / X.ravel(), upper_alpha)
        
        # Store calibrator
        self.calibrators[symbol]['quantile'] = {
            'lower_quantile': lower_quantile,
            'upper_quantile': upper_quantile
        }
        
        logger.info(f"Fitted Quantile Regression for {symbol} with quantiles: "
                   f"{lower_quantile:.4f}, {upper_quantile:.4f}")
    
    def _fit_gaussian_process(self, symbol, residuals):
        """
        Fit Gaussian process for prediction intervals.
        
        This approach models the residuals as a Gaussian distribution.
        """
        # Calculate mean and std of residuals
        mean = np.mean(residuals)
        std = np.std(residuals)
        
        # Calculate z-score for the confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        # Store calibrator
        self.calibrators[symbol]['gaussian'] = {
            'mean': mean,
            'std': std,
            'z_score': z_score
        }
        
        logger.info(f"Fitted Gaussian Process for {symbol} with mean: {mean:.4f}, std: {std:.4f}")
    
    def calibrate(self, symbol, predictions):
        """
        Calibrate regression predictions to produce prediction intervals.
        
        Args:
            symbol: Trading symbol
            predictions: Point predictions to calibrate
            
        Returns:
            Dictionary with point predictions and prediction intervals
        """
        if symbol not in self.calibrators:
            logger.warning(f"No calibration model for {symbol}, returning raw predictions")
            return {
                'predictions': predictions,
                'lower_bound': predictions * 0.9,  # Default fallback
                'upper_bound': predictions * 1.1,  # Default fallback
                'confidence_level': self.confidence_level
            }
        
        # Apply the appropriate calibration method
        if self.method == 'conformal':
            return self._apply_conformal_prediction(symbol, predictions)
        elif self.method == 'quantile':
            return self._apply_quantile_regression(symbol, predictions)
        elif self.method == 'gaussian':
            return self._apply_gaussian_process(symbol, predictions)
        else:
            logger.warning(f"Unknown calibration method: {self.method}, returning raw predictions")
            return {
                'predictions': predictions,
                'lower_bound': predictions * 0.9,  # Default fallback
                'upper_bound': predictions * 1.1,  # Default fallback
                'confidence_level': self.confidence_level
            }
    
    def _apply_conformal_prediction(self, symbol, predictions):
        """Apply conformal prediction to create prediction intervals."""
        if 'conformal' not in self.calibrators[symbol]:
            logger.warning(f"No Conformal Prediction model for {symbol}")
            return {
                'predictions': predictions,
                'lower_bound': predictions * 0.9,
                'upper_bound': predictions * 1.1,
                'confidence_level': self.confidence_level
            }
        
        # Get the quantile
        quantile = self.calibrators[symbol]['conformal']['quantile']
        
        # Create prediction intervals
        lower_bound = predictions - quantile
        upper_bound = predictions + quantile
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': self.confidence_level
        }
    
    def _apply_quantile_regression(self, symbol, predictions):
        """Apply quantile regression to create prediction intervals."""
        if 'quantile' not in self.calibrators[symbol]:
            logger.warning(f"No Quantile Regression model for {symbol}")
            return {
                'predictions': predictions,
                'lower_bound': predictions * 0.9,
                'upper_bound': predictions * 1.1,
                'confidence_level': self.confidence_level
            }
        
        # Get the quantiles
        lower_quantile = self.calibrators[symbol]['quantile']['lower_quantile']
        upper_quantile = self.calibrators[symbol]['quantile']['upper_quantile']
        
        # Create prediction intervals (scale by prediction magnitude)
        X = np.abs(predictions)
        
        # To avoid numerical issues
        X = np.maximum(X, 1e-10)
        
        lower_bound = predictions - lower_quantile * X
        upper_bound = predictions + upper_quantile * X
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': self.confidence_level
        }
    
    def _apply_gaussian_process(self, symbol, predictions):
        """Apply Gaussian process to create prediction intervals."""
        if 'gaussian' not in self.calibrators[symbol]:
            logger.warning(f"No Gaussian Process model for {symbol}")
            return {
                'predictions': predictions,
                'lower_bound': predictions * 0.9,
                'upper_bound': predictions * 1.1,
                'confidence_level': self.confidence_level
            }
        
        # Get the parameters
        mean = self.calibrators[symbol]['gaussian']['mean']
        std = self.calibrators[symbol]['gaussian']['std']
        z_score = self.calibrators[symbol]['gaussian']['z_score']
        
        # Create prediction intervals
        lower_bound = predictions + mean - z_score * std
        upper_bound = predictions + mean + z_score * std
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': self.confidence_level
        }
    
    def evaluate_calibration(self, symbol, predictions, true_values):
        """
        Evaluate the calibration quality for regression.
        
        Args:
            symbol: Trading symbol
            predictions: Model predictions
            true_values: Actual values
            
        Returns:
            Dictionary with calibration metrics
        """
        # Apply calibration to get prediction intervals
        calibrated = self.calibrate(symbol, predictions)
        lower_bound = calibrated['lower_bound']
        upper_bound = calibrated['upper_bound']
        
        # Calculate coverage (fraction of true values within prediction intervals)
        coverage = np.mean((true_values >= lower_bound) & (true_values <= upper_bound))
        
        # Calculate interval width
        interval_width = np.mean(upper_bound - lower_bound)
        
        # Calculate normalized interval width
        norm_interval_width = np.mean((upper_bound - lower_bound) / np.abs(predictions))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(predictions - true_values))
        
        return {
            'coverage': coverage,
            'target_coverage': self.confidence_level,
            'coverage_error': abs(coverage - self.confidence_level),
            'interval_width': interval_width,
            'normalized_interval_width': norm_interval_width,
            'rmse': rmse,
            'mae': mae
        }
    
    def plot_calibration(self, symbol, n_samples=100, save_path=None):
        """
        Plot calibration results for regression.
        
        Args:
            symbol: Trading symbol
            n_samples: Number of samples to plot
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib figure object
        """
        if symbol not in self.calibrators:
            logger.warning(f"No calibration data available for {symbol}")
            return None
        
        # Get the data
        predictions = self.calibrators[symbol]['predictions']
        true_values = self.calibrators[symbol]['true_values']
        
        # Apply calibration
        calibrated = self.calibrate(symbol, predictions)
        lower_bound = calibrated['lower_bound']
        upper_bound = calibrated['upper_bound']
        
        # Select subset of samples for plotting
        if len(predictions) > n_samples:
            indices = np.random.choice(len(predictions), n_samples, replace=False)
            indices = np.sort(indices)  # Sort for better visualization
        else:
            indices = np.arange(len(predictions))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot prediction intervals
        for i, idx in enumerate(indices):
            ax.plot([i, i], [lower_bound[idx], upper_bound[idx]], 'b-', alpha=0.3)
        
        # Plot predictions and true values
        ax.plot(range(len(indices)), predictions[indices], 'bo', label='Predictions')
        ax.plot(range(len(indices)), true_values[indices], 'ro', label='True values')
        
        # Set labels and title
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Value')
        ax.set_title(f'Prediction Intervals for {symbol} ({self.confidence_level*100:.0f}% confidence)')
        
        # Add legend
        ax.legend()
        
        # Add metrics
        metrics = self.evaluate_calibration(symbol, predictions, true_values)
        
        metrics_text = (f"Coverage: {metrics['coverage']:.4f} (target: {metrics['target_coverage']:.4f})\n"
                       f"Interval Width: {metrics['interval_width']:.4f}\n"
                       f"RMSE: {metrics['rmse']:.4f}\n"
                       f"MAE: {metrics['mae']:.4f}")
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        plt.tight_layout()
        return fig
    
    def save(self, filename=None):
        """
        Save the regression calibration models.
        
        Args:
            filename: Custom filename, defaults to regression_calibrator.pkl
        """
        if filename is None:
            filename = os.path.join(self.models_dir, 'regression_calibrator.pkl')
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save calibrators
        joblib.dump({
            'calibrators': self.calibrators,
            'method': self.method,
            'confidence_level': self.confidence_level
        }, filename)
        
        logger.info(f"Saved regression calibration models to {filename}")
    
    def load(self, filename=None):
        """
        Load regression calibration models.
        
        Args:
            filename: Custom filename, defaults to regression_calibrator.pkl
        """
        if filename is None:
            filename = os.path.join(self.models_dir, 'regression_calibrator.pkl')
            
        if not os.path.exists(filename):
            logger.warning(f"No saved regression calibration models found at {filename}")
            return False
        
        # Load calibrators
        data = joblib.load(filename)
        self.calibrators = data['calibrators']
        self.method = data['method']
        self.confidence_level = data['confidence_level']
        
        logger.info(f"Loaded regression calibration models from {filename}")
        return True 
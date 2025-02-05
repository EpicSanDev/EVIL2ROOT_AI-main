from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from typing import Tuple, Dict, Optional

class RiskManagementModel:
    def __init__(self, max_risk=0.02, volatility_window=20, confidence_level=0.95, kelly_fraction=0.5):
        self.max_risk = max_risk
        self.volatility_window = volatility_window
        self.confidence_level = confidence_level
        self.kelly_fraction = kelly_fraction  # Kelly criterion fraction
        self.scalers = {}
        self.models = {}
        self.volatility_models = {}
        self.correlation_matrix = None
        logging.info(f"RiskManagement initialized with max risk {self.max_risk}, kelly fraction {self.kelly_fraction}")

    def calculate_risk(self, portfolio_value: float, position_size: float, volatility: float, 
                      stop_loss: float, win_rate: Optional[float] = None, 
                      avg_win: Optional[float] = None, avg_loss: Optional[float] = None) -> Tuple[bool, float, Dict]:
        """
        Calculate position risk using Value at Risk (VaR), Kelly Criterion, and position sizing
        
        Args:
            portfolio_value: Current portfolio value
            position_size: Proposed position size
            volatility: Current market volatility
            stop_loss: Stop loss level
            win_rate: Historical win rate (optional)
            avg_win: Average winning trade (optional)
            avg_loss: Average losing trade (optional)
        
        Returns:
            Tuple containing:
            - bool: Whether risk is acceptable
            - float: Total risk
            - dict: Additional risk metrics
        """
        # Calculate VaR
        var = self._calculate_var(volatility)
        position_risk = (position_size * var) / portfolio_value
        stop_loss_risk = (position_size * (1 - stop_loss)) / portfolio_value
        
        # Calculate Kelly position size if we have the required data
        kelly_size = None
        if all(x is not None for x in [win_rate, avg_win, avg_loss]):
            kelly_size = self._calculate_kelly_position(win_rate, avg_win, avg_loss, portfolio_value)
        
        # Calculate portfolio-level VaR if correlation matrix exists
        portfolio_var = None
        if self.correlation_matrix is not None:
            portfolio_var = self._calculate_portfolio_var(position_size, volatility)
        
        # Combine risk metrics
        total_risk = max(position_risk, stop_loss_risk)
        if portfolio_var:
            total_risk = max(total_risk, portfolio_var)
        
        # Prepare risk metrics dictionary
        risk_metrics = {
            'var_risk': position_risk,
            'stop_loss_risk': stop_loss_risk,
            'portfolio_var': portfolio_var,
            'kelly_size': kelly_size,
            'total_risk': total_risk
        }
        
        # Check if risk is acceptable
        is_acceptable = total_risk <= self.max_risk
        if not is_acceptable:
            logging.warning(f"Risk {total_risk:.2%} exceeds max allowed {self.max_risk:.2%}")
            if kelly_size and position_size > kelly_size:
                logging.warning(f"Position size {position_size} exceeds Kelly size {kelly_size}")
        
        return is_acceptable, total_risk, risk_metrics

    def _calculate_var(self, volatility: float) -> float:
        """
        Calculate Value at Risk using volatility and confidence level
        """
        z_score = norm.ppf(self.confidence_level)
        var = volatility * z_score
        return var

    def _calculate_kelly_position(self, win_rate: float, avg_win: float, 
                                avg_loss: float, portfolio_value: float) -> float:
        """
        Calculate optimal position size using the Kelly Criterion
        """
        # Kelly formula: f = (p*b - q) / b
        # where: p = win rate, q = loss rate, b = win/loss ratio
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly_pct = (win_rate * b - q) / b
        # Apply fractional Kelly to be more conservative
        kelly_pct *= self.kelly_fraction
        
        # Ensure Kelly percentage is within reasonable bounds
        kelly_pct = max(0, min(kelly_pct, self.max_risk))
        
        return kelly_pct * portfolio_value

    def _calculate_portfolio_var(self, position_size: float, volatility: float) -> float:
        """
        Calculate portfolio-level Value at Risk considering correlations
        """
        if self.correlation_matrix is None:
            return 0.0
            
        # Calculate weighted volatility
        portfolio_vol = np.sqrt(
            np.dot(
                np.dot([position_size], self.correlation_matrix),
                [position_size]
            )
        ) * volatility
        
        return self._calculate_var(portfolio_vol)

    def update_correlation_matrix(self, returns_data: pd.DataFrame) -> None:
        """
        Update the correlation matrix for portfolio risk calculations
        """
        self.correlation_matrix = returns_data.corr().values
        logging.info("Updated correlation matrix for portfolio risk calculations")

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced risk features from price data
        """
        df = pd.DataFrame(index=data.index)
        
        # Price-based features
        df['returns'] = data['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.volatility_window).std()
        df['high_low_range'] = (data['High'] - data['Low']) / data['Close']
        
        # Volume-based features
        df['volume_ma'] = data['Volume'].rolling(10).mean()
        df['volume_std'] = data['Volume'].rolling(10).std()
        
        # Technical indicators
        df['atr'] = self._calculate_atr(data)
        df['beta'] = self._calculate_rolling_beta(data)
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        return df

    def _calculate_atr(self, data, period=14):
        """
        Calculate Average True Range
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def _calculate_rolling_beta(self, data, window=20):
        """
        Calculate rolling beta against market returns
        """
        returns = data['Close'].pct_change()
        rolling_std = returns.rolling(window=window).std()
        return rolling_std

    def train(self, data: pd.DataFrame, symbol: str, market_data: Optional[pd.DataFrame] = None) -> None:
        """
        Train risk prediction models using advanced features
        """
        features_df = self._calculate_features(data)
        
        # Prepare training data
        X = features_df.values
        y = data['Close'].pct_change().abs().values  # Use absolute returns as risk target
        
        # Remove NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        self.scalers[symbol] = StandardScaler()
        X_scaled = self.scalers[symbol].fit_transform(X)
        
        # Train main risk model
        risk_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        risk_model.fit(X_scaled, y)
        self.models[symbol] = risk_model
        
        # Train volatility model
        vol_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )
        vol_model.fit(X_scaled, features_df['volatility'].values[mask])
        self.volatility_models[symbol] = vol_model
        
        logging.info(f"Trained risk and volatility models for {symbol}")

    def predict(self, data: pd.DataFrame, symbol: str) -> Tuple[float, Dict[str, float]]:
        """
        Predict risk level and volatility
        """
        features_df = self._calculate_features(data)
        features = features_df.iloc[-1].values.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scalers[symbol].transform(features)
        
        # Predict risk and volatility
        risk_level = self.models[symbol].predict(features_scaled)[0]
        volatility = self.volatility_models[symbol].predict(features_scaled)[0]
        
        # Combine predictions into risk score
        combined_risk = 0.7 * risk_level + 0.3 * volatility
        
        return combined_risk

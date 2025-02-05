import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import logging
from typing import Tuple, Dict, Optional, List

class TpSlManagementModel:
    def __init__(self, atr_period=14, base_risk_reward_ratio=2.0, volatility_adjustment=True):
        self.models = {}
        self.scalers = {}
        self.atr_period = atr_period
        self.base_risk_reward_ratio = base_risk_reward_ratio
        self.volatility_adjustment = volatility_adjustment
        self.feature_columns = None
        self.market_regime = 'normal'  # Can be 'low_vol', 'normal', 'high_vol'
        self.trailing_stops = {}  # Store trailing stop levels for each position
        logging.info(f"TpSlManagement initialized with ATR period {atr_period} and base R:R {base_risk_reward_ratio}")

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close'].shift()
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        return atr

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for TP/SL prediction"""
        df = pd.DataFrame(index=data.index)
        
        # Price action features
        df['returns'] = data['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['atr'] = self._calculate_atr(data)
        df['high_low_range'] = (data['High'] - data['Low']) / data['Close']
        
        # Moving averages and trends
        df['sma_20'] = data['Close'].rolling(window=20).mean()
        df['sma_50'] = data['Close'].rolling(window=50).mean()
        df['price_to_sma20'] = data['Close'] / df['sma_20']
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(data['Close'])
        df['momentum'] = data['Close'] / data['Close'].shift(10) - 1
        
        # Volatility indicators
        df['volume_ma'] = data['Volume'].rolling(window=20).mean()
        df['volume_std'] = data['Volume'].rolling(window=20).std()
        df['volatility_regime'] = df['volatility'].rolling(window=50).mean()
        
        df.fillna(method='bfill', inplace=True)
        self.feature_columns = df.columns
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _adjust_risk_reward_ratio(self, volatility: float, trend_strength: float) -> float:
        """
        Dynamically adjust risk-reward ratio based on market conditions
        
        Args:
            volatility: Current market volatility
            trend_strength: Strength of the current trend (0 to 1)
        """
        # Base adjustment on volatility regime
        if volatility > 0.02:  # High volatility
            self.market_regime = 'high_vol'
            vol_factor = 0.8  # More conservative in high vol
        elif volatility < 0.005:  # Low volatility
            self.market_regime = 'low_vol'
            vol_factor = 1.2  # More aggressive in low vol
        else:
            self.market_regime = 'normal'
            vol_factor = 1.0

        # Adjust based on trend strength
        trend_factor = 1.0 + (trend_strength * 0.5)  # Up to 50% increase for strong trends
        
        return self.base_risk_reward_ratio * vol_factor * trend_factor

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build an advanced LSTM model for TP/SL prediction"""
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(2, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber_loss',
            metrics=['mae']
        )
        return model

    def _prepare_sequences(self, features: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(lookback, len(features)):
            # Input sequence
            X.append(features[i-lookback:i])
            
            # Target values (TP and SL)
            current_price = features[i-1][-1]  # Last feature is assumed to be price
            atr = features[i-1][-2]  # Second to last feature is assumed to be ATR
            volatility = features[i-1][-3]  # Third to last feature is volatility
            trend_strength = features[i-1][-4]  # Fourth to last is trend strength
            
            # Dynamic TP/SL based on market conditions
            risk_reward_ratio = self._adjust_risk_reward_ratio(volatility, trend_strength)
            tp_distance = atr * risk_reward_ratio
            sl_distance = atr
            
            # Adjust based on trend
            if features[i-1][-5] > 0:  # If momentum is positive
                tp = current_price + tp_distance
                sl = current_price - sl_distance
            else:
                tp = current_price - tp_distance
                sl = current_price + sl_distance
            
            y.append([tp, sl])
        
        return np.array(X), np.array(y)

    def train(self, data: pd.DataFrame, symbol: str) -> None:
        """Train the TP/SL prediction model"""
        features_df = self._calculate_features(data)
        features = features_df.values
        
        # Scale features
        self.scalers[symbol] = StandardScaler()
        scaled_features = self.scalers[symbol].fit_transform(features)
        
        # Prepare sequences
        X, y = self._prepare_sequences(scaled_features)
        
        # Build and train model
        model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model with reduced learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )
        
        # Train model
        model.fit(
            X, y,
            validation_split=0.2,
            batch_size=32,
            epochs=100,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models[symbol] = model
        logging.info(f"Trained TP/SL model for {symbol}")

    def predict(self, data: pd.DataFrame, symbol: str, position_id: Optional[str] = None) -> Dict[str, float]:
        """
        Predict TP/SL levels and manage trailing stops
        
        Args:
            data: Market data
            symbol: Trading symbol
            position_id: Unique identifier for the position (for trailing stop management)
        
        Returns:
            Dictionary containing take profit, stop loss, and trailing stop levels
        """
        if symbol not in self.models or symbol not in self.scalers:
            raise ValueError(f"Model for {symbol} not trained")
            
        # Calculate features
        features_df = self._calculate_features(data)
        features = features_df.values
        
        # Scale features
        scaled_features = self.scalers[symbol].transform(features)
        
        # Prepare sequence
        lookback = 60
        if len(scaled_features) < lookback:
            raise ValueError(f"Not enough data points. Need at least {lookback}")
            
        X = scaled_features[-lookback:].reshape(1, lookback, -1)
        
        # Get model predictions
        predictions = self.models[symbol].predict(X)
        current_price = data['Close'].iloc[-1]
        
        # Get market conditions
        atr = self._calculate_atr(data).iloc[-1]
        volatility = features_df['volatility'].iloc[-1]
        trend_strength = features_df['trend_strength'].iloc[-1]
        momentum = features_df['momentum'].iloc[-1]
        
        # Adjust risk-reward ratio based on market conditions
        adjusted_rr = self._adjust_risk_reward_ratio(volatility, trend_strength)
        
        # Get base TP/SL levels from model
        tp, sl = predictions[0]
        
        # Fine-tune based on market conditions
        if momentum > 0:
            tp = max(tp, current_price + atr * adjusted_rr)
            sl = min(sl, current_price - atr)
        else:
            tp = min(tp, current_price - atr * adjusted_rr)
            sl = max(sl, current_price + atr)
        
        # Update trailing stop if position exists
        trailing_stop = None
        if position_id and position_id in self.trailing_stops:
            old_trailing_stop = self.trailing_stops[position_id]
            if momentum > 0:
                trailing_stop = max(old_trailing_stop, current_price - atr)
            else:
                trailing_stop = min(old_trailing_stop, current_price + atr)
            self.trailing_stops[position_id] = trailing_stop
        elif position_id:
            trailing_stop = sl
            self.trailing_stops[position_id] = trailing_stop
        
        return {
            'take_profit': tp,
            'stop_loss': sl,
            'trailing_stop': trailing_stop,
            'risk_reward_ratio': adjusted_rr,
            'market_regime': self.market_regime
        }

    def remove_trailing_stop(self, position_id: str) -> None:
        """Remove trailing stop for closed position"""
        if position_id in self.trailing_stops:
            del self.trailing_stops[position_id]

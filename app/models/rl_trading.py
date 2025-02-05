import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import logging
import ta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import deque

class AdvancedTradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], initial_balance=100000, transaction_fee=0.001,
                timeframes: List[str] = ['1h', '4h', '1d']):
        super(AdvancedTradingEnv, self).__init__()
        
        self.timeframes = timeframes
        self.data = {tf: self._prepare_data(df) for tf, df in data.items()}
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = 30  # Number of past observations to consider
        
        # Enhanced state tracking
        self.position_history = deque(maxlen=100)  # Track recent positions
        self.reward_history = deque(maxlen=100)    # Track recent rewards
        self.drawdown_history = deque(maxlen=100)  # Track drawdowns
        
        # Action space: continuous values between -1 (full sell) and 1 (full buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: Multi-timeframe OHLCV + indicators + account info
        features_per_timeframe = 20  # Increased from 15 to include new indicators
        total_features = sum(features_per_timeframe for _ in timeframes)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size * total_features + 5,),  # +5 for enhanced account info
            dtype=np.float32
        )
        
        self.reset()
        logging.info(f"Trading environment initialized with timeframes: {timeframes}")

    def _prepare_data(self, data: pd.DataFrame, include_sentiment: bool = True) -> pd.DataFrame:
        """Prepare data with technical indicators"""
        df = data.copy()
        
        # Add technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['bb_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['mfi'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['williamsr'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Normalize data
        for column in df.columns:
            if column != 'Date':
                df[column] = (df[column] - df[column].mean()) / df[column].std()
        
        df.fillna(0, inplace=True)
        return df

    def _get_observation(self) -> np.ndarray:
        """Get the current observation state across all timeframes"""
        obs_list = []
        
        # Get data for each timeframe
        for tf in self.timeframes:
            tf_data = self.data[tf]
            obs_window = tf_data.iloc[self.current_step-self.window_size:self.current_step]
            obs_list.append(obs_window.values.flatten())
        
        # Combine all timeframe data
        obs = np.concatenate(obs_list)
        
        # Enhanced account information
        account_info = [
            self.balance / self.initial_balance - 1,  # Normalized balance
            self.current_position,                    # Current position
            np.mean(self.reward_history) if self.reward_history else 0,  # Average recent reward
            min(self.drawdown_history) if self.drawdown_history else 0,  # Max drawdown
            len([p for p in self.position_history if p != 0]) / max(len(self.position_history), 1)  # Activity rate
        ]
        
        obs = np.append(obs, account_info)
        return obs.astype(np.float32)

    def _calculate_reward(self, action: float, risk_metrics: Optional[Dict] = None) -> float:
        """Calculate reward based on action and market movement"""
        """Calculate reward with enhanced risk-adjustment and multi-timeframe analysis"""
        # Get current data across timeframes
        current_data = {tf: self.data[tf].iloc[self.current_step] for tf in self.timeframes}
        
        # Portfolio value change
        old_portfolio_value = self.balance + self.current_position * self.data['1h'].iloc[self.current_step-1]['Close']
        new_portfolio_value = self.balance + self.current_position * self.data['1h'].iloc[self.current_step]['Close']
        value_change = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Transaction cost penalty
        transaction_cost = abs(action) * self.transaction_fee
        
        # Multi-timeframe trend alignment bonus
        trend_alignment = self._calculate_trend_alignment()
        trend_bonus = 0.1 * trend_alignment  # Reward for trading with the trend
        
        # Risk-adjusted components
        volatility = self.data['1h'].iloc[self.current_step-20:self.current_step]['Close'].std()
        sharpe_ratio = value_change / (volatility + 1e-9)
        
        # Drawdown penalty
        current_drawdown = (new_portfolio_value - self.initial_balance) / self.initial_balance
        self.drawdown_history.append(current_drawdown)
        drawdown_penalty = min(0, current_drawdown) * 0.1
        
        # Position sizing penalty if risk metrics available
        position_penalty = 0
        if risk_metrics and 'kelly_size' in risk_metrics:
            kelly_size = risk_metrics['kelly_size']
            if kelly_size and abs(self.current_position) > kelly_size:
                position_penalty = 0.1 * (abs(self.current_position) - kelly_size) / kelly_size
        
        # Combine components
        reward = (
            sharpe_ratio +           # Risk-adjusted returns
            trend_bonus +           # Trend alignment
            drawdown_penalty -      # Drawdown penalty
            transaction_cost -      # Transaction costs
            position_penalty        # Position sizing penalty
        )
        
        self.reward_history.append(reward)
        return float(reward)

    def _calculate_trend_alignment(self) -> float:
        """Calculate how well aligned trends are across timeframes"""
        trends = []
        for tf in self.timeframes:
            df = self.data[tf]
            sma_fast = df['Close'].rolling(20).mean()
            sma_slow = df['Close'].rolling(50).mean()
            trend = 1 if sma_fast.iloc[-1] > sma_slow.iloc[-1] else -1
            trends.append(trend)
        
        # Calculate agreement ratio
        agreement = sum(1 for t in trends if t == trends[0]) / len(trends)
        return agreement * 2 - 1  # Convert to [-1, 1] range

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        self.current_step += 1
        
        if self.current_step >= len(self.data):
            return self._get_observation(), 0, True, False, {}
        
        """Execute one step in the environment with enhanced risk management"""
        self.current_step += 1
        
        if self.current_step >= len(self.data['1h']):
            return self._get_observation(), 0, True, False, {}
        
        # Convert action from [-1, 1] to position size
        target_position = float(action[0])
        
        # Calculate position change
        position_change = target_position - self.current_position
        
        # Get current market data
        current_price = self.data['1h'].iloc[self.current_step]['Close']
        
        # Calculate transaction costs
        transaction_cost = abs(position_change) * current_price * self.transaction_fee
        
        # Get risk metrics from the latest data
        risk_metrics = self._get_risk_metrics()
        
        # Execute trade if it passes risk checks
        if self._validate_trade(position_change, risk_metrics):
            self.balance -= position_change * current_price + transaction_cost
            self.current_position = target_position
            self.position_history.append(target_position)
        else:
            # If trade violates risk limits, force position reduction
            safe_position = self._calculate_safe_position(risk_metrics)
            position_change = safe_position - self.current_position
            self.balance -= position_change * current_price + transaction_cost
            self.current_position = safe_position
            self.position_history.append(safe_position)
        
        # Calculate reward with risk metrics
        reward = self._calculate_reward(action[0], risk_metrics)
        
        # Update metrics
        self.returns.append(reward)
        portfolio_value = self.balance + self.current_position * current_price
        self.portfolio_values.append(portfolio_value)
        
        done = self.current_step >= len(self.data['1h']) - 1
        
        return self._get_observation(), reward, done, False, {
            'portfolio_value': portfolio_value,
            'position': self.current_position,
            'balance': self.balance,
            'risk_metrics': risk_metrics
        }

    def _get_risk_metrics(self) -> Dict:
        """Calculate current risk metrics"""
        current_data = self.data['1h'].iloc[self.current_step-20:self.current_step]
        volatility = current_data['Close'].pct_change().std()
        avg_volume = current_data['Volume'].mean()
        
        return {
            'volatility': volatility,
            'avg_volume': avg_volume,
            'kelly_size': self._calculate_kelly_size(),
            'var': self._calculate_var(volatility)
        }

    def _validate_trade(self, position_change: float, risk_metrics: Dict) -> bool:
        """Validate if trade meets risk management criteria"""
        if 'kelly_size' in risk_metrics and risk_metrics['kelly_size']:
            if abs(self.current_position + position_change) > risk_metrics['kelly_size']:
                return False
        
        if 'var' in risk_metrics:
            position_var = abs(position_change) * risk_metrics['var']
            if position_var > self.balance * 0.02:  # 2% VaR limit
                return False
        
        return True

    def _calculate_safe_position(self, risk_metrics: Dict) -> float:
        """Calculate maximum safe position size based on risk metrics"""
        if 'kelly_size' in risk_metrics and risk_metrics['kelly_size']:
            return np.clip(self.current_position, -risk_metrics['kelly_size'], risk_metrics['kelly_size'])
        return 0.0

    def _calculate_kelly_size(self) -> float:
        """Calculate Kelly Criterion position size"""
        recent_trades = list(self.position_history)
        if len(recent_trades) < 10:
            return None
        
        wins = sum(1 for i in range(1, len(recent_trades)) if 
                  (recent_trades[i] > 0 and self.returns[i] > 0) or
                  (recent_trades[i] < 0 and self.returns[i] < 0))
        
        win_rate = wins / len(recent_trades)
        avg_win = np.mean([r for r in self.returns if r > 0]) if any(r > 0 for r in self.returns) else 0
        avg_loss = abs(np.mean([r for r in self.returns if r < 0])) if any(r < 0 for r in self.returns) else 0
        
        if avg_loss == 0:
            return self.balance * 0.02  # Default to 2% if no loss data
            
        kelly_fraction = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return self.balance * max(0, min(kelly_fraction, 0.02))  # Cap at 2%

    def _calculate_var(self, volatility: float, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        z_score = norm.ppf(confidence_level)
        return volatility * z_score

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.current_position = 0
        self.returns = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation(), {}

    def render(self):
        """Render the environment"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()

def create_trading_env(data: pd.DataFrame) -> gym.Env:
    """Create and wrap the trading environment"""
    def make_env():
        env = AdvancedTradingEnv(data)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    return env

def train_rl_agent(data: pd.DataFrame, total_timesteps=100000):
    """Train a reinforcement learning agent"""
    # Create environment
    env = create_trading_env(data)
    
    # Initialize agent
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=0.0003,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[400, 300], qf=[400, 300])
        )
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=100
    )
    
    return model

def evaluate_rl_model(model, env, num_episodes=10):
    """Evaluate the trained model"""
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }

def run_reinforcement_learning(data_path: str):
    """Main function to run reinforcement learning training"""
    logging.info("Starting reinforcement learning training...")
    
    # Load and prepare data
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Train model
    model = train_rl_agent(data)
    
    # Evaluate model
    env = create_trading_env(data)
    evaluation_results = evaluate_rl_model(model, env)
    
    logging.info("Training completed. Evaluation results: %s", evaluation_results)
    
    return model, evaluation_results

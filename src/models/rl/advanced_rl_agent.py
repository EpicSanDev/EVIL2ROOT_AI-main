import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt
from .rl_trading import AdvancedTradingEnv

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), \
               np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), batches

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha=0.0003, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh()  # Outputs between -1 and 1
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        return self.actor(state)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha=0.0003, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        return self.critic(state)

class CustomPPOAgent:
    """
    Custom PPO Agent implementation with PyTorch
    """
    def __init__(self, input_dims, n_actions=1, gamma=0.99, alpha=0.0003, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        self.actor = ActorNetwork(input_dims, n_actions, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
        
    def save_models(self, path):
        print(f'Saving models to {path}')
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        
    def load_models(self, path):
        print(f'Loading models from {path}')
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        
    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)
        
        action = self.actor(state)
        value = self.critic(state)
        
        return action.detach().cpu().numpy()[0], value.item()
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-dones_arr[k]) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            
            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                
                new_actions = self.actor(states)
                critic_value = self.critic(states)
                
                critic_value = torch.squeeze(critic_value)
                
                # Calculate actor loss
                actor_loss = -torch.min(
                    new_actions * advantage[batch],
                    new_actions * torch.clamp(advantage[batch], -self.policy_clip, self.policy_clip)
                )
                actor_loss = actor_loss.mean()
                
                # Calculate critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = F.mse_loss(critic_value, returns)
                
                total_loss = actor_loss + 0.5 * critic_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()

class RLAgentManager:
    """
    Manages RL agents for different market environments and strategies
    """
    def __init__(self, model_dir='saved_models/rl_agents'):
        self.model_dir = model_dir
        self.agents = {}
        self.envs = {}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
    def create_agent(self, agent_id: str, env: gym.Env, model_type: str = 'PPO', 
                     params: Dict = None):
        """
        Create and train a new agent or load an existing one
        
        Parameters:
        -----------
        agent_id : str
            Unique identifier for this agent
        env : gym.Env
            The environment to train the agent in
        model_type : str
            'PPO', 'SAC', 'TD3', or 'Custom'
        params : Dict
            Parameters for the model
        """
        agent_path = os.path.join(self.model_dir, agent_id)
        os.makedirs(agent_path, exist_ok=True)
        
        # Wrap the environment
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        # Set default parameters if none provided
        if params is None:
            params = {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'batch_size': 64,
                'buffer_size': 10000,
                'train_freq': 1,
                'gradient_steps': 1
            }
        
        # Create the agent based on the model type
        if model_type == 'PPO':
            model = PPO('MlpPolicy', vec_env, verbose=1, **params)
        elif model_type == 'SAC':
            model = SAC('MlpPolicy', vec_env, verbose=1, **params)
        elif model_type == 'TD3':
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = TD3('MlpPolicy', vec_env, action_noise=action_noise, verbose=1, **params)
        elif model_type == 'Custom':
            # For the custom PPO implementation
            n_actions = env.action_space.shape[-1]
            input_dims = env.observation_space.shape[0]
            model = CustomPPOAgent(input_dims=input_dims, n_actions=n_actions, **params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.agents[agent_id] = model
        self.envs[agent_id] = vec_env
        
        return model
    
    def train_agent(self, agent_id: str, total_timesteps: int = 100000, 
                    eval_freq: int = 10000, save_freq: int = 10000):
        """
        Train an existing agent
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        model = self.agents[agent_id]
        agent_path = os.path.join(self.model_dir, agent_id)
        
        # Create callbacks for evaluation and checkpoints
        eval_callback = EvalCallback(
            self.envs[agent_id],
            best_model_save_path=os.path.join(agent_path, 'best_model'),
            log_path=os.path.join(agent_path, 'logs'),
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(agent_path, 'checkpoints'),
            name_prefix=agent_id
        )
        
        # Train the model
        if hasattr(model, 'learn'):  # For stable-baselines3 models
            model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, checkpoint_callback]
            )
        else:  # For custom implementation
            # Custom training loop would go here
            pass
        
        # Save the final model
        if hasattr(model, 'save'):
            model.save(os.path.join(agent_path, 'final_model'))
        else:
            model.save_models(agent_path)
        
        # Also save the normalization stats
        self.envs[agent_id].save(os.path.join(agent_path, 'vec_normalize.pkl'))
        
        return model
    
    def load_agent(self, agent_id: str, model_type: str = 'PPO'):
        """
        Load a previously saved agent
        """
        agent_path = os.path.join(self.model_dir, agent_id)
        final_model_path = os.path.join(agent_path, 'final_model')
        
        if not os.path.exists(final_model_path + '.zip') and model_type != 'Custom':
            best_model_path = os.path.join(agent_path, 'best_model', 'best_model')
            if os.path.exists(best_model_path + '.zip'):
                final_model_path = best_model_path
            else:
                raise FileNotFoundError(f"No saved model found for agent {agent_id}")
        
        # Load the appropriate model type
        if model_type == 'PPO':
            model = PPO.load(final_model_path)
        elif model_type == 'SAC':
            model = SAC.load(final_model_path)
        elif model_type == 'TD3':
            model = TD3.load(final_model_path)
        elif model_type == 'Custom':
            # For our custom implementation
            # Need to initialize model with correct dimensions first
            model = CustomPPOAgent(input_dims=1, n_actions=1)  # Placeholder
            model.load_models(agent_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.agents[agent_id] = model
        
        # Load environment normalization if it exists
        vec_normalize_path = os.path.join(agent_path, 'vec_normalize.pkl')
        if os.path.exists(vec_normalize_path):
            self.envs[agent_id] = VecNormalize.load(vec_normalize_path)
        
        return model
    
    def predict(self, agent_id: str, observation, deterministic: bool = True):
        """
        Make a prediction with the agent
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        model = self.agents[agent_id]
        
        # Normalize observation if we have a VecNormalize environment
        if agent_id in self.envs and hasattr(self.envs[agent_id], 'normalize_obs'):
            observation = self.envs[agent_id].normalize_obs(observation)
        
        # Get prediction
        if hasattr(model, 'predict'):
            action, _ = model.predict(observation, deterministic=deterministic)
        else:
            # For custom implementation
            action, _ = model.choose_action(observation)
        
        return action
    
    def save_all_agents(self):
        """
        Save all agents
        """
        for agent_id, model in self.agents.items():
            agent_path = os.path.join(self.model_dir, agent_id)
            os.makedirs(agent_path, exist_ok=True)
            
            if hasattr(model, 'save'):
                model.save(os.path.join(agent_path, 'final_model'))
            else:
                model.save_models(agent_path)
            
            if agent_id in self.envs and hasattr(self.envs[agent_id], 'save'):
                self.envs[agent_id].save(os.path.join(agent_path, 'vec_normalize.pkl'))
    
    def evaluate_agent(self, agent_id: str, env: gym.Env, n_episodes: int = 10):
        """
        Evaluate an agent's performance
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        model = self.agents[agent_id]
        
        # Run evaluation episodes
        episode_rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action, _ = model.choose_action(obs)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'n_episodes': n_episodes
        }

# Classe spécialisée pour entrainer un agent sur différents régimes de marché
class MarketRegimeRLTrainer:
    """
    Trains specialized RL agents for different market regimes
    """
    def __init__(self, base_env_class, model_dir='saved_models/market_regime_agents'):
        self.base_env_class = base_env_class
        self.model_dir = model_dir
        self.agent_manager = RLAgentManager(model_dir)
        self.regime_agents = {}  # Maps market regime to agent ID
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
    def create_regime_specific_env(self, data, regime, **kwargs):
        """
        Create an environment specialized for a specific market regime
        """
        # Filter data for the specific regime
        regime_data = {}
        for timeframe, df in data.items():
            if 'market_regime' in df.columns:
                regime_data[timeframe] = df[df['market_regime'] == regime].copy()
            else:
                # If no regime column, use all data
                regime_data[timeframe] = df.copy()
                
            # Add some noise to prevent overfitting and add some variety
            for col in regime_data[timeframe].columns:
                if col not in ['Date', 'market_regime'] and regime_data[timeframe][col].dtype in [np.float64, np.int64]:
                    # Add small Gaussian noise
                    noise = np.random.normal(0, 0.01 * regime_data[timeframe][col].std(), size=len(regime_data[timeframe]))
                    regime_data[timeframe][col] = regime_data[timeframe][col] + noise
        
        # Create the environment
        env = self.base_env_class(data=regime_data, **kwargs)
        return env
    
    def train_regime_agents(self, data, regimes, timeframes=None, **kwargs):
        """
        Train specialized agents for each market regime
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Data for each timeframe
        regimes : List[str]
            List of market regimes to train agents for
        timeframes : List[str]
            List of timeframes to include
        """
        if timeframes is None:
            timeframes = list(data.keys())
        
        filtered_data = {tf: data[tf] for tf in timeframes if tf in data}
        
        for regime in regimes:
            agent_id = f"regime_{regime.lower().replace(' ', '_')}"
            self.regime_agents[regime] = agent_id
            
            # Create specialized environment
            env = self.create_regime_specific_env(filtered_data, regime, **kwargs)
            
            # Create and train agent
            model = self.agent_manager.create_agent(
                agent_id=agent_id,
                env=env,
                model_type='PPO'
            )
            
            self.agent_manager.train_agent(
                agent_id=agent_id,
                total_timesteps=100000,
                eval_freq=5000,
                save_freq=10000
            )
    
    def load_regime_agents(self, regimes):
        """
        Load trained agents for each market regime
        """
        for regime in regimes:
            agent_id = f"regime_{regime.lower().replace(' ', '_')}"
            self.regime_agents[regime] = agent_id
            
            self.agent_manager.load_agent(agent_id, model_type='PPO')
    
    def predict(self, observation, current_regime, deterministic=True):
        """
        Make a prediction using the appropriate regime-specific agent
        """
        if current_regime not in self.regime_agents:
            # Use a default regime if the current one is not available
            logging.warning(f"No agent for regime {current_regime}, using default")
            current_regime = list(self.regime_agents.keys())[0]
        
        agent_id = self.regime_agents[current_regime]
        return self.agent_manager.predict(agent_id, observation, deterministic) 
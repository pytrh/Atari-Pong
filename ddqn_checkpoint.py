# Double Q-Learning with Target Networks and Prioritized Experience Replay (PER)
# Uses two Q-networks that alternate roles, each with their own target network
# Target networks are periodically updated (hard copied) from their corresponding main networks
# Includes prioritized replay buffer with importance sampling

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from replay_buffer import PrioritizedReplayBuffer


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, gamma=0.95, alpha=0.00025, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.05,
                 buffer_capacity=100000, batch_size=32, min_buffer_size=1000,
                 target_update_frequency=1000,
                 per_alpha=0.6, per_alpha_increment=0.0, per_beta=0.4, per_beta_increment=None, 
                 per_epsilon=1e-6, total_training_steps=200000):
        """
        Initialize DQN Agent with Target Network and Prioritized Experience Replay.
        
        Args:
            env: Gym environment
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            min_epsilon: Minimum epsilon value
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            min_buffer_size: Minimum buffer size before training starts
            target_update_frequency: How often to update target network (in training steps)
            per_alpha: PER prioritization exponent (0=uniform, 1=full prioritization)
            per_alpha_increment: Amount to increment alpha per sample (0=no increment)
            per_beta: PER importance sampling correction (0=no correction, 1=full correction)
            per_beta_increment: Amount to increment beta per sample (None=auto-calculate from total_training_steps)
            per_epsilon: Small constant to prevent zero priorities
            total_training_steps: Expected total training steps (used to calculate beta_increment if not provided)
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Calculate beta_increment if not provided
        # Anneal beta from per_beta to 1.0 over the training duration
        if per_beta_increment is None:
            per_beta_increment = (1.0 - per_beta) / total_training_steps
        
        # Replay buffer parameters
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            alpha_increment=per_alpha_increment,
            beta=per_beta,
            beta_increment=per_beta_increment,
            epsilon=per_epsilon
        )
        
        # Target network parameters
        self.target_update_frequency = target_update_frequency
        self.training_steps = 0  # Counter for training steps
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # TODO: Why mps appears to be so much slower than cpu?
        else:
            self.device = torch.device("cpu")
        
        # Double Q-Learning: Two main Q-networks
        self.q_network_1 = DQN(self.state_dim, self.action_dim).to(self.device)
        self.q_network_2 = DQN(self.state_dim, self.action_dim).to(self.device)
        # Two target networks for stable Q-value targets
        self.target_network_1 = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network_2 = DQN(self.state_dim, self.action_dim).to(self.device)
        
        # Initialize target networks with same weights as main networks
        self.update_target_network()
        
        # Separate optimizers for each network using RMSProp
        self.optimizer_1 = optim.RMSprop(self.q_network_1.parameters(), lr=alpha)
        self.optimizer_2 = optim.RMSprop(self.q_network_2.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.episodes_trained = 0

    def update_target_network(self):
        """Copy weights from main networks to target networks (hard update)."""
        self.target_network_1.load_state_dict(self.q_network_1.state_dict())
        self.target_network_2.load_state_dict(self.q_network_2.state_dict())

    def play(self, state):
        """Select action using epsilon-greedy policy based on average of both Q-networks."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                # Use average of both Q-networks for action selection
                q_values_1 = self.q_network_1(state_tensor)
                q_values_2 = self.q_network_2(state_tensor)
                q_values = (q_values_1 + q_values_2) / 2
            return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, state, action, reward, next_state, done):
        """
        Store transition and perform batch update if buffer is ready.
        Uses Double Q-Learning with target networks for stable training.
        Uses prioritized experience replay with importance sampling.
        Returns True if an update was performed, False otherwise.
        """
        # Store the transition in replay buffer
        self.store_transition(state, action, reward, next_state, done)
        
        # Only train if we have enough samples
        if not self.replay_buffer.is_ready(max(self.batch_size, self.min_buffer_size)):
            return False
        
        # Sample a batch from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Double Q-Learning with Target Networks: randomly update either network 1 or network 2
        if np.random.rand() < 0.5:
            # Update Q1: use target_network_2 to select action, target_network_1 to evaluate
            with torch.no_grad():
                q_values_2_next = self.target_network_2(next_states_tensor)
                selected_actions = torch.argmax(q_values_2_next, dim=1)
                q_values_1_next = self.target_network_1(next_states_tensor)
                next_q_values = q_values_1_next.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

            q_values_1 = self.q_network_1(states_tensor)
            q_values = q_values_1.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Calculate TD errors for priority update
            td_errors = (target_q - q_values).detach().cpu().numpy()
            
            # Weighted MSE loss (importance sampling)
            elementwise_loss = (q_values - target_q) ** 2
            loss = torch.mean(elementwise_loss * weights_tensor)

            self.optimizer_1.zero_grad()
            loss.backward()
            # After loss.backward(), before optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.q_network_1.parameters(), max_norm=10.0)
            self.optimizer_1.step()
        else:
            # Update Q2: use target_network_1 to select action, target_network_2 to evaluate
            with torch.no_grad():
                q_values_1_next = self.target_network_1(next_states_tensor)
                selected_actions = torch.argmax(q_values_1_next, dim=1)
                q_values_2_next = self.target_network_2(next_states_tensor)
                next_q_values = q_values_2_next.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

            q_values_2 = self.q_network_2(states_tensor)
            q_values = q_values_2.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Calculate TD errors for priority update
            td_errors = (target_q - q_values).detach().cpu().numpy()
            
            # Weighted MSE loss (importance sampling)
            elementwise_loss = (q_values - target_q) ** 2
            loss = torch.mean(elementwise_loss * weights_tensor)

            self.optimizer_2.zero_grad()
            loss.backward()
            # After loss.backward(), before optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.q_network_2.parameters(), max_norm=10.0)
            self.optimizer_2.step()
        
        # Update priorities in the replay buffer based on TD errors
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Increment training step counter
        self.training_steps += 1
        
        # Periodically update target networks
        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network()
            print(f"  [Target networks updated at step {self.training_steps}]")

        # Decay epsilon at end of episode
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.episodes_trained += 1
        
        return True
    
    def save_best_model(self, episode_reward, save_path='best_model.pth'):
        """
        Save the model if current episode reward is better than best reward.
        
        Args:
            episode_reward: Total reward from the current episode
            save_path: Path where to save the model
        
        Returns:
            bool: True if model was saved, False otherwise
        """
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            # Save all model components
            checkpoint = {
                'episode': self.episodes_trained,
                'best_reward': self.best_reward,
                'q_network_1_state_dict': self.q_network_1.state_dict(),
                'q_network_2_state_dict': self.q_network_2.state_dict(),
                'target_network_1_state_dict': self.target_network_1.state_dict(),
                'target_network_2_state_dict': self.target_network_2.state_dict(),
                'optimizer_1_state_dict': self.optimizer_1.state_dict(),
                'optimizer_2_state_dict': self.optimizer_2.state_dict(),
                'epsilon': self.epsilon,
                'training_steps': self.training_steps,
                'replay_buffer_alpha': self.replay_buffer.alpha,
                'replay_buffer_beta': self.replay_buffer.beta,
            }
            
            torch.save(checkpoint, save_path)
            print(f"  [New best model saved! Reward: {episode_reward:.2f} at episode {self.episodes_trained}]")
            return True
        
        return False
    
    def load_best_model(self, load_path='best_model.pth'):
        """
        Load a saved model checkpoint.
        
        Args:
            load_path: Path to the saved model
        
        Returns:
            dict: Checkpoint information (episode, best_reward, etc.)
        """
        if not os.path.exists(load_path):
            print(f"Warning: Model file {load_path} not found.")
            return None
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load network states
        self.q_network_1.load_state_dict(checkpoint['q_network_1_state_dict'])
        self.q_network_2.load_state_dict(checkpoint['q_network_2_state_dict'])
        self.target_network_1.load_state_dict(checkpoint['target_network_1_state_dict'])
        self.target_network_2.load_state_dict(checkpoint['target_network_2_state_dict'])
        
        # Load optimizer states
        self.optimizer_1.load_state_dict(checkpoint['optimizer_1_state_dict'])
        self.optimizer_2.load_state_dict(checkpoint['optimizer_2_state_dict'])
        
        # Load training state
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.best_reward = checkpoint['best_reward']
        self.episodes_trained = checkpoint['episode']
        
        # Load replay buffer parameters
        self.replay_buffer.alpha = checkpoint['replay_buffer_alpha']
        self.replay_buffer.beta = checkpoint['replay_buffer_beta']
        
        print(f"Model loaded from {load_path}")
        print(f"  Episode: {self.episodes_trained}, Best Reward: {self.best_reward:.2f}")
        print(f"  Training Steps: {self.training_steps}, Epsilon: {self.epsilon:.4f}")
        
        return checkpoint

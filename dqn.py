# DQN implementation with prioritized experience replay and importance sampling

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


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
    def __init__(self, env, gamma=0.95, alpha=0.001, epsilon=0.1, epsilon_decay=0.995, 
                 min_epsilon=0.01, buffer_size=10000, batch_size=32, per_alpha=0.6, 
                 per_epsilon=1e-6, per_beta=0.4, per_beta_increment=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)
        
        # Prioritized Experience Replay parameters
        self.per_alpha = per_alpha  # Priority exponent (0 = uniform, 1 = fully prioritized)
        self.per_epsilon = per_epsilon  # Small constant to avoid zero priority
        self.per_beta = per_beta  # Importance sampling exponent (starts at per_beta, increases to 1.0)
        self.per_beta_increment = per_beta_increment  # How much to increment beta per step
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Experience replay buffer: stores (state, action, reward, next_state, done, priority)
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Main Q-network and target network
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.target_network = DQN(self.state_dim, self.action_dim)
        # Initialize target network with same weights as main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set to evaluation mode (no gradient updates)
        
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        
        # Target network update parameters
        self.update_counter = 0
        self.target_update_frequency = 4000  # Update target network every 10000 updates

    def play(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done, priority=None):
        """Store a transition in the replay buffer with priority"""
        # If no priority provided, use maximum priority (for new experiences)
        if priority is None:
            priority = 1.0  # Maximum priority for new experiences
        self.replay_buffer.append((state, action, reward, next_state, done, priority))

    def sample_batch(self):
        """Sample a mini-batch from the replay buffer using prioritized sampling"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None
        
        # Extract priorities
        priorities = np.array([exp[5] for exp in self.replay_buffer])  # priority is at index 5
        
        # Compute sampling probabilities: P(i) = (p_i^alpha) / sum(p_j^alpha)
        priorities_alpha = priorities ** self.per_alpha
        probabilities = priorities_alpha / priorities_alpha.sum()
        
        # Sample batch_size indices based on probabilities
        indices = np.random.choice(len(self.replay_buffer), size=self.batch_size, p=probabilities, replace=False)
        
        # Get experiences and their sampling probabilities
        experiences = [self.replay_buffer[idx] for idx in indices]
        sample_probs = probabilities[indices]
        
        return experiences, indices, sample_probs

    def compute_importance_weights(self, sample_probs, N):
        """Compute importance sampling weights: w_i = (1/N * 1/P(i))^β"""
        # Avoid division by zero
        sample_probs = np.maximum(sample_probs, 1e-8)
        
        # Compute weights: w_i = (1/N * 1/P(i))^β
        weights = np.power(1.0 / (N * sample_probs), self.per_beta)
        
        # Normalize weights by max weight to stabilize training
        weights = weights / weights.max()
        
        return torch.FloatTensor(weights)

    def update(self, state, action, reward, next_state, done):
        # Store the new transition with maximum priority
        self.store_transition(state, action, reward, next_state, done, priority=1.0)
        
        # Sample a mini-batch from buffer and train on it
        if len(self.replay_buffer) >= self.batch_size:
            batch_data = self.sample_batch()
            if batch_data[0] is not None:
                experiences, indices, sample_probs = batch_data
                
                # Unpack batch
                states = np.array([exp[0] for exp in experiences])
                actions = np.array([exp[1] for exp in experiences])
                rewards = np.array([exp[2] for exp in experiences])
                next_states = np.array([exp[3] for exp in experiences])
                dones = np.array([exp[4] for exp in experiences])
                
                # Convert to tensors
                state_tensors = torch.FloatTensor(states)
                next_state_tensors = torch.FloatTensor(next_states)
                action_tensors = torch.LongTensor(actions)
                reward_tensors = torch.FloatTensor(rewards)
                done_tensors = torch.BoolTensor(dones)
                
                # Compute current Q values
                q_values = self.q_network(state_tensors)  # [batch_size, action_dim]
                q_value = q_values.gather(1, action_tensors.unsqueeze(1)).squeeze(1)  # [batch_size]
                
                # Compute target Q values using target network
                with torch.no_grad():
                    next_q_values = self.target_network(next_state_tensors)  # [batch_size, action_dim]
                    max_next_q = next_q_values.max(1)[0]  # [batch_size]
                    target_q = reward_tensors + (~done_tensors * self.gamma * max_next_q)  # [batch_size]
                
                # Calculate TD errors for priority updates
                td_errors = torch.abs(q_value - target_q).detach().numpy()
                
                # Compute importance sampling weights
                N = len(self.replay_buffer)
                importance_weights = self.compute_importance_weights(sample_probs, N)
                
                # Compute elementwise squared error
                elementwise_loss = (q_value - target_q) ** 2  # Elementwise squared error
                # Apply importance sampling weights
                weighted_loss = (importance_weights * elementwise_loss).mean()
                
                # Update network
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
                
                # Update counter for target network updates
                self.update_counter += 1
                
                # Update target network periodically
                if self.update_counter % self.target_update_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    print(f"Target network updated at step {self.update_counter}")
                
                # Update priorities based on TD error
                new_priorities = td_errors + self.per_epsilon
                for i, idx in enumerate(indices):
                    state_exp, action_exp, reward_exp, next_state_exp, done_exp, _ = self.replay_buffer[idx]
                    self.replay_buffer[idx] = (state_exp, action_exp, reward_exp, next_state_exp, done_exp, new_priorities[i])
                
                # Increment beta (importance sampling exponent)
                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def decay_epsilon(self):
        """Decay epsilon after episode"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

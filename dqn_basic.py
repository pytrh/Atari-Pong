# DQN implementation with prioritized experience replay and importance sampling

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


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


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with importance sampling."""
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=1e-6):
        """
        Args:
            capacity: Maximum size of the buffer
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: How much to increment beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        # Dictionary to store experiences: {index: {'data': (s,a,r,s',done), 'priority': p}}
        self.buffer = {}
        self.max_priority = 1.0
        self.next_idx = 0
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition with maximum priority."""
        data = (state, action, reward, next_state, done)
        # Store raw priority (alpha will be applied during sampling)
        priority = self.max_priority
        
        # If buffer is full, remove oldest entry
        if len(self.buffer) >= self.capacity:
            # Find and remove the oldest entry
            oldest_idx = min(self.buffer.keys())
            removed_priority = self.buffer[oldest_idx]['priority']
            del self.buffer[oldest_idx]
            
            # If removed entry had max priority, recompute max_priority from remaining buffer
            if removed_priority >= self.max_priority and len(self.buffer) > 0:
                self.max_priority = max(self.buffer[idx]['priority'] for idx in self.buffer)
        
        # Add new experience
        self.buffer[self.next_idx] = {
            'data': data,
            'priority': priority
        }
        self.next_idx = (self.next_idx + 1) % (self.capacity * 2)  # Wrap around
    
    def sample(self, batch_size):
        """Sample a batch of transitions with importance sampling weights."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        # Get all indices and priorities
        indices = list(self.buffer.keys())
        priorities = np.array([self.buffer[idx]['priority'] for idx in indices])
        
        # Compute sampling probabilities (apply alpha here)
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices based on probabilities
        sampled_indices = np.random.choice(len(indices), size=batch_size, p=probabilities, replace=False)
        selected_indices = [indices[i] for i in sampled_indices]
        
        # Get samples
        batch = [self.buffer[idx]['data'] for idx in selected_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Compute importance sampling weights
        sampling_probabilities = probabilities[sampled_indices]
        is_weights = np.power(len(self.buffer) * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize by max weight
        is_weights = torch.FloatTensor(is_weights)
        
        # Increment beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, selected_indices, is_weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        for idx, priority in zip(indices, priorities):
            if idx in self.buffer:
                self.buffer[idx]['priority'] = priority
        
        # Recompute max_priority from current buffer to ensure it's accurate
        if len(self.buffer) > 0:
            self.max_priority = max(self.buffer[idx]['priority'] for idx in self.buffer)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env, gamma=0.95, alpha=0.001, epsilon=0.1, epsilon_decay=0.995, 
                 min_epsilon=0.01, replay_buffer_size=10000, batch_size=32,
                 per_alpha=0.6, per_beta=0.4, per_beta_increment=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)
        self.batch_size = batch_size

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss(reduction='none')  # No reduction for IS weighting
        
        # Prioritized experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_buffer_size,
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_increment
        )

    def play(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def store(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, state, action, reward, next_state, done):
        """Store experience in replay buffer (for backward compatibility)."""
        self.store(state, action, reward, next_state, done)
    
    def train(self):
        """Train the network on a batch of experiences from the prioritized replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough experiences yet
        
        # Sample a batch with importance sampling weights
        states, actions, rewards, next_states, dones, indices, is_weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + (~dones).float() * self.gamma * max_next_q
        
        # Compute current Q-values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute TD errors for priority updates
        td_errors = (q_value - target_q).detach().cpu().numpy()
        
        # Compute loss with importance sampling weights
        elementwise_loss = self.loss_fn(q_value, target_q)
        weighted_loss = (elementwise_loss * is_weights).mean()
        
        # Update network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer based on TD errors
        self.replay_buffer.update_priorities(indices, td_errors)
        
        return weighted_loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon when done is True."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


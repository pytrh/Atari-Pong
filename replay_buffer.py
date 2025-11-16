"""
Uniform Replay Buffer implementation using deque.
Used for experience replay in DQN-based algorithms.
"""

import random
import numpy as np
from collections import deque


class UniformReplayBuffer:
    """
    A simple uniform replay buffer using deque for efficient storage.
    
    Stores transitions (state, action, reward, next_state, done) and allows
    for uniform random sampling of batches.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
                Each element is a numpy array of shape (batch_size, ...)
        """
        # Sample random indices
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """
        Check if the buffer has enough samples for a batch.
        
        Args:
            batch_size (int): Desired batch size
            
        Returns:
            bool: True if buffer size >= batch_size
        """
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer using deque for storage.
    
    Samples transitions based on their priorities (typically TD-error).
    Higher priority transitions are sampled more frequently.
    """
    
    def __init__(self, capacity, alpha=0.6, alpha_increment=0.0, beta=0.4, beta_increment=0.0, epsilon=1e-6):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
            alpha (float): Controls how much prioritization to use (0 = uniform, 1 = full prioritization)
            alpha_increment (float): Amount to increment alpha by each sample (default 0.0 = no increment)
            beta (float): Controls importance sampling correction (0 = no correction, 1 = full correction)
            beta_increment (float): Amount to increment beta by each sample (default 0.0 = no increment, should be set based on training duration)
            epsilon (float): Small constant to prevent zero priorities
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.alpha_increment = alpha_increment
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
        # New transitions get maximum priority to ensure they're sampled at least once
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
                - states, actions, rewards, next_states, dones: numpy arrays
                - indices: list of sampled indices (for updating priorities)
                - weights: importance sampling weights
        """
        # Convert priorities to numpy array and apply alpha
        priorities_array = np.array(self.priorities, dtype=np.float32)
        priorities_array = np.power(priorities_array, self.alpha)
        
        # Calculate sampling probabilities
        probabilities = priorities_array / np.sum(priorities_array)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        # Weight = (1 / (N * P(i)))^beta
        total_samples = len(self.buffer)
        weights = np.power(total_samples * probabilities[indices], -self.beta)
        # Normalize weights by max weight for stability
        weights = weights / np.max(weights)
        
        # Increment alpha and beta
        self.alpha = min(1.0, self.alpha + self.alpha_increment)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get the actual transitions
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        weights = np.array(weights, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: List or array of indices to update
            priorities: New priority values (typically TD-errors)
        """
        for idx, priority in zip(indices, priorities):
            # Add epsilon to prevent zero priorities
            priority = abs(priority) + self.epsilon
            self.priorities[idx] = priority
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """
        Check if the buffer has enough samples for a batch.
        
        Args:
            batch_size (int): Desired batch size
            
        Returns:
            bool: True if buffer size >= batch_size
        """
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all transitions and priorities from the buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.max_priority = 1.0


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


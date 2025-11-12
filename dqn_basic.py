# DQN basic implementation without experience replay and target network

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# choose device: cuda if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ReplayBuffer:
    """Simple uniform experience replay buffer."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions uniformly at random."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


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
    def __init__(self, env, gamma=0.95, alpha=0.001, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 replay_buffer_size=10000, batch_size=32):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)
        self.batch_size = batch_size

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # create network and move to device
        self.q_network = DQN(self.state_dim, self.action_dim).to(device)
        # create optimizer after model is on device
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

    def play(self, state):
        # ensure state tensor is on the selected device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, state_dim]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, state=None, action=None, reward=None, next_state=None, done=None):
        """
        Update the Q-network using a batch from the replay buffer.
        If individual transition is provided, store it and update if buffer has enough samples.
        """
        # Store transition if provided
        if state is not None:
            self.store_transition(state, action, reward, next_state, done)
        
        # Only update if we have enough samples in the buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors and move to device
        states_tensor = torch.FloatTensor(states).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        
        # Compute Q-values for current states
        q_values = self.q_network(states_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states_tensor)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards_tensor + (~dones_tensor).float() * self.gamma * max_next_q
        
        # Compute loss and update
        loss = self.loss_fn(q_value, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon if episode is done (only if transition was provided)
        if done and state is not None:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


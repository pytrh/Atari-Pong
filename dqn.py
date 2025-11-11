# DQN implementation with experience replay (single-sample updates)

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

class DQNAgent:
    def __init__(self, env, gamma=0.95, alpha=0.001, epsilon=0.1, epsilon_decay=0.995, 
                 min_epsilon=0.01, buffer_size=10000, per_alpha=0.6, per_epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)
        
        # Prioritized Experience Replay parameters
        self.per_alpha = per_alpha  # Priority exponent (0 = uniform, 1 = fully prioritized)
        self.per_epsilon = per_epsilon  # Small constant to avoid zero priority

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Experience replay buffer: stores (state, action, reward, next_state, done, priority)
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

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

    def sample_experience(self):
        """Sample a single experience from the replay buffer using prioritized sampling"""
        if len(self.replay_buffer) == 0:
            return None, None
        
        # Extract priorities
        priorities = np.array([exp[5] for exp in self.replay_buffer])  # priority is at index 5
        
        # Compute sampling probabilities: P(i) = (p_i^alpha) / sum(p_j^alpha)
        priorities_alpha = priorities ** self.per_alpha
        probabilities = priorities_alpha / priorities_alpha.sum()
        
        # Sample based on probabilities
        idx = np.random.choice(len(self.replay_buffer), p=probabilities)
        experience = self.replay_buffer[idx]
        
        return experience, idx

    def update(self, state, action, reward, next_state, done):
        # Store the new transition with maximum priority
        self.store_transition(state, action, reward, next_state, done, priority=1.0)
        
        # Sample a prioritized experience from buffer and train on it
        if len(self.replay_buffer) > 0:
            experience, idx = self.sample_experience()
            if experience is not None:
                state_exp, action_exp, reward_exp, next_state_exp, done_exp, old_priority = experience
                
                state_tensor = torch.FloatTensor(state_exp).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state_exp).unsqueeze(0)

                with torch.no_grad(): # no grad since it is the target, a fixed value
                    max_next_q = torch.max(self.q_network(next_state_tensor))
                    target_q = reward_exp + (0 if done_exp else self.gamma * max_next_q.item())

                target_q = torch.tensor([target_q], dtype=torch.float32)

                q_values = self.q_network(state_tensor)
                q_value = q_values[0, action_exp].unsqueeze(0)

                # Calculate TD error for priority update
                td_error = abs((q_value - target_q).item())
                new_priority = td_error + self.per_epsilon

                loss = self.loss_fn(q_value, target_q)
                self.optimizer.zero_grad() # clears old gradients from the last step
                loss.backward() # computes the gradients of the loss w.r.t. each parameter and adds them to the .grad attribute of each parameter.
                self.optimizer.step()
                
                # Update priority of the sampled experience
                self.replay_buffer[idx] = (state_exp, action_exp, reward_exp, next_state_exp, done_exp, new_priority)

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def decay_epsilon(self):
        """Decay epsilon after episode"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

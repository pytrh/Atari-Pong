# Double Q-Learning with Uniform Experience Replay
# Uses two Q-networks that alternate roles: one selects actions, the other evaluates them
# Includes uniform replay buffer for experience replay

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import UniformReplayBuffer


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
                 buffer_capacity=10000, batch_size=32, min_buffer_size=1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Replay buffer parameters
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.replay_buffer = UniformReplayBuffer(capacity=buffer_capacity)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # TODO: Why mps appears to be so much slower than cpu?
        else:
            self.device = torch.device("cpu")
        
        self.q_network_1 = DQN(self.state_dim, self.action_dim).to(self.device)
        self.q_network_2 = DQN(self.state_dim, self.action_dim).to(self.device)
        # Separate optimizers for each network using RMSProp
        self.optimizer_1 = optim.RMSprop(self.q_network_1.parameters(), lr=alpha)
        self.optimizer_2 = optim.RMSprop(self.q_network_2.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def play(self, state):
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
        Returns True if an update was performed, False otherwise.
        """
        # Store the transition in replay buffer
        self.store_transition(state, action, reward, next_state, done)
        
        # Only train if we have enough samples
        if not self.replay_buffer.is_ready(max(self.batch_size, self.min_buffer_size)):
            return False
        
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Double Q-Learning: randomly update either network 1 or network 2 with 0.5 probability
        if np.random.rand() < 0.5:
            # Update Q1: use Q2 to select action, Q1 to evaluate
            with torch.no_grad():
                q_values_2_next = self.q_network_2(next_states_tensor)
                selected_actions = torch.argmax(q_values_2_next, dim=1)
                q_values_1_next = self.q_network_1(next_states_tensor)
                next_q_values = q_values_1_next.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

            q_values_1 = self.q_network_1(states_tensor)
            q_values = q_values_1.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            loss = self.loss_fn(q_values, target_q)

            self.optimizer_1.zero_grad()
            loss.backward()
            self.optimizer_1.step()
        else:
            # Update Q2: use Q1 to select action, Q2 to evaluate
            with torch.no_grad():
                q_values_1_next = self.q_network_1(next_states_tensor)
                selected_actions = torch.argmax(q_values_1_next, dim=1)
                q_values_2_next = self.q_network_2(next_states_tensor)
                next_q_values = q_values_2_next.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

            q_values_2 = self.q_network_2(states_tensor)
            q_values = q_values_2.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            loss = self.loss_fn(q_values, target_q)

            self.optimizer_2.zero_grad()
            loss.backward()
            self.optimizer_2.step()

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return True


# Double Q-Learning basic implementation without experience replay
# Uses two Q-networks that alternate roles: one selects actions, the other evaluates them
# Using device selection: CUDA > MPS > CPU

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    def __init__(self, env, gamma=0.95, alpha=0.001, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
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

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Double Q-Learning: randomly update either network 1 or network 2 with 0.5 probability
        if np.random.rand() < 0.5:
            # Update Q1: use Q2 to select action, Q1 to evaluate
            with torch.no_grad():
                q_values_2_next = self.q_network_2(next_state_tensor)
                selected_action = torch.argmax(q_values_2_next).item()
                q_values_1_next = self.q_network_1(next_state_tensor)
                next_q_value = q_values_1_next[0, selected_action].item()
                target_q = reward + (0 if done else self.gamma * next_q_value)

            target_q = torch.tensor([target_q], dtype=torch.float32).to(self.device)
            q_values_1 = self.q_network_1(state_tensor)
            q_value = q_values_1[0, action].unsqueeze(0)
            loss = self.loss_fn(q_value, target_q)

            self.optimizer_1.zero_grad()
            loss.backward()
            self.optimizer_1.step()
        else:
            # Update Q2: use Q1 to select action, Q2 to evaluate
            with torch.no_grad():
                q_values_1_next = self.q_network_1(next_state_tensor)
                selected_action = torch.argmax(q_values_1_next).item()
                q_values_2_next = self.q_network_2(next_state_tensor)
                next_q_value = q_values_2_next[0, selected_action].item()
                target_q = reward + (0 if done else self.gamma * next_q_value)

            target_q = torch.tensor([target_q], dtype=torch.float32).to(self.device)
            q_values_2 = self.q_network_2(state_tensor)
            q_value = q_values_2[0, action].unsqueeze(0)
            loss = self.loss_fn(q_value, target_q)

            self.optimizer_2.zero_grad()
            loss.backward()
            self.optimizer_2.step()

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


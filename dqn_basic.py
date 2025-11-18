# DQN basic implementation without experience replay and target network

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
        
        # Device selection: CUDA > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def play(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        with torch.no_grad(): # no grad since it is the target, a fixed value
            max_next_q = torch.max(self.q_network(next_state_tensor))
            target_q = reward + (0 if done else self.gamma * max_next_q.item())

        target_q = torch.tensor([target_q], dtype=torch.float32).to(self.device)

        q_values = self.q_network(state_tensor)
        q_value = q_values[0, action].unsqueeze(0)

        loss = self.loss_fn(q_value, target_q)
        self.optimizer.zero_grad() # clears old gradients from the last step
        loss.backward() # computes the gradients of the loss w.r.t. each parameter and adds them to the .grad attribute of each parameter.
        self.optimizer.step()

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


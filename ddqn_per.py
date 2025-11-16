# Double Q-Learning with Prioritized Experience Replay (PER)
# Uses two Q-networks that alternate roles: one selects actions, the other evaluates them
# Includes prioritized replay buffer with importance sampling

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, env, gamma=0.95, alpha=0.001, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
                 buffer_capacity=10000, batch_size=32, min_buffer_size=1000,
                 per_alpha=0.6, per_alpha_increment=0.0, per_beta=0.4, per_beta_increment=None, 
                 per_epsilon=1e-6, total_training_steps=200000):
        """
        Initialize DDQN Agent with Prioritized Experience Replay.
        
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
            
            # Calculate TD errors for priority update
            td_errors = (target_q - q_values).detach().cpu().numpy()
            
            # Weighted MSE loss (importance sampling)
            elementwise_loss = (q_values - target_q) ** 2
            loss = torch.mean(elementwise_loss * weights_tensor)

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
            
            # Calculate TD errors for priority update
            td_errors = (target_q - q_values).detach().cpu().numpy()
            
            # Weighted MSE loss (importance sampling)
            elementwise_loss = (q_values - target_q) ** 2
            loss = torch.mean(elementwise_loss * weights_tensor)

            self.optimizer_2.zero_grad()
            loss.backward()
            self.optimizer_2.step()
        
        # Update priorities in the replay buffer based on TD errors
        self.replay_buffer.update_priorities(indices, td_errors)

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return True


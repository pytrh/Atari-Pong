# Dueling Double Q-Learning with Target Networks and Prioritized Experience Replay (PER)
# Drop-in replacement for DQNAgent: splits the Q-network into a state-value stream
# and a state-dependent action-advantage stream, recombined as
#     Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
# All Double-Q / PER / target-network machinery is inherited unchanged from DQNAgent,
# so pong.py and lander.py can adopt this by swapping a single import line:
#     from dueling_ddqn import DuelingDQNAgent as DQNAgent

import torch
import torch.nn as nn
import torch.optim as optim

from ddqn_checkpoint import DQNAgent


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Mean-subtraction aggregation for identifiability (Wang et al., 2016)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DuelingDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

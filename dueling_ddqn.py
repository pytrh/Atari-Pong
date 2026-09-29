"""Dueling Double Q-Learning implementation with target networks and prioritized experience replay.

This module provides a drop-in replacement for DQNAgent that uses dueling network architecture,
splitting the Q-network into separate value and advantage streams. The dueling architecture
computes Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a')) for improved learning stability.
All Double-Q learning, prioritized experience replay, and target network functionality is
inherited unchanged from the base DQNAgent class.
"""

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
    """A Dueling Deep Q-Network implementation that separates value and advantage estimation.

    This neural network architecture splits Q-value computation into two streams:
    a value stream that estimates the state value V(s) and an advantage stream 
    that estimates the advantage A(s,a) for each action. The final Q-values are
    computed by combining these streams using mean-subtraction aggregation to
    ensure identifiability as described in Wang et al. (2016).

    The network consists of shared feature layers followed by separate value
    and advantage streams, each with their own fully connected layers.
    """
    def __init__(self, input_dim, output_dim):
        """Initialize a Dueling Deep Q-Network with separate value and advantage streams.

        Args:
            input_dim (int): Dimension of the input state space.
            output_dim (int): Dimension of the output action space.
        """
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
        """Forward pass through the Dueling DQN network.

        Computes Q-values by combining value and advantage streams using mean-subtraction
        aggregation for identifiability as described in Wang et al., 2016.

        Args:
            x: Input tensor, typically representing state observations.

        Returns:
            Tensor of Q-values with the same batch size as input, where each row
            contains Q-values for all possible actions.
        """
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Mean-subtraction aggregation for identifiability (Wang et al., 2016)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DuelingDQNAgent(DQNAgent):
    """A Deep Q-Network agent that uses dueling network architecture to separately estimate state values and action advantages for improved learning stability and performance."""
    def __init__(self, *args, **kwargs):
        """Initialize a DuelingDQNAgent instance.

        Args:
            *args: Variable length argument list passed to parent class.
            **kwargs: Arbitrary keyword arguments passed to parent class.
        """
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

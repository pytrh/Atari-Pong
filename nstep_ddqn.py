"""N-step prioritized experience replay buffer for deep reinforcement learning.

This module implements an N-step variant of prioritized experience replay that computes
multi-step returns by accumulating discounted rewards over N consecutive transitions.
The buffer maintains a queue of recent transitions and calculates N-step targets for
more efficient learning in deep Q-networks and similar algorithms.
"""

from collections import deque

from replay_buffer import PrioritizedReplayBuffer
from ddqn_checkpoint import DQNAgent


class NStepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """A prioritized replay buffer that implements n-step temporal difference learning.

    Extends PrioritizedReplayBuffer to accumulate rewards over n consecutive time steps
    using discounted returns. Transitions are stored only after collecting n steps,
    with rewards computed as the sum of discounted future rewards up to n steps ahead
    or until a terminal state is reached.

    Args:
        capacity: Maximum number of transitions to store in the buffer.
        n_step: Number of steps to look ahead for reward accumulation (default: 3).
        gamma: Discount factor for future rewards (default: 0.99).
        alpha: Prioritization exponent controlling sampling probability (default: 0.6).
        alpha_increment: Amount to increment alpha per sample (default: 0.0).
        beta: Importance sampling correction exponent (default: 0.4).
        beta_increment: Amount to increment beta per sample (default: 0.0).
        epsilon: Small constant to prevent zero priorities (default: 1e-6).
    """
    def __init__(
        self,
        capacity,
        n_step=3,
        gamma=0.99,
        alpha=0.6,
        alpha_increment=0.0,
        beta=0.4,
        beta_increment=0.0,
        epsilon=1e-6,
    ):
        """Initialize an N-step prioritized experience replay buffer.

        Args:
            capacity: Maximum number of experiences to store in the buffer.
            n_step: Number of steps for n-step returns calculation.
            gamma: Discount factor for future rewards.
            alpha: Prioritization exponent controlling how much prioritization is used.
            alpha_increment: Amount to increment alpha by over time.
            beta: Importance sampling exponent for correcting bias from prioritized sampling.
            beta_increment: Amount to increment beta by over time.
            epsilon: Small constant added to priorities to ensure non-zero sampling probability.
        """
        super(NStepPrioritizedReplayBuffer, self).__init__(
            capacity=capacity,
            alpha=alpha,
            alpha_increment=alpha_increment,
            beta=beta,
            beta_increment=beta_increment,
            epsilon=epsilon,
        )
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_queue = deque(maxlen=n_step)

    def _get_n_step_info(self):
        # Accumulate the discounted reward over the queued transitions and take the
        # final (next_state, done) as the n-step landing point. If any intermediate
        # transition is terminal, the return is truncated at that point.
        """Calculate n-step return and terminal state from queued transitions.

        Accumulates discounted rewards over queued transitions using the discount factor gamma.
        If any intermediate transition is terminal, the return is truncated at that point and
        the terminal state becomes the landing point.

        Returns:
            tuple: A 3-tuple containing (n_step_reward, next_state, done) where n_step_reward
                is the accumulated discounted reward, next_state is the final or first terminal
                state encountered, and done is the terminal flag.
        """
        reward, next_state, done = self.n_step_queue[-1][2:]
        for transition in reversed(list(self.n_step_queue)[:-1]):
            r, n_s, d = transition[2:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = n_s, d
        return reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the n-step replay buffer queue and store complete n-step transitions.

        Args:
            state: Current state observation
            action: Action taken in the current state
            reward: Reward received for the action
            next_state: Next state observation after taking the action
            done: Boolean indicating if the episode is complete
        """
        self.n_step_queue.append((state, action, reward, next_state, done))

        # Only emit a transition once we have a full n-step window
        if len(self.n_step_queue) < self.n_step:
            return

        n_reward, n_next_state, n_done = self._get_n_step_info()
        n_state, n_action = self.n_step_queue[0][:2]
        super(NStepPrioritizedReplayBuffer, self).push(
            n_state, n_action, n_reward, n_next_state, n_done
        )

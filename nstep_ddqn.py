from collections import deque

from replay_buffer import PrioritizedReplayBuffer
from ddqn_checkpoint import DQNAgent


class NStepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
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
        reward, next_state, done = self.n_step_queue[-1][2:]
        for transition in reversed(list(self.n_step_queue)[:-1]):
            r, n_s, d = transition[2:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = n_s, d
        return reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_step_queue.append((state, action, reward, next_state, done))

        # Only emit a transition once we have a full n-step window
        if len(self.n_step_queue) < self.n_step:
            return

        n_reward, n_next_state, n_done = self._get_n_step_info()
        n_state, n_action = self.n_step_queue[0][:2]
        super(NStepPrioritizedReplayBuffer, self).push(
            n_state, n_action, n_reward, n_next_state, n_done
        )

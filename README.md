# Atari-Pong

> Documentation générée automatiquement par l'agent de documentation (Amazon Bedrock AgentCore).

## Modules

### `ddqn_checkpoint.py`

Double Q-Learning agent implementation with target networks and prioritized experience replay.

**API publique :** `DQN`, `DQNAgent`

### `dueling_ddqn.py`

**API publique :** `DuelingDQN`, `DuelingDQNAgent`

### `lander.py`

Training script for a Double Deep Q-Network (DDQN) agent on the LunarLander-v3 environment.

### `pong.py`

Training module for a Deep Q-Network (DQN) agent on the Atari Pong environment.

### `replay_buffer.py`

Uniform Replay Buffer implementation using deque.

**API publique :** `UniformReplayBuffer`, `PrioritizedReplayBuffer`

---

_Ce README est maintenu par l'agent de documentation : à chaque push sur `main`, l'agent régénère la documentation et ouvre une pull request._

import datetime
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
import numpy as np

from ddqn_per import DQNAgent

import matplotlib.pyplot as plt
import torch


# --------- Saving Policy ---------
save_policy = True       # Enable/Disable saving
save_every = 300        # Save every X episodes
PARAM = "test_numero_42_"  # Description of the parameters
save_path = "policies/pong_" + PARAM
add_info = {}


# ------------------ Environment and Agent Setup ------------------

gym.register_envs(ale_py)
env_name = 'ALE/Pong-v5'
add_info = {'obs_type': "ram"}

env = gym.make(env_name, render_mode="human", **add_info)
env = FrameStackObservation(env, stack_size=4, padding_type="zero")
env = FlattenObservation(env)

episodes = 100000
render_every = 1000
how_much_to_render = 0
rewards = []

agent = DQNAgent(
    env,
    gamma=0.99,
    alpha=0.0005,
    epsilon=1.0,
    epsilon_decay=0.99995,
    min_epsilon=0.01,
    per_alpha=0.6,
    per_beta=0.4,
    total_training_steps=episodes * 5000  # Approx 5k-10k steps per Pong episode
)

print(f"Using device: {agent.device}")

# ---------------- Main Training Loop ------------------
avg_rewards = 0
number_episodes = 0

for episode in range(episodes):
    if episode % render_every < how_much_to_render and episode > 99:
        env = gym.make(env_name, render_mode="human", **add_info)
    else:
        env = gym.make(env_name, **add_info)

    env = FrameStackObservation(env, stack_size=4, padding_type="zero")
    env = FlattenObservation(env)
    
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done and total_reward < 500:
        action = agent.play(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    avg_rewards += 1/ (number_episodes + 1) * (total_reward - avg_rewards)
    if number_episodes > 100:

        avg_rewards = 0
        number_episodes = 0
    number_episodes += 1

    print(f"Episode {episode} | Avg Reward: {avg_rewards:.2f} | Epsilon: {agent.epsilon:.3f} | Total Reward: {total_reward:.2f}")
    rewards.append(total_reward)

    # ---- Save policy periodically ----
    save_path_2 = save_path + f"avg{int(avg_rewards)}_ep{episode}.pth"
    if save_policy and episode % save_every == 0 and episode > 0:
        if hasattr(agent, "q_network"):
            torch.save(agent.q_network.state_dict(), save_path_2)
            print(f"Policy (q_network) saved at episode {episode} -> {save_path_2}")
        elif hasattr(agent, "model"):
            torch.save(agent.model.state_dict(), save_path_2)
            print(f"Policy (model) saved at episode {episode} -> {save_path_2}")
        else:
            print("No neural network found in agent, skipping save...")

    env.close()

# Plot average reward over last 100 episodes
avg_rewards = [np.mean(rewards[max(0, i - 100): i + 1]) for i in range(len(rewards))]
plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward (100 ep)")
plt.title("Pong with Double Q-Learning and Uniform Experience Replay")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"plots/pong_ddqn_uniform_avg_rewards_{timestamp}.png")
plt.show()

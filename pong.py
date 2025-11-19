import datetime
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
import numpy as np

from ddqn_checkpoint import DQNAgent

import matplotlib.pyplot as plt
import torch

# --------- Saving Policy ---------
PARAM = "test_42_"  # Description of the parameters
save_path = "policies/pong_" + PARAM
save_plot_every = 100
add_info = {}

# ------------------ Environment and Agent Setup ------------------

gym.register_envs(ale_py)
env_name = 'ALE/Pong-v5'
add_info = {'obs_type': "ram"}

env = gym.make(env_name, render_mode=None, **add_info)
env = FrameStackObservation(env, stack_size=4, padding_type="zero")
env = FlattenObservation(env)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
total_training_steps=400000
render_every = 1
how_much_to_render = 1
rewards = []

agent = DQNAgent(
    env,
    gamma=0.99,
    alpha=0.00025,
    epsilon=1.0,
    epsilon_decay=0.999996,
    min_epsilon=0.01,
    per_alpha=0.6,
    per_beta=0.4,
    target_update_frequency=1000,
    total_training_steps=total_training_steps,
    batch_size=128,
    buffer_capacity=500000,
)

print(f"Using device: {agent.device}")

# ---------------- Main Training Loop ------------------
avg_rewards = 0
episode = 0

# Optional: Load previous best model
agent.load_best_model('policies/pong_best_avg_time20251119_115820.pth')

while agent.training_steps < total_training_steps:
    if episode % render_every < how_much_to_render:
        env = gym.make(env_name, render_mode="human", **add_info)
    else:
        env = gym.make(env_name, render_mode=None, **add_info)
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
    
    if len(rewards) >=100:
        avg_rewards = np.mean(rewards[-100:])
    else:
        avg_rewards = np.mean(rewards)

    # Save best models automatically (both episode and average)
    agent.save_best_model(total_reward, save_path=f'policies/pong_best_episode_time{timestamp}.pth')
    agent.save_best_average_model(avg_rewards, save_path=f'policies/pong_best_avg_time{timestamp}.pth')

    print(f"Step {agent.training_steps} | Episode {episode} | Avg {avg_rewards:.2f} | Best Ep {agent.best_reward:.2f} | Best Avg {agent.best_avg_reward:.2f} | Epsilon {agent.epsilon:.3f}")
    # print(f"PER Beta {agent.replay_buffer.per_beta}")
    rewards.append(total_reward)
    episode = episode + 1
    env.close()

    # ---- Save plots (average reward over last 100 episodes) periodically ----
    if episode % save_plot_every == 0 and episode > 0:
        avg_rewards = [np.mean(rewards[max(0, i - 100): i + 1]) for i in range(len(rewards))]
        plt.plot(avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward (100 ep)")
        plt.title("Pong with Double Q-Learning, PER and Target Networks")
        plt.savefig(f"plots/pong_{timestamp}.png")

    # Save last plot
    avg_rewards = [np.mean(rewards[max(0, i - 100): i + 1]) for i in range(len(rewards))]
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (100 ep)")
    plt.title("Pong with Double Q-Learning, PER and Target Networks")
    plt.savefig(f"plots/pong_{timestamp}.png")

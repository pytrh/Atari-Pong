import datetime
import gymnasium as gym
import numpy as np

from ddqn_checkpoint import DQNAgent

import matplotlib.pyplot as plt
import torch


# --------- Saving Policy ---------
PARAM = "test_42_"  # Description of the parameters
save_path = "policies/lunar_" + PARAM
save_plot_every = 500
add_info = {}


# ------------------ Environment and Agent Setup ------------------

env_name = "LunarLander-v3"
env = gym.make(env_name)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
total_training_steps=300000
# episodes = 10000
render_every = 100000
how_much_to_render = 0
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
    device="cuda",
)

print(f"Using device: {agent.device}")

# Optional: Load previous best model if it exists
# agent.load_best_model('policies/lunar_best_model.pth')
# agent.load_best_model('policies/lunar_best_avg.pth')

# ---------------- Main Training Loop ------------------
avg_rewards = 0

episode = 0
while agent.training_steps < total_training_steps:
    if episode % render_every < how_much_to_render and episode > 99:
        env = gym.make(env_name, render_mode="human", **add_info)
    else:
        env = gym.make(env_name, **add_info)

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

    # Save best models automatically (both episode and average)
    agent.save_best_model(total_reward, save_path='policies/lunar_best_episode_time{timestamp}.pth')
    agent.save_best_average_model(avg_rewards, save_path='policies/lunar_best_avg_time{timestamp}.pth')

    # ---- Save plots (average reward over last 100 episodes) periodically ----
    if episode % save_plot_every == 0 and episode > 0:
        avg_rewards = [np.mean(rewards[max(0, i - 100): i + 1]) for i in range(len(rewards))]
        plt.plot(avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward (100 ep)")
        plt.title("Lunar Lander with Double Q-Learning, PER and Target Networks")
        plt.savefig(f"plots/lunar_{timestamp}.png")

    print(f"Step {agent.training_steps} | Episode {episode} | Avg: {avg_rewards:.2f} | Best Ep: {agent.best_reward:.2f} | Best Avg: {agent.best_avg_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:.2f}")
    rewards.append(total_reward)
    episode = episode + 1
    env.close()

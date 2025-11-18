import datetime
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
import numpy as np

from ddqn_checkpoint import DQNAgent

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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
total_training_steps=1000000
# episodes = 10000
render_every = 1000
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
# agent.load_best_model('policies/pong_best_model.pth')

# ---------------- Main Training Loop ------------------
avg_rewards = 0

episode = 0
while agent.training_steps < total_training_steps:
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
    
    if len(rewards) >=100:
        avg_rewards = np.mean(rewards[-100:])

    # Save best models automatically (both episode and average)
    agent.save_best_model(total_reward, save_path='policies/pong_best_episode_time{timestamp}.pth')
    agent.save_best_average_model(avg_rewards, save_path='policies/pong_best_avg_time{timestamp}.pth')

    print(f"Step {agent.training_steps} | Episode {episode} | Avg: {avg_rewards:.2f} | Best Ep: {agent.best_reward:.2f} | Best Avg: {agent.best_avg_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:.2f}")
    rewards.append(total_reward)
    episode = episode + 1
    env.close()

    # ---- Save policy periodically ----
#    save_path_2 = save_path + f"avg{int(avg_rewards)}_ep{episode}_time{timestamp}"
#    if save_policy and episode % save_every == 0 and episode > 0:
#        if hasattr(agent, "q_network"):
#            torch.save(agent.q_network.state_dict(), save_path_2)
#            print(f"Policy (q_network) saved at episode {episode} -> {save_path_2}")
#        if hasattr(agent, "q_network_1"):
#            save_path_3 = save_path_2 + f"q1.pth"
#            torch.save(agent.q_network_1.state_dict(), save_path_3)
#            print(f"Policy (q_network_1) saved at episode {episode} -> {save_path_2}")
#        if hasattr(agent, "q_network_2"):
#            save_path_3 = save_path_2 + f"q2.pth"
#            torch.save(agent.q_network_2.state_dict(), save_path_3)
#            print(f"Policy (q_network_2) saved at episode {episode} -> {save_path_2}")
#        if hasattr(agent, "model"):
#            torch.save(agent.model.state_dict(), save_path_2)
#            print(f"Policy (model) saved at episode {episode} -> {save_path_2}")

# Plot average reward over last 100 episodes
avg_rewards = [np.mean(rewards[max(0, i - 100): i + 1]) for i in range(len(rewards))]
plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward (100 ep)")
plt.title("Pong with Double Q-Learning, PER and Target Networks")
plt.savefig(f"plots/pong_{timestamp}.png")

import gymnasium as gym
import ale_py
import torch
import os
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
from dqn import DQNAgent

# Setup environment (same as training)
gym.register_envs(ale_py)
env_name = 'ALE/Pong-v5'
add_info = {'obs_type': "ram"}

env = gym.make(env_name, render_mode="human", **add_info)
env = FrameStackObservation(env, stack_size=4, padding_type="zero")
env = FlattenObservation(env)

# Create agent (same parameters as training)
agent = DQNAgent(
    env,
    gamma=0.99,
    alpha=0.0005,
    epsilon=0.0,  # Set to 0 for evaluation (no exploration)
    epsilon_decay=0.995,
    min_epsilon=0.01
)

# Load the saved policy
policy_path = "policies/pong_test_numero_42_best.pth"  # Change to your saved file
# Or use: policy_path = "policies/pong_test_numero_42_avg-20_ep300.pth"

if not os.path.exists(policy_path):
    print(f"❌ Policy file not found: {policy_path}")
    print("Available policy files:")
    if os.path.exists("policies"):
        for f in os.listdir("policies"):
            if f.endswith(".pth"):
                print(f"  - policies/{f}")
    exit(1)

agent.q_network.load_state_dict(torch.load(policy_path, map_location='cpu'))
agent.target_network.load_state_dict(torch.load(policy_path, map_location='cpu'))
agent.q_network.eval()  # Set to evaluation mode
agent.target_network.eval()

print(f"✅ Loaded policy from {policy_path}")
print("Testing policy...\n")

# Test the policy
num_episodes = 10
total_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 1000:  # Max 1000 steps per episode
        action = agent.play(state)  # Will use epsilon=0, so always greedy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        steps += 1
    
    total_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

avg_reward = sum(total_rewards) / len(total_rewards)
print(f"\n{'='*50}")
print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
print(f"Best episode reward: {max(total_rewards):.2f}")
print(f"Worst episode reward: {min(total_rewards):.2f}")
print(f"{'='*50}")

env.close()


import gymnasium as gym
import ale_py
import numpy as np

from gymnasium.wrappers import FrameStackObservation, FlattenObservation

from dqn import DQNAgent

import matplotlib.pyplot as plt
import torch
import os
import json
from datetime import datetime


# --------- Policy Loading/Resuming ---------
load_policy = True  # Set to True to auto-detect and resume from best policy's checkpoint
load_optimizer = True  # Set to True to also load optimizer state (if saved)

# --------- Saving Policy ---------
save_policy = True       # Enable/Disable saving
save_every = 100        # Save every X episodes (reduced for more frequent saves)
save_best = True         # Save best policy separately
PARAM = "test_numero_42_"  # Description of the parameters

# Add date and time to filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"policies/pong_{PARAM}{timestamp}_"
best_policy_path = f"policies/pong_{PARAM}{timestamp}_best.pth"
checkpoint_path = f"policies/pong_{PARAM}{timestamp}_checkpoint.pth"  # Full checkpoint with optimizer
add_info = {}

# Create policies directory if it doesn't exist
os.makedirs("policies", exist_ok=True)

# ------------------ Environment and Agent Setup ------------------

gym.register_envs(ale_py)
env_name = 'ALE/Pong-v5'
# env_name = "ALE/Breakout-v5"
add_info = {'obs_type': "ram"}

env = gym.make(env_name, render_mode="human", **add_info)
env = FrameStackObservation(env, stack_size=4, padding_type="zero")
env = FlattenObservation(env)


agent = DQNAgent(
    env,
    gamma=0.99,
    alpha=0.0005,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.01
)

# Auto-detect and load best policy's checkpoint
start_episode = 0
best_avg_reward = float('-inf')

def find_most_recent_best_policy():
    """Find the most recent best policy file and return its path and timestamp"""
    if not os.path.exists("policies"):
        return None, None
    
    # Pattern: pong_{PARAM}{timestamp}_best.pth
    pattern = f"pong_{PARAM}"
    best_policies = []
    
    for filename in os.listdir("policies"):
        if filename.startswith(pattern) and filename.endswith("_best.pth"):
            filepath = os.path.join("policies", filename)
            # Extract timestamp: between PARAM and "_best"
            # Format: pong_{PARAM}{timestamp}_best.pth
            prefix_len = len(pattern)
            suffix = "_best.pth"
            if len(filename) > prefix_len + len(suffix):
                timestamp = filename[prefix_len:-len(suffix)]
                mtime = os.path.getmtime(filepath)
                best_policies.append((filepath, timestamp, mtime))
    
    if not best_policies:
        return None, None
    
    # Sort by modification time (most recent first)
    best_policies.sort(key=lambda x: x[2], reverse=True)
    return best_policies[0][0], best_policies[0][1]  # Return path and timestamp

if load_policy:
    best_policy_file, detected_timestamp = find_most_recent_best_policy()
    
    if best_policy_file and detected_timestamp:
        # Find corresponding checkpoint
        detected_checkpoint = f"policies/pong_{PARAM}{detected_timestamp}_checkpoint.pth"
        
        if os.path.exists(detected_checkpoint):
            print(f"🔍 Found best policy: {best_policy_file}")
            print(f"🔍 Found corresponding checkpoint: {detected_checkpoint}")
            
            checkpoint = torch.load(detected_checkpoint, map_location='cpu')
            agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Loaded optimizer state")
            
            start_episode = checkpoint.get('episode', 0)
            best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
            agent.update_counter = checkpoint.get('update_counter', 0)
            agent.per_beta = checkpoint.get('per_beta', agent.per_beta)
            agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
            
            print(f"✅ Resumed from checkpoint: Episode {start_episode}, Best Reward: {best_avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        else:
            # Fallback: load just the best policy (no checkpoint found)
            print(f"⚠️ Checkpoint not found for {detected_timestamp}, loading best policy only")
            agent.q_network.load_state_dict(torch.load(best_policy_file, map_location='cpu'))
            agent.target_network.load_state_dict(torch.load(best_policy_file, map_location='cpu'))
            print(f"✅ Loaded best policy from {best_policy_file}")
    else:
        print("ℹ️ No previous best policy found, starting fresh training")


episodes = 100000
render_every = 5000
how_much_to_render = 3
rewards = []


# ---------------- Main Training Loop ------------------
avg_rewards = 0
number_episodes = 0

for episode in range(start_episode, episodes):
    if episode % render_every < how_much_to_render and episode > 99:
        env = gym.make(env_name, render_mode="human", **add_info)
    else:
        env = gym.make(env_name, **add_info)
    # Stack frames and flatten the observation
    env = FrameStackObservation(env, stack_size=4, padding_type="zero")
    env = FlattenObservation(env)
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done and total_reward < 500:
        action = agent.play(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition and train on batches
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    # Decay epsilon after episode
    agent.decay_epsilon()
    
    avg_rewards += 1/ (number_episodes + 1) * (total_reward - avg_rewards)
    if number_episodes > 100:

        avg_rewards = 0
        number_episodes = 0
    number_episodes += 1

    print(f"Episode {episode} | Avg Reward: {avg_rewards:.2f} | Epsilon: {agent.epsilon:.3f} | Total Reward: {total_reward:.2f}")
    rewards.append(total_reward)

    # ---- Save best policy ----
    if save_best and avg_rewards > best_avg_reward:
        best_avg_reward = avg_rewards
        torch.save(agent.q_network.state_dict(), best_policy_path)
        print(f"🏆 New best policy saved! Avg Reward: {best_avg_reward:.2f} -> {best_policy_path}")

    # ---- Save policy periodically ----
    save_path_2 = save_path + f"avg{int(avg_rewards)}_ep{episode}.pth"
    if save_policy and episode % save_every == 0 and episode > 0:
        if hasattr(agent, "q_network"):
            torch.save(agent.q_network.state_dict(), save_path_2)
            print(f"✅ Policy (q_network) saved at episode {episode} -> {save_path_2}")
            
            # Save full checkpoint (for complete resume)
            checkpoint = {
                'episode': episode,
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'update_counter': agent.update_counter,
                'per_beta': agent.per_beta,
                'epsilon': agent.epsilon,
                'best_avg_reward': best_avg_reward,
                'rewards': rewards[-1000:] if len(rewards) > 1000 else rewards  # Save last 1000 rewards
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Full checkpoint saved -> {checkpoint_path}")
        elif hasattr(agent, "model"):
            torch.save(agent.model.state_dict(), save_path_2)
            print(f"✅ Policy (model) saved at episode {episode} -> {save_path_2}")
        else:
            print("⚠️ No neural network found in agent, skipping save...")

    env.close()

# Save final checkpoint
if save_policy:
    final_checkpoint = {
        'episode': episodes - 1,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'update_counter': agent.update_counter,
        'per_beta': agent.per_beta,
        'epsilon': agent.epsilon,
        'best_avg_reward': best_avg_reward,
        'rewards': rewards
    }
    torch.save(final_checkpoint, checkpoint_path)
    print(f"💾 Final checkpoint saved -> {checkpoint_path}")

# Plot average reward over last 100 episodes
avg_rewards_plot = [np.mean(rewards[max(0, i - 100): i + 1]) for i in range(len(rewards))]
plt.plot(avg_rewards_plot)
plt.xlabel("Episode")
plt.ylabel("Average Reward (100 ep)")
plt.title("DQN on " + env_name)
training_curve_path = f"policies/training_curve_{PARAM}{timestamp}.png"
plt.savefig(training_curve_path)
print(f"📊 Training curve saved to {training_curve_path}")
plt.show()

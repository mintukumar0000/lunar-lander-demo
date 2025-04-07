# 1. Import dependencies 
import gymnasium as gym
from gymnasium.utils.play import play
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import matplotlib.pyplot as plt

# 2. Human Play Section --------------------------------------------------------
# Initialize variables to track rewards
current_rewards = []
all_episodes = []

# Create environment for human play
env_play = gym.make("LunarLander-v3", render_mode="rgb_array")

def reward_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global current_rewards, all_episodes
    current_rewards.append(rew)
    
    if terminated or truncated:
        episode_total = sum(current_rewards)
        all_episodes.append(current_rewards.copy())
        current_rewards.clear()
        print(f"Episode ended with total reward: {episode_total}")

# Play with human control
play(env_play, keys_to_action={
    "w": 2,  # main engine
    "a": 3,  # left engine
    "d": 1,  # right engine
    "s": 0   # do nothing
}, callback=reward_callback)

# Plot human performance
if len(all_episodes) > 0:
    plt.figure(figsize=(10, 5))
    cumulative_rewards = [sum(episode) for episode in all_episodes]
    plt.plot(cumulative_rewards, 'b-o')
    plt.title("Human Play Performance")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
else:
    print("No human play episodes recorded.")

# 3. Training Section ----------------------------------------------------------
log_dir = "ppo_lunarlander_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# Configure training environment
env_train = Monitor(gym.make("LunarLander-v3"), log_dir)

# Define PPO model with optimized hyperparameters
model = PPO(
    "MlpPolicy",
    env_train,
    verbose=1,
    tensorboard_log=log_dir,
    gamma=0.99,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    learning_rate=2.5e-4,
    policy_kwargs={
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
    }
)

# Train the model
timesteps = 1_000_000
model.learn(
    total_timesteps=timesteps,
    tb_log_name="ppo_lunar_lander",
    progress_bar=True
)

# Save the trained model
model.save("ppo_lunarlander")
env_train.close()

# 4. Evaluation Section --------------------------------------------------------
# Load the trained model
del model
eval_env = Monitor(gym.make("LunarLander-v3", render_mode="human"), log_dir)
model = PPO.load("ppo_lunarlander", env=eval_env)

# Quantitative evaluation
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"\n{'='*40}")
print(f"Mean reward over 10 episodes: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"{'='*40}\n")

# Qualitative evaluation
obs, _ = eval_env.reset()
total_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    eval_env.render()
    
    if terminated or truncated:
        print(f"Episode ended with reward: {total_reward}")
        total_reward = 0
        obs, _ = eval_env.reset()

eval_env.close()
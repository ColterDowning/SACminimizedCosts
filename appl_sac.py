import os
import numpy as np
import pandas as pd
import torch
from Environment import AAPLSellEnvV2
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt


# Get the current script's directory
current_dir = os.getcwd()

# Construct the path to the CSV file
csv_path = os.path.join(current_dir, '..', 'data', 'merged_bid_ask_ohlcv_data.csv')

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_path)

#--------------Create Environment------------------------------------------------------------------

# Create the environment
env = AAPLSellEnvV2(data)

# Check the environment for functioning obs space, action space, reward structure, term conditions
# reset and step methods. Throws an error if any of these are inconsistent with the Gym interface
check_env(env)

# Create an evaluation environment and wrap it with Monitor
eval_env = AAPLSellEnvV2(data)
eval_env = Monitor(eval_env)

#--------------Create and Train Agent--------------------------------------------------------------

# Instantiate the agent
model = SAC('MlpPolicy', env, verbose=1, device='cuda')

# Define the evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

# Check if PyTorch is using the GPU
if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")

# Train the agent by learning from 100,000 actions
model.learn(total_timesteps=1000, callback=eval_callback)

# Save the trained model
model.save('sac_aapl_sell_model')


#--------------Evaluate Agent--------------------------------------------------------------

# Load the trained model from the saved zip file
loaded_model = SAC.load('sac_aapl_sell_model.zip')

# Evaluate the model
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10, deterministic=True)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

#import sys; sys.exit(0)

# Lists to store the data for visualization
steps = []
shares_remaining_list = []
time_remaining_list = []
close_price_list = []
volume_list = []
volatility_list = []
actions_list = []

# Reset the environment and extract the observation from the tuple
obs, _ = env.reset()

# Run the model for some episodes to evaluate its performance
for step in range(1000):
    # Ensure obs is a Numpy array and pass it to the model's predict function
    action, _states = loaded_model.predict(obs, deterministic=True)
    actions_list.append(action)  # Store action to visualize later
    obs, reward, terminated, truncated, info = env.step(action)

    # Store the data for visualization
    steps.append(step)
    shares_remaining_list.append(env.shares_remaining)
    time_remaining_list.append(1.0 - (env.current_step / env.max_steps))
    close_price_list.append(env.data.iloc[env.current_step]['close'])
    volume_list.append(env.data.iloc[env.current_step]['volume'])
    volatility_list.append(env.data.iloc[env.current_step]['volatility'])

    # Render the environment (optional)
    # env.render()

    if terminated or truncated:
        obs, _ = env.reset()

# Plotting the data using matplotlib
plt.figure(figsize=(16, 10))

# Plot Shares Remaining
plt.subplot(3, 2, 1)
plt.plot(steps, shares_remaining_list, label='Shares Remaining', color='b')
plt.xlabel('Steps')
plt.ylabel('Shares Remaining')
plt.title('Shares Remaining Over Time')
plt.legend()

# Plot Time Remaining
plt.subplot(3, 2, 2)
plt.plot(steps, time_remaining_list, label='Time Remaining', color='g')
plt.xlabel('Steps')
plt.ylabel('Time Remaining')
plt.title('Time Remaining Over Time')
plt.legend()

# Plot Close Price
plt.subplot(3, 2, 3)
plt.plot(steps, close_price_list, label='Close Price', color='r')
plt.xlabel('Steps')
plt.ylabel('Close Price')
plt.title('Close Price Over Time')
plt.legend()

# Plot Volume
plt.subplot(3, 2, 4)
plt.plot(steps, volume_list, label='Volume', color='orange')
plt.xlabel('Steps')
plt.ylabel('Volume')
plt.title('Volume Over Time')
plt.legend()

# Plot Volatility
plt.subplot(3, 2, 5)
plt.plot(steps, volatility_list, label='Volatility', color='purple')
plt.xlabel('Steps')
plt.ylabel('Volatility')
plt.title('Volatility Over Time')
plt.legend()

# Plot actions over time
plt.figure(figsize=(10, 5))
plt.plot(steps, actions_list, label='Actions', color='c')
plt.xlabel('Steps')
plt.ylabel('Action Value')
plt.title('Actions Over Time')
plt.legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.show()

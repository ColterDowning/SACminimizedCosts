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

normalized_reward = 0.0


#--------------Create and Train Agent--------------------------------------------------------------

#Instantiate the agent
model = SAC('MlpPolicy', env, verbose=1, device='cuda')

# Check if PyTorch is using the GPU
if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")

# Train the agent by learning from 10,000 actions
model.learn(total_timesteps=100000)

# Save the trained model
model.save('sac_aapl_sell_model')


#--------------Evaluate Agent--------------------------------------------------------------

# Load the trained model from the saved zip file
loaded_model = SAC.load('sac_aapl_sell_model.zip')

# Evaluate the model
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=100, deterministic=True)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Lists to store the data for visualization
steps = []
shares_remaining_list = []
time_remaining_list = []
close_price_list = []
volume_list = []
volatility_list = []
actions_list = []
portfolio_list = []
transaction_costs_list = []

# Reset the environment and extract the observation from the tuple
obs, _ = env.reset()

# Run the model for some episodes to evaluate its performance
for step in range(10000):
    # Ensure obs is a Numpy array and pass it to the model's predict function
    action, _states = loaded_model.predict(obs, deterministic=True)
    actions_list.append(action)  # Store action to visualize later
    obs, reward, terminated, truncated, info = env.step(action)

    # Store the data for visualization
    steps.append(step)
    shares_remaining_list.append(env.shares_remaining)
    time_remaining_list.append(1.0 - (env.step_addr / env.max_steps))
    close_price_list.append(env.data.iloc[env.current_step]['close']) 
    volume_list.append(env.data.iloc[env.current_step]['volume']) 
    volatility_list.append(env.data.iloc[env.current_step]['volatility'])
    portfolio_list.append(env.portfolio_value)
    transaction_costs_list.append(env.transaction_cost)

    # Render the environment (optional)
    # env.render()

    if terminated or truncated:
        obs, _ = env.reset()

vwap_trades = env.benchmark.get_vwap_trades(data, 1000)

vwap_metrics = env.simulate_vwap_strategy(vwap_trades)

max_steps = min(len(steps), len(vwap_metrics['steps']))



plt.figure(figsize=(15, 15))

# Transaction Costs
plt.subplot(4, 1, 1)
plt.plot(steps[:max_steps], transaction_costs_list[:max_steps], label='SAC Agent')
plt.plot(vwap_metrics['steps'], vwap_metrics['transaction_costs'], label='VWAP Strategy')
plt.xlabel('Step')
plt.ylabel('Transaction Cost')
plt.title('Transaction Cost Comparison')
plt.legend()

# Shares Remaining
plt.subplot(4, 1, 2)
plt.plot(steps[:max_steps], shares_remaining_list[:max_steps], label='SAC Agent')
plt.plot(vwap_metrics['steps'], vwap_metrics['shares_remaining'], label='VWAP Strategy')
plt.xlabel('Step')
plt.ylabel('Shares Remaining')
plt.title('Shares Remaining Comparison')
plt.legend()

# Portfolio Value
plt.subplot(4, 1, 3)
plt.plot(steps[:max_steps], portfolio_list[:max_steps], label='SAC Agent')
plt.plot(vwap_metrics['steps'], vwap_metrics['portfolio_value'], label='VWAP Strategy')
plt.xlabel('Step')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()

# Total Shares Sold
plt.subplot(4, 1, 4)
plt.plot(steps[:max_steps], [env.total_shares - sr for sr in shares_remaining_list[:max_steps]], label='SAC Agent')
plt.plot(vwap_metrics['steps'], [env.total_shares - sr for sr in vwap_metrics['shares_remaining']], label='VWAP Strategy')
plt.xlabel('Step')
plt.ylabel('Total Shares Sold')
plt.title('Total Shares Sold Comparison')
plt.legend()

plt.tight_layout()
plt.show()
import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
from benchmark_costs_script import Benchmark
import warnings

# Suppress only FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


class AAPLSellEnvV2(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(AAPLSellEnvV2, self).__init__()

        # Load data
        self.data = data.reset_index(drop=True)
        self.max_steps = 390 # 390 minutes in a trading day. Assumes our data timesteps are 1 min
        # DataFrame to track trades
        self.trade_log = pd.DataFrame(columns=['shares', 'action'])

        # Initialize the Benchmark instance for cost calculations
        self.benchmark = Benchmark(data)

        # Find max values for normalization
        self.max_bid_price = self.data[[f'bid_price_{i}' for i in range(1, 6)]].max().max()
        self.max_bid_size = self.data[[f'bid_size_{i}' for i in range(1, 6)]].max().max()
        self.max_ask_price = self.data[[f'ask_price_{i}' for i in range(1, 6)]].max().max()
        self.max_ask_size = self.data[[f'ask_size_{i}' for i in range(1, 6)]].max().max()
        self.max_close_price = self.data['close'].max() if 'close' in self.data else 1.0
        self.max_volume = self.data['volume'].max() if 'volume' in self.data else 1.0

        # Action space: Number of shares to sell (0 to shares_remaining)
        # Initially set to the maximum possible range
        self.action_space = spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32)

        # Observation space
        # Here we are considering 25 dimensions including bid and ask size(10), bid and ask price (10)
        # time_remaining, shares_remaining, close price, volume, and volatility
        obs_low = np.array([0.0] * 25, dtype=np.float32)
        obs_high = np.array([np.inf] * 25, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize state variables
        self.reset()

        # Risk aversion parameter for reward function
        # Reference 'Machine Learning for Trading, Gordon Ritter'
        self.kappa = 0.001

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.shares_remaining = 1000  # Starting with 1000 shares to sell
        self.total_shares = 1000
        self.done = False
        self.trade_log = pd.DataFrame(columns=['shares', 'action'])
        initial_observation = self._next_observation()
        return initial_observation, {}

    def _next_observation(self):
        # Loop until we find a row with complete data
        while True:
            frame = self.data.iloc[self.current_step]

            # Check if there are any missing values in the row
            if frame.isna().any():
                # Move to the next step if there are missing values
                self.current_step += 1
                # If we reach the end, terminate the episode
                if self.current_step >= self.max_steps:
                    raise ValueError("No complete data available in the dataset for the remaining timesteps.")
            else:
                # If no missing values, break the loop and proceed
                break

        # Normalize time and shares remaining
        time_remaining = np.float32(1.0 - (self.current_step / self.max_steps))
        shares_remaining_normalized = np.float32(self.shares_remaining / self.total_shares)

        # Extract and normalize bid and ask prices and sizes (levels 1-5)
        bid_prices = [np.float32(frame[f'bid_price_{i}']) / self.max_bid_price if self.max_bid_price > 0 else 0 for i in range(1, 6)]
        bid_sizes = [np.float32(frame[f'bid_size_{i}']) / self.max_bid_size if self.max_bid_size > 0 else 0 for i in range(1, 6)]
        ask_prices = [np.float32(frame[f'ask_price_{i}']) / self.max_ask_price if self.max_ask_price > 0 else 0 for i in range(1, 6)]
        ask_sizes = [np.float32(frame[f'ask_size_{i}']) / self.max_ask_size if self.max_ask_size > 0 else 0 for i in range(1, 6)]

        # Extract OHLCV data and volatility, and normalize if applicable
        close_price = np.float32(frame['close']) / self.max_close_price if 'close' in frame and self.max_close_price > 0 else np.float32(0.0)
        volume = np.float32(frame['volume']) / self.max_volume if 'volume' in frame and self.max_volume > 0 else np.float32(0.0)
        volatility = np.float32(frame['volatility']) if 'volatility' in frame else np.float32(0.0)

        # Concatenate all features into a single observation array
        obs = np.array([
            time_remaining,
            shares_remaining_normalized,
            *bid_prices,
            *bid_sizes,
            *ask_prices,
            *ask_sizes,
            close_price,
            volume,
            volatility
        ], dtype=np.float32)

        return obs


    def step(self, action):
        # Ensure action is a scalar between 0 and shares_remaining
        action = np.clip(action[0], 0, self.shares_remaining)

        # Check for NaN action values
        if np.isnan(action):
            raise ValueError("Action contains NaN values.")

        # Execute trade
        shares_to_sell = action
        execution_price, transaction_cost = self._execute_trade(shares_to_sell)

        # Update state
        if shares_to_sell > 0:
            self.shares_remaining -= shares_to_sell
            self.shares_remaining = max(self.shares_remaining, 0)  # Redundant, but ensure it never goes below zero
            action_taken = True
        else:
            action_taken = False

        # Log the trade
        self.trade_log = pd.concat([self.trade_log, pd.DataFrame({'shares': [shares_to_sell], 'action': [action_taken]})], ignore_index=True)

        # --------------Reward Calculation--------------------------------------------------------------
        # Enforce that all shares must be sold by the end of the day
        if self.current_step >= self.max_steps and self.shares_remaining > 0:
            # Large negative reward if shares are not sold by the end of the day
            reward = -1000.0
            terminated = True
        else:
            selling_price = self.data.iloc[self.current_step]['close']
            
            #delta_v_t = 
            profit_loss = -transaction_cost  # Negative of transaction cost as a base
            # Reward function as per Equation 15
            reward = profit_loss - (self.kappa / 2) * (profit_loss ** 2)
        # ----------------------------------------------------------------------------------------------

        # Check if done
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps or self.shares_remaining <= 0)
        truncated = False  # No time limit truncation for this environment

        # Get next observation
        obs = self._next_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        # Check for NaN values in observation and reward
        if np.isnan(obs).any():
            raise ValueError("Observation contains NaN values.")
        if np.isnan(reward):
            raise ValueError("Reward contains NaN value.")

        return obs, reward, terminated, truncated, {}


    def _execute_trade(self, shares_to_sell):
        # Define the market impact scaling factor
        alpha = 4.439584265535017e-06 

        # Use the benchmark method to compute the slippage and market impact
        components = self.benchmark.compute_components(alpha, shares_to_sell, self.current_step)
        slippage = float(components[0]) 
        market_impact = float(components[1])
        # Calculate the effective execution price using VWAP
        effective_price = self.benchmark.calculate_vwap(self.current_step, shares_to_sell)

        # Total transaction cost is slippage plus market impact
        transaction_cost = slippage + market_impact

        return effective_price, transaction_cost



    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Shares Remaining: {self.shares_remaining}')
        print(f'Time Remaining: {1.0 - (self.current_step / self.max_steps):.2f}')
        print(f'Close Price: {self.data.iloc[self.current_step]["close"]}')
        print(f'Volume: {self.data.iloc[self.current_step]["volume"]}')
        print(f'Volatility: {self.data.iloc[self.current_step]["volatility"]}')

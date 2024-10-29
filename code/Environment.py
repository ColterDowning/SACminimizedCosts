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
        self.current_step = 0

        # Initialize the Benchmark instance for cost calculations
        self.benchmark = Benchmark(data)

        # Find max values for normalization
        self.max_bid_price = self.data[[f'bid_price_{i}' for i in range(1, 6)]].max().max()
        self.max_bid_size = self.data[[f'bid_size_{i}' for i in range(1, 6)]].max().max()
        self.max_ask_price = self.data[[f'ask_price_{i}' for i in range(1, 6)]].max().max()
        self.max_ask_size = self.data[[f'ask_size_{i}' for i in range(1, 6)]].max().max()
        self.max_close_price = self.data['close'].max() if 'close' in self.data else 1.0
        self.max_volume = self.data['volume'].max() if 'volume' in self.data else 1.0
        self.max_volatility = self.data['volatility'].max() if 'volatility' in self.data else 1.0

        # Action space: Number of shares to sell (0 to shares_remaining)
        # Initially set to the maximum possible range
        self.action_space = spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32)

        # Observation space
        # Here we are considering 6 dimensions 
        # time_remaining, shares_remaining, close price, volume, volatility and
        # portfolio value
        obs_low = np.array([0.0] * 6, dtype=np.float32)
        obs_high = np.array([np.inf] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize state variables
        self.reset()

        # Risk aversion parameter for reward function
        # Reference 'Machine Learning for Trading, Gordon Ritter'
        self.kappa = 10e-4

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Do not reset current_step unless we've reached the end of the dataset
        if self.current_step >= len(self.data) - 1:
            self.current_step = 0
        self.step_addr = 0 # track how many steps we have taken in an episode
        self.shares_remaining = 1000  # Starting with 1000 shares to sell
        self.total_shares = 1000 # Used for normalization
        self.portfolio_value = 1000 * self.data['close'].iloc[0] if self.data['close'].iloc[0] > 0 else 1000 * self.data['ask_price_1'].iloc[0]
        self.cash = 0.0  # No cash initially, all value in holdings
        self.holdings = self.total_shares * self.data['close'].iloc[0] if self.data['close'].iloc[0] > 0 else 1000 * self.data['ask_price_1'].iloc[0]
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
                if self.current_step >= len(self.data):
                    self.done = True
                    return np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                # If no missing values, break the loop and proceed
                break

        # Normalize time and shares remaining
        time_remaining = np.float32(1.0 - (self.step_addr / self.max_steps))
        shares_remaining_normalized = np.float32(self.shares_remaining / self.total_shares)
        portfolio_value_normalized = np.float32(self.portfolio_value / (self.total_shares * self.max_close_price))
        # cash_normalized = np.float32(self.cash / (self.total_shares * self.max_close_price))
        # holdings_normalized = np.float32(self.holdings / (self.total_shares * self.max_close_price))

        # Extract and normalize bid and ask prices and sizes (levels 1-5) DECIDED NOT TO USE IN OBS SPACE
        # bid_prices = [np.float32(frame[f'bid_price_{i}']) / self.max_bid_price if self.max_bid_price > 0 else 0 for i in range(1, 6)]
        # bid_sizes = [np.float32(frame[f'bid_size_{i}']) / self.max_bid_size if self.max_bid_size > 0 else 0 for i in range(1, 6)]
        # ask_prices = [np.float32(frame[f'ask_price_{i}']) / self.max_ask_price if self.max_ask_price > 0 else 0 for i in range(1, 6)]
        # ask_sizes = [np.float32(frame[f'ask_size_{i}']) / self.max_ask_size if self.max_ask_size > 0 else 0 for i in range(1, 6)]

        # Extract OHLCV data and volatility, and normalize if applicable
        close_price = np.float32(frame['close']) / self.max_close_price 
        volume = np.float32(frame['volume']) / self.max_volume 
        volatility = np.float32(frame['volatility']) / self.max_volatility 

        # Concatenate all features into a single observation array
        obs = np.array([
            time_remaining,
            shares_remaining_normalized,
            portfolio_value_normalized,
            #cash_normalized,
            #holdings_normalized,
            #*bid_prices,
            #*bid_sizes,
            #*ask_prices,
            #*ask_sizes,
            close_price,
            volume,
            volatility
        ], dtype=np.float32)

        return obs


    def step(self, action):
        # Ensure action is a scalar between 0 and shares_remaining
        action = np.round(np.clip(action[0], 0, self.shares_remaining))

        # Check for NaN action values
        if np.isnan(action):
            action = np.float32(0.0)
        
        # Information about current state before we take the action.
        # We need this for the reward
        shares_remaining_tminus1 = self.shares_remaining
        price_tminus1 = self.data.iloc[self.current_step]['close']

        # Execute trade
        shares_to_sell = action
        execution_price, transaction_cost = self._execute_trade(shares_to_sell)

        # Update state
        if shares_to_sell > 0:
            self.shares_remaining -= shares_to_sell
            self.shares_remaining = max(self.shares_remaining, 0)  # Redundant, but ensure it never goes below zero
            self.cash += np.round(shares_to_sell * execution_price - transaction_cost)
            self.holdings = np.round(self.shares_remaining * execution_price)
            action_taken = True
        else:
            action_taken = False

        # Update portfolio value (holdings + cash)
        self.portfolio_value = self.holdings + self.cash

        # Log the trade
        self.trade_log = pd.concat([self.trade_log, pd.DataFrame({'shares': [shares_to_sell], 'action': [action_taken]})], ignore_index=True)

        # Check if done
        self.current_step += 1
        self.step_addr += 1
        terminated = bool(self.step_addr >= self.max_steps or self.shares_remaining <= 0 or self.current_step >= len(self.data))
        truncated = False  # No time limit truncation for this environment

        # --------------Reward Calculation--------------------------------------------------------------
        # Enforce that all shares must be sold by the end of the day
        if self.step_addr >= self.max_steps and self.shares_remaining > 0:
            # Large negative reward if shares are not sold by the end of the day
            reward = -1000.0
            terminated = True
        else:
            delta_v_t = shares_remaining_tminus1 * (self.data.iloc[self.current_step]['close'] - price_tminus1) - transaction_cost
            #holding_incentive = 100 * self.shares_remaining  # Small positive reward for holding shares
            # Reward function as per Equation 15
            reward = delta_v_t - (self.kappa / 2) * (delta_v_t ** 2) #+ holding_incentive
        # ----------------------------------------------------------------------------------------------

        # Get next observation
        obs = self._next_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        # Check for NaN values in observation and reward
        if np.isnan(obs).any():
            raise ValueError("Observation contains NaN values.")
        if np.isnan(reward):
            reward = np.float32(0.0)

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
        print(f'Time Remaining: {1.0 - (self.step_addr / self.max_steps):.2f}')
        print(f'Close Price: {self.data.iloc[self.current_step]["close"]}')
        print(f'Volume: {self.data.iloc[self.current_step]["volume"]}')
        print(f'Volatility: {self.data.iloc[self.current_step]["volatility"]}')

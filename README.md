This projects aims to solve the following problem: "I want to sell 1000 shares of AAPL in one day, how should I split the trades to optimize transaction costs?" 

To understand the complexity of this problem, we first need to understand what transaction costs are. In a financial market, we will consider two types of transaction costs: slippage and market impact. 
Slippage occurs when the actual execution price of a trade differs from the expected price, usually due to rapid price changes or limited liquidity. In this project, we assume we are only placing market
orders, which execute at the most favorable bid price. The slippage would then be the difference between the prevailing bid price and our expected cost, calculated as the volume weighted average price.
Market impact refers to the price movement as a consequence of our own trading. Historically, this is the most significant source of transaction costs, according to [Baldacci et al.](https://arxiv.org/abs/2110.03810)
The market impact is proportional to the square root of the number of shares sold, scaled by a linear coefficient λ, normally referred to as Kyle's lambda.

For active traders, these transaction costs are a constant friction that reduces profits, and is obviously something we want to minimize. How might we solve this problem? At it's core, this is an
optimization problem, and quite a complex one. The stock market is a dynamic, complex, highly-dimensional environment. Due to this, we need to rely on computers. In the world of Machine Learning, a sub-branch called 
reinforcement learning is concerned with how an intelligent agent should take actions in a dynamic environment in order to maximize a reward signal. If we can be clever with how we craft our reward signal, we would 
be training an agent to minimize our transaction costs!

There exist many types of agents/algorithms in reinforcement learning, so which one should we use? In 2018, Haarnoja et al. released a paper called [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), which was motivated to solve two major problems:
1. > model-free deep RL methods are notoriously expensive in terms of their sample complexity. Even relatively simple tasks can require millions of steps of data collection, and complex behaviors with high-dimensional
observations might need substantially more.
2. > these methods are often brittle with respect to their hyperparameters: learning rates, exploration constants, and other settings must be set carefully for different problem settings to achieve good results.

To solve these problems, the Soft Actor Critic algorithm introduces several key innovations:
1. The Soft Actor Critic is an off-policy algorithm that leverages a replay buffer to store and reuse past experiences. This means you don't need to rely solely on your most recent experiences.
2. The objective function of the SAC includes an entropy maximization term, which encourages robustness and exploration.
3. Practical neural network applications for the Actor / Critic architecture that allows for high-dimensional environments and stable learning.

The Soft Actor Critic algorithm is quite beautiful in the sense that is draws lots of parallels to thermodynamics. Let's take a look.

![SAC_objective](https://github.com/user-attachments/assets/9ad9f5c9-cb6d-4dc3-9be6-ff4192e60ae7)

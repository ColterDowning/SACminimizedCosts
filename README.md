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

To solve these problems, the Soft Actor Critic (SAC) algorithm introduces several key innovations:
1. The Soft Actor Critic is an off-policy algorithm that leverages a replay buffer to store and reuse past experiences. This means you don't need to rely solely on your most recent experiences.
2. The objective function of the SAC includes an entropy maximization term, which encourages robustness and exploration.
3. Practical neural network applications for the Actor / Critic architecture that allows for high-dimensional environments and stable learning.

The Soft Actor Critic algorithm is quite beautiful in the sense that is draws lots of parallels to thermodynamics. Let's take a look.

![SAC_objective](https://github.com/user-attachments/assets/9ad9f5c9-cb6d-4dc3-9be6-ff4192e60ae7)

Standard RL seeks to maximize the expected sum of rewards [r(s_t,a_t)]. SAC maximizes this aswell, as seen in the objective function J(π), but it also contains a policy entropy term denoted as H(π(.|s_t)).

Our first parallel comes when we consider that thermodynamical systems tend to minimize their Helmholtz free energy F, where E is the internal energy, T is the temperature, and S is the entropy.
![helm_free_energy](https://github.com/user-attachments/assets/a4b0d689-5753-468f-a114-55bbceb9affb)

Maximizing the SAC objective is analogous to minimizing the free energy of a system where the negative reward r is analogous to the internal energy E, policy entropy H(π(.|s_t)) is analogous to entropy S,
and the temperature parameter α is analogous to the temperature of the system T. It's not a coincidence α is called the temperature parameter! The agent's policy can be viewed as a probability distribution that evolves
through training to minimize free energy, similar to how a particle system reaches equilibrium in statistical mechanics. 

A further insight comes when we consider a definition of entropy. In thermodynamics, the Gibbs entropy is defined as  
![thermo_entropy](https://github.com/user-attachments/assets/11477712-867a-4495-8751-18cd6b765f53)
where k_B is the Boltzmann constant and p_i is the probability of the system being in the i-th microstate. Think of a microstate as a way that a system could be configured. If we had a system of 2 coins, 1 microstate
would be the case where both coins are heads, and another microstate would be where 1 is head and 1 is tails. p_i is the probabilty of that state occuring, so 2 heads is 25% but 1 heads 1 tail is 50% (tail/head and head/tail). Maximizing the entropy means maximizing the available microstates.


The SAC objective seeks to maximize the policy entropy, where we can define the differential entropy (not accounting for action transformations) to be:
![standard_entropy](https://github.com/user-attachments/assets/3511a751-ba36-4634-909d-023ec4b24967)
Notice the similarities?! The objective maximizes the policy entropy, which means it is maximizing the available actions the agent can take (captured in a normal distribution). Why is this important? An agent that selects an action from a random distribution is occasionally going to select a sub-optimal action as determined by the critic, simply because there are more options to choose from. This stochasticity will lead the agent to 'explore' new areas of the environment. This exploration, as shown by [experiment](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf), allows the agent to find global minima instead of exploiting the highest expected sum of rewards in a local minima! The exploration builds a robustness, which is ideal for complex environments like the stock market.

On last note is the parallel to temperature. Temperature is the average kinetic energy of the particles in a system, and dictates the distribution of particles across the available energy states. A higher temperature leads to a wider distribution of occupied energy states, while a lower temperature leads to tighter distribution over a smaller amount of energy states. In our SAC objective, the α term scales the entropy. A higher α leads to a higher bias towards H(π(.|s_t)) in the objective, which will broaden the available action distribution for the agent!! The same is true for lower α all the way to 0 where we reduce our distribution to the single value of the calculated highest expected reward (absolute zero)!

For our purposes, the SAC algorithm will be ideal since it performs well in complex environments, and has the ability to store historical data (off-policy). Now, how do we craft our reward function to minimize transaction costs? For this, I turned to Gordon Ritter and his paper [Machine Learning for Trading](https://cims.nyu.edu/~ritter/ritter2017machine.pdf). In his framework, the rational investor with a finite investment horizon chooses actions to maximize the expected utility of terminal wealth. While this isn't the exact problem we are setting out to solve, the insights are still useful since minimized transaction costs are required to maximize expected wealth. The principle reward function used is found in equation (15) as 
![reward](https://github.com/user-attachments/assets/b88a0961-f943-4485-b9a1-6b377572b1cd)

where deltav_t is the change in 'portfolio value' from one time increment. 





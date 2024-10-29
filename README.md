This projects aims to solve the following problem: "I want to sell 1000 shares of AAPL in one day, how should I split the trades to optimize transaction costs?" 

To understand the complexity of this problem, we first need to understand what transaction costs are. In a financial market, we will consider two types of transaction costs: slippage and market impact. 
Slippage occurs when the actual execution price of a trade differs from the expected price, usually due to rapid price changes or limited liquidity. In this project, we assume we are only placing market
orders, which execute at the most favorable bid price. The slippage would then be the difference between the prevailing bid price and our expected cost, calculated as the volume weighted average price.
Market impact refers to the price movement as a consequence of our own trading. Historically, this is the most significant source of transaction costs, according to Baldacci et al. (https://arxiv.org/abs/2110.03810)
The market impact is proportional to the square root of the number of shares sold, scaled by a linear coefficient \lambda 

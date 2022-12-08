import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

theta = 1.1 # Elasticity of substitution
K = 1 # Constant
M = 10000 # Total money available in market
sigma = 0.1 # Price fluctuation parameter in absence of social planner
N_DISCRETE_ACTIONS = [3, 4]

class firmEnv(gym.Env):
    def __init__(self, tax, sigma, eq_demand, cost, price, quantity, demand):
        # Price actions we can take: {lower price, keep price, increase price}
        # Quantity actions we can take: {produce 0%, 33%, 66% or 100% of aggregate demand}

        self.action_space = gym.spaces.MultiDiscrete(N_DISCRETE_ACTIONS)

        #low = np.array(
        #    [-0.0,
        #     -0.0,
        #     -0.0,
        #     -0.0,
        #     -0.0,
        #     -0.0,
        #     -0.0]
        #).astype(np.float32)

        #high = np.array(
        #    [1.0,
        #     1.0,
        #     10000.0,
        #     100.0,
        #     1.0,
        #     100.0,
        #     1.0]
        #).astype(np.float32)

        #self.observation_space = gym.spaces.Box(low=low, high=high)

        self.tax = tax
        self.sigma = sigma
        self.eq_demand = eq_demand
        self.demand = demand
        self.cost = cost
        self.price = price
        self.quantity = quantity

    # Create helper functions:
    # 1. equilibrium_price: computes market equilibrium price given individual firm prices, n_firms and elasticity of sub
    # 2. equilibrium_demand: computes market equilibrium demand
    # 3. firm_demand: computes demand for individual firm's product. The firm does not observe this, but it is used to
    # estimate firm sales.
    # 4. normalize_reward: normalizes reward, applied in step function

    def firm_demand(self, eq_demand, eq_price, K=K, theta=theta):
        self.demand = K * eq_demand * (self.price / eq_price) ** (-theta)

    def normalize_reward(self, reward):
        reward = 1 - 2 ** (-reward / 1e7)
        return reward

    def compute_some_stuff(self, actions):
        price_action, quantity_action = actions
        if price_action == 0:
            price_action = -1.0
        elif price_action == 1:
            price_action = 0.0
        else:
            price_action = 1.0

        if quantity_action == 0:
            quantity_action = 0.0
        elif quantity_action == 1:
            quantity_action = 0.333
        elif quantity_action == 2:
            quantity_action = 0.666
        else:
            quantity_action = 1.0

        self.price = (price_action * sigma + 1) * self.price

        return self.price

    def step(self, actions, eq_demand, eq_price, K=K, theta=theta, sigma=sigma):
        price_action, quantity_action = actions
        if price_action == 0:
            price_action = -1.0
        elif price_action == 1:
            price_action = 0.0
        else:
            price_action = 1.0

        if quantity_action == 0:
            quantity_action = 0.0
        elif quantity_action == 1:
            quantity_action = 0.333
        elif quantity_action == 2:
            quantity_action = 0.666
        else:
            quantity_action = 1.0

        self.price = (price_action * sigma + 1) * self.price
        self.quantity = (quantity_action * eq_demand)

        self.demand = self.firm_demand(eq_demand, eq_price, K=K, theta=theta)

        penalty = 0

        if (self.quantity > self.demand):
            penalty = self.cost * (self.quantity - self.demand)

        reward = self.normalize_reward((1 - self.tax) * (self.price - self.cost) * self.quantity + penalty)

        return self.get_observations(), reward, False

    def get_observations(self):
        tax = self.tax
        sigma = self.sigma
        price = self.price
        eq_demand = self.eq_demand
        cost = self.cost

        return eq_demand, tax, sigma, price, cost

    def reset(self):
        self.cost += 0.02 * self.cost

        return self.get_observations()

#class socialPlannerEnv:
#    def __init__(self):

class Environment:
    def __init__(self, n_firms, eq_price, eq_demand):
        #self.social = SocialPlanner([0.21,0.21],[0.1,0.1],1000,0.3,0.3)
        self.firms = [firmEnv(0.21, 0.1, 1000, 5, 5, 500, 500), firmEnv(0.21, 0.1, 1000, 5, 5, 500, 500)]
        self.n_firms = n_firms
        self.eq_price = eq_price
        self.eq_demand = eq_demand

    def equilibrium_price(self, n_firms, prices, theta=theta):
        self.eq_price = 0
        for i in range(n_firms-1):
            self.eq_price += (1/n_firms) * (prices[i]**(1-theta))**(1/(1-theta))

    def equilibrium_demand(self, eq_price, K=K, M=M):
        self.eq_demand = K * (M/self.eq_price)


    def equilibrium_price_demand(self, n_firms, price, K=K, M=M, theta=theta):
        eq_price = Environment.equilibrium_price(self.n_firms, self.price, theta)
        eq_demand = Environment.equilibrium_demand(self.self, eq_price, K=K, M=M)
        return eq_price, eq_demand
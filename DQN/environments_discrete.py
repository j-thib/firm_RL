import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

theta = 10 # Elasticity of substitution
K = 1 # Constant
M = 10000000 # Total money available in market
sigma = 0.1 # Price fluctuation parameter in absence of social planner
gamma = 1
eta = 1
all_actions = [(0,0),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2)]
sp_actions = [(0.1,0.1),(0.1,0.25),(0.1,0.4),(0.25,0.1),(0.25,0.25),(0.25,0.4),(0.4,0.1),(0.4,0.25),(0.4,0.4)]

class firmEnv(gym.Env):
    def __init__(self, tax, sigma, eq_demand, cost, price, quantity, demand):
        # Price actions we can take: {lower price, keep price, increase price}
        # Quantity actions we can take: {produce 0%, 33%, 66% or 100% of aggregate demand}

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
        reward = 1 - 2 ** (-reward/1e7)
        return reward

    def compute_price_from_action(self, actions, sigma=sigma):
        quantity_action, price_action = all_actions[actions]
        if price_action == 0:
            price_action = -1.0
        elif price_action == 1:
            price_action = 0.0
        else:
            price_action = 1.0

        self.price = (price_action * sigma + 1) * self.price
        #return self.price

    def compute_quantity_from_action(self, actions, sigma=sigma):
        quantity_action, price_action = all_actions[actions]
        if quantity_action == 0:
            quantity_action = 0.0
        elif quantity_action == 1:
            quantity_action = 0.333
        elif quantity_action == 2:
            quantity_action = 0.666
        else:
            quantity_action = 1.0

        self.quantity = (quantity_action * self.eq_demand)


    def step(self, actions, eq_demand, eq_price, K=K, theta=theta, sigma=sigma):
        quantity_action, price_action = all_actions[actions]
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
        self.price = np.clip(self.price, 2.0, 8.0)
        self.quantity = (quantity_action * eq_demand)

        self.firm_demand(eq_demand, eq_price, K=K, theta=theta)

        penalty = 0

        if (self.quantity > self.demand):
            penalty = self.cost * (self.quantity - self.demand)

        #reward = self.normalize_reward((1 - self.tax) * (self.price - self.cost) * self.quantity + penalty)
        reward = (1 - self.tax) * (self.price - self.cost) * self.quantity #+ penalty

        return self.get_observations(), reward, False

    def get_observations(self):
        tax = self.tax
        sigma = self.sigma
        price = self.price / 100
        eq_demand = self.eq_demand / 10000
        cost = self.cost / 10

        return eq_demand, tax, sigma, price, cost

    def reset(self):
        #self.cost += 0.02 * self.cost

        return self.get_observations()

"""class socialPlannerEnv:
    def __init__(self, tax, price_reg=0.1, gamma=1, eta=1):
        # Tax actions we can take: {10%, 25%, 40%}
        # Allow firms to fluctuate prices by: {5%, 10%}

        self.tax = tax
        self.price_reg = price_reg
        self.demand = demand
        self.gamma = gamma
        self.eta = eta

    def step(self, actions, firm_rewards, quantities):
        tax_action1, tax_action2 = sp_actions[actions]

        #fairness = np.sqrt()
        reward = sum(firm_rewards) - self.gamma*abs(self.demand - sum(quantities)) - self.

        return reward, tax_action, price_reg_action

    def get_observations(self):
        demand = self.demand / 10000
        #profit =
        return demand, profit # should return normalized observations"""

class Environment:
    def __init__(self, n_firms):
        #self.social = SocialPlanner([0.21,0.21],[0.1,0.1],1000,0.3,0.3)
        self.firms = [firmEnv(0.21, 0.1, 1000, 1, 5, 500, 500), firmEnv(0.21, 0.1, 1000, 1, 5, 500, 500)]
        self.n_firms = n_firms
        #self.eq_price = eq_price
        #self.eq_demand = eq_demand

    def equilibrium_price(self, n_firms, prices, theta=theta):
        self.eq_price = 0
        for i in range(n_firms):
            self.eq_price += (1/n_firms) * (prices[i]**(1-theta))**(1/(1-theta))

    def equilibrium_demand(self, eq_price, K=K, M=M):
        self.eq_demand = K * (M/self.eq_price)


    def equilibrium_price_demand(self, n_firms, price, K=K, M=M, theta=theta):
        eq_price = Environment.equilibrium_price(self.n_firms, self.price, theta)
        eq_demand = Environment.equilibrium_demand(self.self, eq_price, K=K, M=M)
        return eq_price, eq_demand
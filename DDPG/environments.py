"""import random
from random import gauss
import statistics
import numpy as np

theta = 1.1 # Elasticity of substitution
K = 1 # Constant
M = 10000 # Total money available in market

# Firms compete and produce imperfect substitutes. Increasing their price may lead to

class Firm:
    def __init__(self, tax, sigma, demand, cost, q_other, price, quantity):
        self.tax = tax
        self.sigma = sigma
        self.demand = demand
        self.cost = cost
        #self.p_other = p_other
        self.q_other = q_other
        self.price = price
        self.quantity = quantity

    def normalize_reward(self, reward):
        reward = 1 - 2**(-reward/1e7)
        return reward

    def firm_demand(self, agg_demand, agg_price, K=K, theta=theta):
        self.demand = K * agg_demand * (self.price / agg_price) ** (-theta)
        return self.demand

    def step(self, actions, q_other, sigma=0.1): # update to take in the

        price_action, quantity_action = actions
        price_action = np.clip(price_action, -1, 1)
        self.price = (price_action * sigma + 1) * self.price
        self.price = np.clip(self.price, 50, 500)
        quantity_action = np.clip(quantity_action, 0, 1)
        self.quantity = quantity_action * self.demand
        self.q_other = q_other
        penalty = 0

        if (self.quantity + self.q_other > self.demand):
            penalty = self.cost * ((self.quantity + self.q_other - self.demand) / 2)  # (self.q_other + 1))

        reward = self.normalize_reward((1-self.tax) * (self.price - self.cost) * self.quantity + penalty)

        return self.get_observations(), reward, False


    def get_observations(self, agg_demand=self.demand):

        #demand = self.demand / 10000 # Change

        tax = self.tax
        sigma = self.sigma
        price = self.price / 500
        demand = agg_demand
        cost = self.cost / 10
        q_other = self.q_other / 1000

        return demand, tax, sigma, price, cost, q_other

    def reset(self, q_other):
        #self.cost = gauss(mu=self.cost, sigma=1)
        self.cost = gauss(mu=2, sigma=1)
        self.q_other = q_other

        return self.get_observations()

class SocialPlanner:
    def __init__(self, tax, price_threshold, gamma=1, eta=1):
        self.tax = tax
        self.price_threshold = price_threshold
        self.demand = demand
        self.gamma = gamma
        self.eta = eta

    def step(self, actions, firm_rewards, quantities):
        tax_action, price_threshold_action = actions
        fairness = 0
        r_fair = [0, 0]
        for i in range(len(firm_rewards)):
            fairness += abs(firm_rewards[i] - r_fair[i])
        reward = sum(firm_rewards) - self.gamma*abs(self.demand - sum(quantities)) - self.eta*fairness
        return reward, tax_action, price_threshold_action

    def get_observations(self):
        demand = self.demand / 10000
        #profit =
        return demand, profit # should return normalized observations


class Environment:
    def __init__(self, n_firms, price):
        #self.social = SocialPlanner([0.21,0.21],[0.1,0.1],1000,0.3,0.3)
        self.firms = [Firm(0.21, 0.1, 1000, 50, 0, 50, 500), Firm(0.21, 0.1, 1000, 50, 0, 50, 0)]
        self.n_firms = n_firms
        self.price = price

        # create a function with for loop that calculate agg_price

    def aggregate_price(self, n_firms, price, theta=theta):
        agg_price = 0
        for i in range(n_firms):
            agg_price += (1/n_firms) * (price**(1-theta))**(1/(1-theta))
            return agg_price

        # " agg_demand
    def aggregate_demand(self, agg_price, K=K, M=M):
        agg_demand = K * (M/agg_price)
        return agg_demand"""



import random
from random import gauss
import statistics
import numpy as np

theta = 1.1 # Elasticity of substitution
K = 1 # Constant
M = 10000 # Money available in total market

# Firms compete and produce imperfect substitutes. Increasing their price may lead to a reduction in demand for their
# product while consumers gravitate to substitutes.

class Firm:
    def __init__(self, tax, sigma, demand, cost, q_other, price, quantity):
        self.tax = tax
        self.sigma = sigma
        self.demand = demand
        self.cost = cost
        self.price = price
        self.quantity = quantity

    def normalize_reward(self, reward):
        reward = 1 - 2**(-reward/1e7)
        return reward

    def firm_demand(self, agg_demand, agg_price, K=10, theta=theta):
        self.demand = K * agg_demand * (self.price / agg_price) ** (-theta)
        return self.demand

    def aggregate_demand(self, agg_price, K=K, M=M):
        agg_demand = K * (M/agg_price)
        return agg_demand

    def step(self, actions, q_other, sigma=0.1): # update to take in the

        price_action, quantity_action = actions
        price_action = np.clip(price_action, -1, 1)
        self.price = abs((price_action * sigma + 1) * self.price)
        quantity_action = np.clip(quantity_action, 0, 1)
        self.quantity = quantity_action * self.demand
        self.q_other = q_other
        penalty = 0

        if (self.quantity + self.q_other > self.demand):
            penalty = self.cost * ((self.quantity + self.q_other - self.demand / 2))  # (self.q_other + 1))

        reward = self.normalize_reward((1-self.tax) * (self.price - self.cost) * self.quantity + penalty)

        return self.get_observations(), reward, False


    def get_observations(self):

        #demand = self.demand / 10000 # Change

        tax = self.tax
        sigma = self.sigma
        price = self.price / 100
        demand = Environment.aggregate_demand(self,
                                              Environment.aggregate_price(self, n_firms=2, price=price, theta=theta))
        cost = self.cost / 10
        q_other = self.q_other / 1000

        return demand, tax, sigma, price, cost, q_other

    def reset(self, q_other):
        #self.cost = gauss(mu=self.cost, sigma=1)
        self.cost = gauss(mu=2, sigma=1)
        self.q_other = q_other

        return self.get_observations()

class SocialPlanner:
    def __init__(self, tax, price_threshold, gamma=1, eta=1):
        self.tax = tax
        self.price_threshold = price_threshold
        self.demand = demand
        self.gamma = gamma
        self.eta = eta

    def step(self, actions, firm_rewards, quantities):
        tax_action, price_threshold_action = actions
        fairness = 0
        r_fair = [0, 0]
        for i in range(len(firm_rewards)):
            fairness += abs(firm_rewards[i] - r_fair[i])
        reward = sum(firm_rewards) - self.gamma*abs(self.demand - sum(quantities)) - self.eta*fairness
        return reward, tax_action, price_threshold_action

    def get_observations(self):
        demand = self.demand / 10000
        #profit =
        return demand, profit # should return normalized observations


class Environment:
    def __init__(self, n_firms, price):
        #self.social = SocialPlanner([0.21,0.21],[0.1,0.1],1000,0.3,0.3)
        self.firms = [Firm(0.21, 0.1, 1000, 1, 0, 50, 500), Firm(0.21, 0.1, 1000, 5, 0, 50, 0)]
        self.n_firms = n_firms
        self.price = price

        # create a function with for loop that calculate agg_price
        # " agg_demand
    def aggregate_price(self, n_firms, price, theta=theta):
        agg_price = 0
        for i in range(n_firms):
            agg_price += (1/n_firms) * (price**(1-theta))**(1/(1-theta))
            return agg_price

    def aggregate_demand(self, agg_price, K=K, M=M):
        agg_demand = K * (M/agg_price)
        return agg_demand
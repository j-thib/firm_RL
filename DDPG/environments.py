import random
from random import gauss
import statistics

import gym
import numpy as np

theta = 1.1 # Elasticity of substitution
K = 1.0 # Constant
M = 10000 # Money available in total market
sigma = 0.1
expected_utility=0

# Firms compete and produce imperfect substitutes. Increasing their price may lead to a reduction in demand for their
# product while consumers gravitate to substitutes.

class firmEnv:
    def __init__(self, tax, sigma, eq_demand, cost, price, quantity, demand):
        self.tax = tax
        self.sigma = sigma
        self.eq_demand = eq_demand
        self.demand = demand
        self.cost = cost
        self.price = price
        self.quantity = quantity

    def firm_demand(self, eq_demand, eq_price, K=K, theta=theta):
        self.demand = K * eq_demand * (self.price / eq_price) ** (-theta)
        return self.demand

    def normalize_reward(self, reward):
        reward = 1 - 2**(-reward/1e7)
        return reward

    def compute_price_from_action(self, actions, sigma=sigma):
        quantity_action, price_action = actions
        price_action = np.clip(price_action, -1, 1)
        self.price = abs((price_action * sigma + 1) * self.price)

    def compute_quantity_from_action(self, actions, sigma=sigma):
        quantity_action, price_action = actions
        quantity_action = np.clip(quantity_action, 0, 1)
        self.quantity = quantity_action * self.eq_demand

    def step(self, actions, eq_demand, eq_price, K=K, theta=theta, sigma=sigma):

        price_action, quantity_action = actions
        price_action = np.clip(price_action, -1, 1)
        self.price = np.clip(abs((price_action * sigma + 1) * self.price), 2, 8)
        quantity_action = np.clip(quantity_action, 0, 1)
        self.quantity = quantity_action * self.eq_demand

        penalty = 0

        if (self.quantity > self.firm_demand(eq_demand, eq_price)):
            penalty = self.cost * (self.quantity - self.firm_demand(eq_demand, eq_price))

        reward = self.normalize_reward((1-self.tax) * (self.price - self.cost) * self.quantity + penalty)
        #reward = (1 - self.tax) * (self.price - self.cost) * self.quantity + penalty

        return self.get_observations(), reward, False


    def get_observations(self):

        #demand = self.demand / 10000 # Change

        tax = self.tax
        sigma = self.sigma
        price = self.price / 10
        eq_demand = self.eq_demand
        cost = self.cost / 10

        return eq_demand, tax, sigma, price, cost

    def reset(self):
        #self.cost = gauss(mu=self.cost, sigma=1)
        self.cost += 0.005 * self.cost

        return self.get_observations()

class socialPlannerEnv:
    def __init__(self, tax_a, tax_b, expected_utility):

        self.tax_a = tax_a
        self.tax_b = tax_b
        self.expected_utility = expected_utility

    def step(self, actions_sp, expected_utility, fairness):
        tax_action_a, tax_action_b = actions_sp

        tax_action_a = np.clip(tax_action_a, 0, 0.5)
        self.tax_a = tax_action_a
        tax_action_b = np.clip(tax_action_b, 0, 0.5)
        self.tax_b = tax_action_b

        reward = np.sqrt(expected_utility) - fairness

        return self.get_observations(), reward, False

    def get_observations(self):

        tax_a = self.tax_a
        tax_b = self.tax_b
        expected_utility = self.expected_utility

        return tax_a, tax_b, expected_utility

    def reset(self):

        return self.get_observations()

class Environment:
    def __init__(self, n_firms):
        self.firms = [firmEnv(tax=0.0, sigma=0.1, eq_demand=1000, cost=1, price=10, quantity=500, demand=500),
                      firmEnv(tax=0.0, sigma=0.1, eq_demand=1000, cost=2, price=10, quantity=500, demand=500)]
        self.social_planner = socialPlannerEnv(tax_a=0.2, tax_b=0.2, expected_utility=0.0)
        self.n_firms = n_firms

        # create a function with for loop that calculate agg_price
        # " agg_demand
    def equilibrium_price(self, n_firms, prices, theta=theta):
        self.eq_price = 0
        for i in range(n_firms):
            self.eq_price += (1/n_firms) * (prices[i]**(1-theta))**(1/(1-theta))

    def equilibrium_demand(self, eq_price, K=K, M=M):
        self.eq_demand = K * (M/self.eq_price)

    def assign_tax(self, firms, social_planner):
        firms[0].get_observations()[1] = social_planner.get_observations()[0]
        firms[1].get_observations()[1] = social_planner.get_observations()[1]
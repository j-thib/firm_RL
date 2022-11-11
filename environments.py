from random import gauss
import statistics
import numpy as np


class Firm:
    def __init__(self, tax, sigma, demand, cost, q_other, price, quantity):
        self.tax = tax
        self.sigma = sigma
        self.demand = demand
        self.cost = cost
        self.q_other = q_other
        self.price = price
        self.quantity = quantity

    def step(self, actions, q_other, sigma=0.1):
        price_action, quantity_action = actions
        price_action = np.clip(price_action, -0.5, 1)
        self.price = (price_action * sigma + 1) * self.price
        quantity_action = np.clip(quantity_action, 0, 1)
        self.quantity = quantity_action * self.demand
        self.q_other = q_other
        penalty = 0

        if (self.quantity + self.q_other > self.demand):
            penalty = self.cost * ((self.quantity + self.q_other - self.demand) / 2) #(self.q_other + 1))

        reward = ((1-self.tax)*(self.price - self.cost)*self.quantity + penalty) / 10000
        return self.get_observations(), reward, False

    def get_observations(self):
        demand = self.demand / 10000
        tax = self.tax
        sigma = self.sigma
        price = self.price / 10
        cost = self.cost / 10
        q_other = self.q_other / 10000

        return demand, tax, sigma, price, cost, q_other

    def reset(self, q_other):
        self.cost = gauss(mu=self.cost, sigma=1)
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
    def __init__(self, n_firms):
        #self.social = SocialPlanner([0.21,0.21],[0.1,0.1],1000,0.3,0.3)
        self.firms = [Firm(0.21, 0.1, 1000, 2, 500, 3, 500), Firm(0.21, 0.1, 1000, 2, 500, 3, 500)]
        self.n_firms = n_firms
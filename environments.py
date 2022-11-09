import random
import statistics

class Firm:
    def __init__(self, tax, price_threshold, demand, cost, q_other, price, quantity):
        self.tax = tax
        self.price_threshold = price_threshold
        self.demand = demand
        self.cost = cost
        self.q_other = q_other
        self.price = price
        self.quantity = quantity

    def step(self, actions, sigma):
        price_action, quantity_action = actions
        self.price = (price_action * sigma + 1) * self.price
        self.quantity = quantity_action * self.demand

        penalty = 0

        if (self.quantity + sum(self.q_other) > self.demand):
            penalty = self.cost((self.quantity + sum(self.q_other) - self.demand) / (len(self.q_other) + 1))

        reward = (1-self.tax)*(self.price - self.cost)*self.quantity + penalty
        return reward

    def get_observations(self):
        return # should return normalized values representing environment.

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
        r_cournot = [0, 0]
        for i in range(len(firm_rewards)):
            fairness += abs(firm_rewards[i] - r_cournot[i])
        reward = sum(firm_rewards) - self.gamma*abs(self.demand - sum(quantities)) - self.eta*fairness
        return reward, tax_action, price_threshold_action

    def get_observations(self):
        demand = self.demand / 10000
        #profit =
        return demand, profit # should return normalized observations


class Environment:
    def __init__(self, n_firms):
        self.social = SocialPlanner([0.21,0.21],[0.1,0.1],1000,0.3,0.3)
        self.firms = [Firm(0.21, 0.1, 1000, 2, 500, 3, 500), Firm(0.21, 0.1, 1000, 2, 500, 3, 500)]
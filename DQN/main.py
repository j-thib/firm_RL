import random
import gym

import environments_discrete
from dqn import Agent
from utils import plot_learning_curve, plotRewards, cumulative, plotPrices, plotQuantities
import numpy as np
from environments_discrete import Environment, firmEnv


env = Environment(n_firms=2)
agent = [Agent(gamma=0.99, epsilon=1.0, batch_size=1, n_actions=10, eps_end=0.01, input_dims=[5], lr=0.001) \
for i in range(env.n_firms)]
#social_agent = Agent(gamma=0.99, epsilon=1.0, batch_size=1, n_actions=9, eps_end=0.01, input_dims=[5], lr=0.001)

score_history = [list(), list()]
eps_history = list()

np.random.seed(42)
random.seed(42)

episodes = 10
steps = 4

score_history = [list(), list()]
price_history = [list(), list()]
quantity_history = [list(), list()]

for i in range(episodes):
    score = [0, 0]
    obs = [None, None]
    act = [None, None]
    done = False
    step = 0
    for j in range(env.n_firms):
        obs[j] = env.firms[j].reset()
    #print(obs)

    while step < steps:
        prices = list()
        quantities = list()
        for j in range(env.n_firms):
            act[j] = agent[j].choose_action(obs[j])
            #print(act[j])
            env.firms[j].compute_price_from_action(act[j])
            env.firms[j].compute_quantity_from_action(act[j])

            prices.append(env.firms[j].price)
            quantities.append(env.firms[j].quantity)

        price_history[0].append(prices[0])
        price_history[1].append(prices[1])
        quantity_history[0].append(quantities[0])
        quantity_history[1].append(quantities[1])
        #print(act)
        #print(prices)

        env.equilibrium_price(n_firms=2, prices=prices, theta=environments_discrete.theta)
        #print(env.eq_price)
        env.equilibrium_demand(env.eq_price, K=environments_discrete.K, M=environments_discrete.M)

        for j in range(env.n_firms):

            observation_, reward, done = env.firms[j].step(act[j], eq_demand=env.eq_demand, eq_price=env.eq_price)
            print(reward)
            #print(reward)
            score[j] += reward
            agent[j].store_transition(obs[j], act[j], reward, observation_, done)
            agent[j].learn()
            obs[j] = observation_
            score_history[j].append(score[j])
        #score[j].append(agent[j].epsilon)
        #eps_history.append(agent[j].epsilon)

        #print(obs)
        step += 1



    avg_score = np.mean(score[-100:])
    #print(i)

    """print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % agent[j].epsilon)

x = [i+1 for i in range(episodes)]
filename = 'learning_curve0.png'
filename1 = 'learning_curve1.png'
plot_learning_curve(x, score[0], eps_history, filename)
plot_learning_curve(x, score[1], eps_history)"""
filename = 'rewards.png'
filename1 = 'cumulative_rewards.png'
filename2 = 'price_history.png'
filename3 = 'quantity_history.png'
plotRewards(score_history, filename)
cumulative(score_history, filename1)
plotPrices(price_history, filename2)
plotQuantities(quantity_history, filename3)

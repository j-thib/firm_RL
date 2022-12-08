import random
import gym

import environments
from dqn import Agent
from utils import plot_learning_curve
import numpy as np
from environments import Environment, firmEnv


env = Environment(n_firms=2, eq_price=5, eq_demand=1000)
agent = [Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[7], lr=0.003) \
for i in range(env.n_firms)]

score_history = [list(), list()]
eps_history = list()

np.random.seed(1)
random.seed(1)

episodes = 100
steps = 4

for i in range(episodes):
    score = 0
    obs = [None, None]
    act = [None, None]
    done = False
    step = 0
    for j in range(env.n_firms):
        obs[j] = env.firms[j].reset()[1-j]

    while step < steps:

        for j in range(env.n_firms):
            prices = []
            act[j] = agent[j].choose_action(obs[j])
            prices.append(firmEnv.compute_price_from_action(env.firms[j], actions=act))

        print(prices)

        env.eq_price = env.equilibrium_price(n_firms=2, prices=prices, theta=environments_test.theta)
        print(env.eq_price)
        env.eq_demand = env.equilibrium_demand(env.eq_price, K=environments_test.K, M=environments_test.M)

        for j in range(env.n_firms):

            observation_, reward, done, info, _ = env.firms[j].step(act[j], env.eq_demand)
            score[j] += reward
            agent[j].store_transition(observation, action, reward, observation_, done)
            agent[j].learn()
            observation[j] = observation_
        scores.append(agent[j].epsilon)
        eps_history.append(agent[j].epsilon)

        step += 1

    avg_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)

x = [i+1 for i in range(n_games)]
filename = 'learning_curve.png'
plot_learning_curve(x, scores, eps_history, filename)
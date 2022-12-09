import random
from statistics import mean, stdev
import environments
from environments import Environment, firmEnv, socialPlannerEnv
from ddpg_nets import Agent
import numpy as np
from utils import plotRewards, cumulative, plotPrices, plotQuantities

steps = 12
episodes = 10
env = Environment(n_firms=2)
agents = [Agent(alpha=0.00025,beta=0.0005, input_dims=[5], tau=0.01, env=env,
                batch_size=2,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99) for i in range(env.n_firms)]
social_agent = Agent(alpha=0.00025,beta=0.0005, input_dims=[4], tau=0.01, env=env,
                batch_size=2,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99)

#agent.load_models()
np.random.seed(42)
random.seed(42)


score_history = [list(), list()]
price_history = [list(), list()]
quantity_history = [list(), list()]
expected_utility = []

for i in range(episodes):
    score = [0, 0]
    score_sp = [0]
    act = [None, None]
    act_sp = [None]
    obs = [None, None]
    obs_sp = [None]
    done = False
    step = 0
    for j in range(env.n_firms):
        obs[j] = env.firms[j].reset()


    while(step<steps):
        prices = list()
        quantities = list()

        for j in range(env.n_firms):
            act[j] = agents[j].choose_action(obs[j])
            env.firms[j].compute_price_from_action(act[j])
            env.firms[j].compute_quantity_from_action(act[j])

            prices.append(env.firms[j].price)
            quantities.append(env.firms[j].quantity)

        price_history[0].append(prices[0])
        price_history[1].append(prices[1])
        quantity_history[0].append(quantities[0])
        quantity_history[1].append(quantities[1])

        env.equilibrium_price(n_firms=env.n_firms, prices=prices, theta=environments.theta)
        env.equilibrium_demand(env.eq_price, K=environments.K, M=environments.M)

        for j in range(env.n_firms):

            new_state, reward, done = env.firms[j].step(act[j], env.eq_demand, env.eq_price)
            score[j] += reward
            agents[j].remember(obs[j], act[j], reward, new_state, int(done))
            agents[j].learn()
            obs[j] = new_state
            score_history[j].append(score[j])
        step += 1

        expected_utility.append((score_history[0][i] + score_history[1][i]) / 2)

    obs_sp = env.social_planner.reset()
    #print(obs_sp)
    act_sp = social_agent.choose_action(obs_sp)
    new_state, reward, done = env.social_planner.step(act_sp, expected_utility)
    print(reward)
    score_sp += reward
    social_agent.remember(obs_sp, act_sp, reward, new_state, int(done))
    social_agent.learn()
    obs_sp = new_state


filename = 'reward.png'
filename1 = 'cumulative-reward.png'
filename2 = 'prices.png'
filename3 = 'quantities.png'
plotRewards(score_history, filename)
cumulative(score_history, filename1)
plotPrices(price_history, filename2)
plotQuantities(quantity_history, filename3)
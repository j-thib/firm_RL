"""import random
from statistics import mean, stdev
from environments import Environment, SocialPlanner, Firm
from ddpg_nets import Agent
import numpy as np
from utils import plotRewards, cumulative, plotPrices

steps = 4
episodes = 100
env = Environment(n_firms=2, price=50)
agents = [Agent(alpha=0.0005,beta=0.00025, input_dims=[6], tau=0.0001, env=env,
                batch_size=16,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99) for i in range(env.n_firms)]


#social_planner = SocialPlanner()
firm1 = env.firms[0]
firm2 = env.firms[1]

#agent.load_models()
np.random.seed(1)
random.seed(1)


score_history = [list(), list()]
price_history = [list(), list()]

for i in range(episodes):
    done = False
    score = [0, 0]
    step = 0
    obs = [None, None]
    act = [None, None]
    for j in range(env.n_firms):
        obs[j] = env.firms[j].reset(env.firms[1-j].quantity)
    while(step<steps):
        for j in range(env.n_firms):
            act[j] = agents[j].choose_action(obs[j])
        # update environment agg_price agg_demand
        env.agg_price = env.aggregate_price(n_firms=2, price=env.price, theta=1.1)
        env.agg_demand = env.aggregate_demand(agg_price=env.agg_price)


        for j in range(env.n_firms):
            new_state, reward, done = env.firms[j].step(act,env.firms[1-j].quantity)
            agents[j].remember(obs[j], act[j], reward, new_state, int(done))
            agents[j].learn()
            #print(reward)
            score[j] += reward
            obs[j] = new_state
            #print(i,obs)
            #env.render()
            score_history[j].append(score[j])
            price_history[j].append(obs[j][4])
            #print(price_history)
        step += 1

        for k in range(env.n_firms):
            obs[k] = env.firms[k].get_observations(env.agg_demand)

filename = 'reward.png'
filename1 = 'cumulative-reward.png'
filename2 = 'prices.png'
plotRewards(score_history, filename)
cumulative(score_history, filename1)
plotPrices(price_history, filename2)

### 1."""

import random
from statistics import mean, stdev
from environments import Environment, SocialPlanner, Firm
from ddpg_nets import Agent
import numpy as np
from utils import plotRewards, cumulative, plotPrices

steps = 6
episodes = 100
env = Environment(n_firms=2, price=0.0001)
agents = [Agent(alpha=0.0005,beta=0.00025, input_dims=[6], tau=0.0001, env=env,
                batch_size=16,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99) for i in range(env.n_firms)]


#social_planner = SocialPlanner()
firm1 = env.firms[0]
firm2 = env.firms[1]

#agent.load_models()
np.random.seed(1)
random.seed(1)


score_history = [list(), list()]
price_history = [list(), list()]

for i in range(episodes):
    done = False
    score = [0, 0]
    step = 0
    obs = [None, None]
    for j in range(env.n_firms):
        obs[j] = env.firms[j].reset(env.firms[1-j].quantity)

    env.agg_price = env.aggregate_price(n_firms=2, price=env.price, theta=1.1)
    env.agg_demand = env.aggregate_demand(agg_price=env.agg_price)
    while(step<steps):
        for j in range(env.n_firms):
            act = agents[j].choose_action(obs[j])
            new_state, reward, done = env.firms[j].step(act,env.firms[1-j].quantity)
            agents[j].remember(obs[j], act, reward, new_state, int(done))
            agents[j].learn()
            score[j] += reward
            obs[j] = new_state
            score_history[j].append(score[j])
            price_history[j].append(obs[j][4])
        step += 1

        for k in range(env.n_firms):
            obs[k] = env.firms[k].get_observations()

filename = 'reward.png'
filename1 = 'cumulative-reward.png'
filename2 = 'prices.png'
plotRewards(score_history, filename)
cumulative(score_history, filename1)
plotPrices(price_history, filename2)
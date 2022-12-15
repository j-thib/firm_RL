import random
from statistics import mean, stdev
import environments
from environments import Environment, firmEnv, socialPlannerEnv
from ddpg_nets import Agent
import numpy as np
from utils import plotRewards, cumulative, plotPrices, plotQuantities, plotSPReward, cumulativeSP

steps = 12
episodes = 500
env = Environment(n_firms=2)
agents = [Agent(alpha=0.00025,beta=0.0005, input_dims=[5], tau=0.01, env=env,
                batch_size=8,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99) for i in range(env.n_firms)]
social_agent = Agent(alpha=0.00025,beta=0.0005, input_dims=[3], tau=0.01, env=env,
                batch_size=8,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99)

#agent.load_models()
np.random.seed(42)
random.seed(42)

score_history = [list(), list()]
sp_score_history = []
price_history = [list(), list()]
quantity_history = [list(), list()]
expected_utility_history = []
expected_utility = 0

for i in range(episodes):
    print(i)
    score = [0, 0]
    score_sp = [0]
    act = [None, None]
    act_sp = [None]
    obs = [None, None]
    obs_sp = [None]
    done = False
    done_sp = False
    step = 0

    obs_sp = env.social_planner.reset()
    act_sp = social_agent.choose_action(obs_sp)
    new_state_sp = env.social_planner.step(act_sp, expected_utility)[0]
    #print("tax 1:", new_state_sp[0], " tax 2:", new_state_sp[1])

    for j in range(env.n_firms):
        obs[j] = env.firms[j].reset()

        obs[j] = list(obs[j])
        if j == 0:
            obs[j][1] = new_state_sp[0]
        else:
            obs[j][1] = new_state_sp[1]
        obs[j] = tuple(obs[j])

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

        expected_utility_history.append((score_history[0][i] + score_history[1][i]) / 2)
    expected_utility = (score_history[0][i] + score_history[1][i]) / 2

    reward_sp = env.social_planner.step(act_sp, expected_utility)[1]
    done_sp = env.social_planner.step(act_sp, expected_utility)[2]
    score_sp += reward_sp
    sp_score_history.append(score_sp)
    social_agent.remember(obs_sp, act_sp, reward_sp, new_state_sp, int(done))
    social_agent.learn()
    obs_sp = new_state_sp

#print(mean(price_history[0]))
#print(mean(price_history[1]))

filename = 'reward.png'
filename1 = 'cumulative-reward.png'
filename2 = 'prices.png'
filename3 = 'quantities.png'
filename4 = 'social-planner-rewards.png'
filename5 = 'social-planner-cumulative-rewards.png'
plotRewards(score_history, filename)
cumulative(score_history, filename1)
plotPrices(price_history, filename2)
plotQuantities(quantity_history, filename3)
plotSPReward(sp_score_history, filename4)
cumulativeSP(sp_score_history, filename5)
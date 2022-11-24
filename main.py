import random
from statistics import mean, stdev
from environments import Environment, SocialPlanner, Firm
from ddpg_nets import Agent
import numpy as np
from utils import plotRewards, cumulative

steps = 12
episodes = 100
env = Environment(n_firms=2)
agents = [Agent(alpha=0.0005,beta=0.00025, input_dims=[6], tau=0.0001, env=env,
                batch_size=16,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99) for i in range(env.n_firms)]


#social_planner = SocialPlanner()
firm1 = env.firms[0]
firm2 = env.firms[1]

#agent.load_models()
np.random.seed(42)
random.seed(42)


score_history = [list(),list()]

for i in range(episodes):
    done = False
    score = [0, 0]
    step = 0
    obs = [None, None]
    for j in range(env.n_firms):
        obs[j] = env.firms[j].reset(env.firms[1-j].quantity)
    while(step<steps):
        for j in range(env.n_firms):
            act = agents[j].choose_action(obs[j])
            new_state, reward, done = env.firms[j].step(act,env.firms[1-j].quantity)
            agents[j].remember(obs[j], act, reward, new_state, int(done))
            agents[j].learn()
            score[j] += reward
            obs[j] = new_state
            print(obs)
            #env.render()
            score_history[j].append(score[j])
        step += 1
    #if i % 25 == 0:
    #    agent.save_models()

    #print('episode ', i, 'score %.2f' % score[0],
          #'trailing 10 games avg %.3f' % np.mean(score_history[-10:]))

filename = 'reward.png'
filename1 = 'cumulative-reward.png'
plotRewards(score_history, filename)
cumulative(score_history, filename1)
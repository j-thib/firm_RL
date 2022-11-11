from environments import Environment, SocialPlanner, Firm
from ddpg_nets import Agent
import numpy as np
from utils import plotRewards

steps = 12
episodes = 10
env = Environment(n_firms=2)
agents = [Agent(alpha=0.000025,beta=0.00025, input_dims=[6], tau=0.001, env=env,
                batch_size=1,  layer1_size=400, layer2_size=300, n_actions=2,
                gamma=0.99, max_size=10) for i in range(env.n_firms)]
#social_planner = SocialPlanner()
firm1 = env.firms[0]
firm2 = env.firms[1]

#agent.load_models()
np.random.seed(0)

score_history = [list(),list()]

for i in range(episodes):
    done = False
    score = 0
    step = 0
    for j in range(env.n_firms):
        obs = env.firms[j].reset(env.firms[1-j].quantity)
    while(step<steps):
        for j in range(env.n_firms):
            act = agents[j].choose_action(obs)
            new_state, reward, done = env.firms[j].step(act,env.firms[1-j].quantity)
            agents[j].remember(obs, act, reward, new_state, int(done))
            agents[j].learn()
            score += reward
            obs = new_state
            print(obs[5])
            #env.render()
            score_history[j].append(score)
        step += 1
    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'firm_competition-alpha000025-beta00025-400-300.png'
plotRewards(score_history, filename)
import matplotlib.pyplot as plt
import numpy as np

def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

def plotRewards(score_history, filename):
    plt.ylabel('Reward')
    plt.xlabel('Steps')
    plt.plot(range(len(score_history[0])), score_history[0], color='blue', label='Firm 0')
    plt.plot(range(len(score_history[1])), score_history[1], color='red', label='Firm 1')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def cumulative(score_history, filename1):
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Steps')
    plt.plot(range(len(score_history[0])), np.cumsum(score_history[0]), color='blue', label='Firm 0')
    plt.plot(range(len(score_history[0])), np.cumsum(score_history[1]), color='red', label='Firm 1')
    plt.legend()
    plt.savefig(filename1)
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


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
    plt.plot(range(len(score_history[1])), np.cumsum(score_history[1]), color='red', label='Firm 1')
    plt.legend()
    plt.savefig(filename1)
    plt.clf()

def plotPrices(price_history, filename2):
    plt.ylabel('Prices')
    plt.xlabel('Steps')
    plt.plot(range(len(price_history[0])), price_history[0], color='blue', label='Firm 0')
    plt.plot(range(len(price_history[1])), price_history[1], color='red', label='Firm 1')
    plt.legend()
    plt.savefig(filename2)
    plt.clf()

def plotQuantities(quantity_history, filename3):
    plt.ylabel('Quantities')
    plt.xlabel('Steps')
    plt.plot(range(len(quantity_history[0])), quantity_history[0], color='blue', label='Firm 0')
    plt.plot(range(len(quantity_history[1])), quantity_history[1], color='red', label='Firm 1')
    plt.legend()
    plt.savefig(filename3)
    plt.clf()

def plotSPReward(sp_score_history, filename4):
    plt.ylabel('SP reward')
    plt.xlabel('Steps')
    plt.plot(range(len(sp_score_history)), sp_score_history, color='blue', label='SP rewards')
    plt.legend()
    plt.savefig(filename4)
    plt.clf()

def cumulativeSP(sp_score_history, filename5):
    plt.ylabel('SP Cumulative Reward')
    plt.xlabel('Steps')
    plt.plot(range(len(sp_score_history)), np.cumsum(sp_score_history), color='green', label='SP rewards')
    plt.legend()
    plt.savefig(filename5)
    plt.clf()
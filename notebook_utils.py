import numpy as np
from best_arm_algos import *


def compute_h1(means):
    return np.sum(np.array((sorted(means)[:len(means)-1] -
                  np.max(means)))**(-2))


def compute_arm_probabilities(samples, n_arms):
    n_timesteps = len(samples.keys())
    proba_table = np.zeros(shape=(n_timesteps - n_arms, n_arms))
    for t in range(n_timesteps-n_arms):
        total = 0
        for i in range(t, t+n_arms - 1):
            for j in range(len(samples[i])):
                proba_table[t, samples[i][j]] += 1
                total += 1
        proba_table[t, :] *= (1/total)
    return proba_table


def get_avg_probabilities(algo, means, scales, n_runs=1000,
                          n_steps=4000,  n_arms=6,):
    tables = np.zeros(shape=(n_runs, n_steps, n_arms))
    for run in range(n_runs):
        tables[run] = compute_arm_probabilities(algo(means, scales,
                                                experiment=True,
                                                n_steps=n_steps), n_arms)
    table = np.mean(tables, axis=0)
    return table


def plot_table(ax, table, n_arms, h1=1):
    for i in range(n_arms):
        ax.set_xlabel('Number of pulls (units of H1)')
        ax.set_ylabel('P(I_t = i)')
        ax.plot(np.array(range(table.shape[0]))/h1,
                table[:, i], label=str(i+1))
    ax.legend()

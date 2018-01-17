import json
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt


def get_linear_fit_models(exps):
    """
    For each experiment, estimate a linear trend throughout the data
    and determine its slope as a measure of how favourable that trend is.
    """

    linear_fit_models = [[] for _ in exps]

    for exp_idx, exp in enumerate(exps):
        episode_reward_sums = [np.sum(rewards) for rewards in exp['rewards']]
        x_values = [i for i in range(0, len(episode_reward_sums))]
        linear_fit_models[exp_idx] = np.polyfit(x_values, episode_reward_sums, 1)

    return np.array(linear_fit_models)


def linear_fit_plot_single(exp, linear_fit_model):
    # plot rewards over episode
    y = [np.sum(rewards) for rewards in exp['rewards']]
    x = np.arange(0, len(y))
    plt.plot(x, y)

    # plot linear trend line
    plt.plot(x, linear_fit_model[0] * x + linear_fit_model[1])
    plt.title("Experiment [identifier/parameters]\nwith linear trend: {:5f}x + {:5f}".format(linear_fit_model[0], linear_fit_model[1]))

    plt.show()


def linear_fit_plot_multiple(exps, linear_fit_models):
    for exp_idx, exp in enumerate(exps):
        linear_fit_plot_single(exp, linear_fit_models[exp_idx])


def get_n_highest_slopes(linear_fit_models, n):
    highest_indexes = linear_fit_models[:, 0].argsort()[-n:]
    return highest_indexes, linear_fit_models[highest_indexes]


if __name__ == '__main__':
    experiments = None

    print("[INFO]: Starting to load.")
    ct = time.time()
    try:
        with open("experiments.pickle", "rb") as f:
            experiments = pickle.load(f)
    except FileNotFoundError as e:
        exit(e)
    print("[INFO]: Done loading in {}s.".format(time.time() - ct))

    lines = get_linear_fit_models(experiments)
    h_idx, h_slp = get_n_highest_slopes(lines, 5)
    linear_fit_plot_multiple([experiments[i] for i in h_idx], h_slp)



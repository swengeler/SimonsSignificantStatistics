import json
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt


def get_parameters(experiment):
    return experiment['parameters']


def get_average_reward(experiment):
    rewards = experiment['rewards']

    average_reward = 0
    for episode_reward in rewards:
        average_reward += np.array(episode_reward).sum() / len(episode_reward)

    return average_reward / len(rewards)


def plot_reward_over_experiment(experiment):
    rewards = experiment['rewards']

    x = [n for n in range(0, len(rewards))]
    y = [np.array(episode).sum() for episode in rewards]

    plt.plot(x, y)
    plt.show()


def get_max_episode_reward(experiment):
    rewards = experiment['rewards']

    max = 0
    max_idx = 0
    for idx, episode in enumerate(rewards):
        episode_reward = np.array(episode).sum()

        if episode_reward > max:
            max = episode_reward
            max_idx = idx

    return max, max_idx


if __name__ == '__main__':
    experiments = None

    experiment = None
    with open('../data-long/DQN-episode_19100-reward_type_0-1516387628.9837182.json', 'r') as jsonf:
        experiment = json.load(jsonf)

    plot_reward_over_experiment(experiment)

    # print("starting to load")
    # ct = time.time()
    # try:
    #     with open("experiments.pickle", "rb") as f:
    #         experiments = pickle.load(f)
    # except FileNotFoundError as e:
    #     exit(e)
    # print("done loading in {} s".format(time.time() - ct))
    #
    # max = 0
    # episode_idx = 0
    # experiment_idx = 0
    # for idx, exp in enumerate(experiments):
    #     m, i = get_max_episode_reward(exp)
    #     if m > max:
    #         print("found new max {} at experiment {} in episode {}".format(m, idx, i))
    #         max, idx, experiment_idx = m, i, idx
    #
    # print(get_parameters(experiments[experiment_idx]))
    # plot_reward_over_experiment(experiments[experiment_idx])











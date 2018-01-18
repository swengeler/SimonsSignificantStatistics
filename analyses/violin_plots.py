import json
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt


def get_param_value_dict(exps, after_epsilon_decay=False):
    """
    Go through all given experiments and gather all values of the average reward (over
    the entire experiment) for each parameter and parameter value in a dictionary.
    """

    parameters = exps[0]['parameters'].keys()
    param_dict = {param: dict() for param in parameters}

    for exp in exps:
        # calculate average reward across the entire experiment,
        # might want average reward per episode?
        if not exp:
            print('[WARNING]: Experiment is None.')
            continue
        elif (exp['parameters']['batchsize'] / exp['parameters']['memory_size']) not in [0.05, 0.1, 0.25, 0.5, 1.0]:
            print('[WARNING]: Batch size is wrong.')
            continue
        elif (exp['parameters']['epsilon_decay_episodes_required'] / exp['parameters']['num_episodes']) not in [0.1, 0.5, 0.9]:
            print('[WARNING]: Number of episodes for epsilon decay is wrong.')
            continue

        if after_epsilon_decay:
            reward_avg = np.mean([np.sum(episode_rewards[exp['parameters']['epsilon_decay_episodes_required']:])
                                  for episode_rewards in exp['rewards']])
        else:
            reward_avg = np.mean([np.sum(episode_rewards) for episode_rewards in exp['rewards']])

        for param in parameters:
            # if the particular parameter value has not been added to the
            # dictionary yet (e.g. learning_rate=0.1), add it and start a
            # list of all values occurring for that parameter value
            param_value = exp['parameters'][param]

            if param == 'batchsize':
                param_value = exp['parameters'][param] / exp['parameters']['memory_size']
                # print("[INFO]: batchsize fraction: {}, batchsize: {}, memory_size: {}".format(param_value,
                #                                                                               exp['parameters'][param],
                #                                                                               exp['parameters']['memory_size']))
            elif param == 'epsilon_decay_episodes_required':
                param_value = exp['parameters'][param] / exp['parameters']['num_episodes']
            elif isinstance(param_value, list):
                param_value = tuple(param_value)

            if param_value not in param_dict[param].keys():
                param_dict[param][param_value] = [reward_avg]
            else:
                param_dict[param][param_value].append(reward_avg)

    return param_dict


def violin_plot_single(param_name, param_value_dict):
    x_position = 2
    for param_value, param_value_data in param_value_dict.items():
        plt.violinplot(param_value_data, [x_position], widths=0.75)
        x_position += 2
    plt.xticks([i for i in range (2, 2 * len(param_value_dict) + 1, 2)],
               ["{}\n[{}]".format(param_value, len(param_value_data)) for param_value, param_value_data in param_value_dict.items()])
    plt.title("Violin plots for parameter {}".format(param_name))
    plt.show()


def violin_plot_all(param_dict):
    for param_name, param_value_dict in param_dict.items():
        violin_plot_single(param_name, param_value_dict)


if __name__ == '__main__':
    experiments = None

    print("[INFO]: Starting to load.")
    ct = time.time()
    try:
        with open("../experiments-new.pickle", "rb") as f:
            experiments = pickle.load(f)
    except FileNotFoundError as e:
        exit(e)
    print("[INFO]: Done loading in {}s.".format(time.time() - ct))

    dictionary = get_param_value_dict(experiments, True)
    violin_plot_all(dictionary)

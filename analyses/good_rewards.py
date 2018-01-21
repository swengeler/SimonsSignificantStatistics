import pickle
import time
from distribution import create_histogram


def get_rewards(experiment, good_reward):
    rewards = []
    for i in range(len(experiment['rewards'])):
        reward = experiment['rewards'][i]
        for j in range(len(reward)):
            if reward[j] > good_reward:
                rewards.append(reward[j])
    return rewards


if __name__ == '__main__':
    experiments = None

    print("starting to load")
    ct = time.time()
    try:
        with open("experiments.pickle", "rb") as f:
            experiments = pickle.load(f)
    except FileNotFoundError as e:
        exit(e)
    print("done loading in {} s".format(time.time() - ct))

    for i in range(3):
        create_histogram(get_rewards(experiments[i], 0), 'Good rewards')


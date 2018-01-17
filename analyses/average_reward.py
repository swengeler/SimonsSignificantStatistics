from distribution import create_histogram
import time
import pickle

def get_rewards(experiment):
    rewards = experiment['rewards']
    index = int(len(rewards) * 0.9)
    return rewards[index:]

def get_average_reward(rewards):
    sums = 0
    total_length = 0
    for reward in rewards:
        sums += sum(reward)
        total_length += len(reward)
    average = sums/total_length
    return average

def reward_histogram(experiments):
    averages = []
    for experiment in experiments:
        averages.append(get_average_reward(get_rewards(experiment)))
    print(averages)
    create_histogram(averages, 'average rewards')

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

    reward_histogram(experiments)
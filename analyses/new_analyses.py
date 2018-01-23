import time
import numpy as np
import matplotlib.pyplot as plt
from load_files import *
from scipy.stats import gaussian_kde

def action_size_over_time(path):
    before = time.time()
    actions = load_action_history(path)
    print("Loaded in {}s.".format(time.time() - before))

    first_joint = []
    second_joint = []
    counter = 0
    dictionary = {}
    for i in range(len(actions)):
        for j in range(len(actions[i])):
            if actions[i][j] not in dictionary.keys():
                dictionary[actions[i][j]] = 0
            else:
                dictionary[actions[i][j]] += 1
            # first_joint.append(actions[i][j][0])
            # second_joint.append(actions[i][j][1])
            counter += 1

    print(dictionary)
    # plt.figure(0)
    # plt.plot([i for i in range(counter)], first_joint)
    # plt.figure(1)
    # plt.plot([i for i in range(counter)], second_joint)
    # plt.show()


def max_q_over_time(path):
    states = load_state_history(path)
    points = []
    model = None
    for i in range(0, 5000, 1):
        if i % 50 == 0:
            model = load_model(path, i, 81)
        for j in range(0, len(states[i]), 50):
            points.append(np.max(model.predict(np.array(states[i][j]).reshape([1, 6]))))

    plt.plot([i for i in range(len(points))], points)
    plt.show()


def position_histogram(path, index):
    states = load_state_history(path)

    counter = 0
    first_state = []
    for i in range(len(states)):
        for j in range(len(states[i])):
            if index % 2 == 0 and states[i][j][index] > 2 * np.pi:
                # print("{}, {}".format(states[i][j][0], np.floor(states[i][j][0] / (2 * np.pi))))
                first_state.append(states[i][j][index] - 2 * np.pi * np.floor(states[i][j][index] / (2 * np.pi)))
                counter += 1
            else:
                first_state.append(states[i][j][index])

            if index % 2 == 0:
                first_state[-1] = first_state[-1] - np.pi

    plt.figure(0)
    plt.hist(first_state, 51)
    plt.show()


def action_difference_histogram(path, index=0):
    actions = load_action_history(path)

    first_joint = []
    second_joint = []
    for i in range(len(actions)):
        for j in range(len(actions[i]) - 1):
            first_joint.append(actions[i][j + 1][index] - actions[i][j][index])
            # second_joint.append(actions[i][j + 1][1] - actions[i][j][1])

    plt.figure(0)
    plt.hist(first_joint[:500*199], 9)
    plt.figure(1)
    plt.hist(first_joint[2500*199:3000*199], 9)
    plt.figure(2)
    plt.hist(first_joint[4500*199:], 9)
    plt.show()


if __name__ == "__main__":
    directory_name = "../test-analysis/23-01-2018_02-16-10_05a2090c-06e1-4d55-9d94-c188b6607b6e/"
    action_difference_histogram(directory_name, 1)

    # before = time.time()
    # states = load_state_history(directory_name)
    # print("Loaded in {}s.".format(time.time() - before))

    # print(states[0][1])
    # first_state = [0.0 for _ in range(len(states) * len(states[0]))]
    # i = 0
    # first_state = []
    # for episode in states:
    #     for s in states:
    #         first_state.append(s[0][0])
    #         i += 1
    #         if i > 200:
    #             break

    # counter = 0
    # first_state = []
    # for i in range(len(states)):
    #     for j in range(len(states[i])):
    #         if states[i][j][2] > 2 * np.pi:
    #             # print("{}, {}".format(states[i][j][0], np.floor(states[i][j][0] / (2 * np.pi))))
    #             first_state.append(states[i][j][2] - 2 * np.pi * np.floor(states[i][j][2] / (2 * np.pi)))
    #             counter += 1
    #         else:
    #             first_state.append(states[i][j][2])
    #         first_state[-1] = first_state[-1] - np.pi
    #         # print(states[i][j][0])
    #
    #
    # print(counter)
    # plt.figure(0)
    # plt.hist(first_state[0:500])
    # plt.figure(1)
    # plt.hist(first_state[2500:3000])
    # plt.figure(2)
    # plt.hist(first_state[-500:])
    # plt.figure(3)
    # plt.hist(first_state, 50)
    #plt.scatter(np.random.randn(len(first_state)), first_state)
    # plt.show()
    #x = np.random.randn(len(first_state))
    #y = first_state
    #xy = np.vstack([x, y])
    #z = gaussian_kde(xy)(xy)

    #fig, ax = plt.subplots()
    #ax.scatter(x, y, c=z, s=100, edgecolor='')
    #plt.show()
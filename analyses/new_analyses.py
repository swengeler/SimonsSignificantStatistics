import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps

from load_files import *


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


def max_q_over_time(path, smoothing=False, window=49, label=None):
    states_q = load_state_history(path)
    points = []
    model = None
    for i in range(0, 5000, 1):
        if i % 50 == 0:
            model = load_model(path, i, 81)
        for j in range(0, len(states_q[i]), 100):
            points.append(np.max(model.predict(np.array(states_q[i][j]).reshape([1, 6]))))
        # points.append(np.max(model.predict(np.array(states[i][100]).reshape([1, 6]))))

    plt.plot([i/3 for i in range(len(points))], points if not smoothing else sps.savgol_filter(points, window, 1))
    plt.xlabel("Episodes (3 values per episode\ntaken at start, middle and end)")
    plt.ylabel("Maximum Q-value")
    plt.subplots_adjust(bottom=0.2, left=0.2)

    if not label:
        plt.show()
    else:
        dir_name = "./plots/" + label + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(dir_name + "max_q_value.png")
        plt.close(True)


def certain_state_q_over_time(path, state, smoothing=False):
    points = []
    model = None
    for i in range(0, 5000, 1):
        if i % 50 == 0:
            model = load_model(path, i, 81)
        points.append(np.max(model.predict(np.array(state).reshape([1, 6]))))

    plt.plot([i*50 for i in range(len(points))], points if not smoothing else sps.savgol_filter(points, 99, 1))
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
    plt.hist(first_state[-500*200:], 51)
    plt.figure(1)
    plt.hist(first_state[:500*200], 51)
    plt.figure(2)
    plt.hist(first_state[2500*200:3000*200], 51)
    # plt.figure(0)
    # plt.hist(first_state, 51)
    # plt.figure(1)
    # print("THing:", int(len(first_state) / 3))
    # plt.hist(first_state[:int(len(first_state) / 3)], 51)
    # plt.figure(2)
    # plt.hist(first_state[int(len(first_state) / 3):int(2 * len(first_state) / 3)], 51)
    # plt.figure(3)
    # plt.hist(first_state[int(2 * len(first_state) / 3):], 51)
    plt.show()


def action_difference_histogram(path, bins=49, label=None):
    actions = load_action_history(path)

    first_joint = []
    second_joint = []
    for i in range(len(actions)):
        for j in range(len(actions[i]) - 1):
            first_joint.append(actions[i][j + 1][0] - actions[i][j][0])
            second_joint.append(actions[i][j + 1][1] - actions[i][j][1])

    dir_name = "./"
    if label:
        dir_name = "./plots/" + label + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    plt.figure(0)
    plt.hist(first_joint, bins)
    plt.hist(second_joint, bins)
    plt.xlabel("Difference in positions\nimmediately following each other")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "action_diff_all.png")
        plt.close(True)

    plt.figure(1)
    plt.hist(first_joint[:500 * 200], bins)
    plt.hist(second_joint[:500 * 200], bins)
    plt.xlabel("Difference in positions\nimmediately following each other")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "action_diff_first_500.png")
        plt.close(True)

    plt.figure(2)
    plt.hist(first_joint[2500 * 200:3000 * 200], bins)
    plt.hist(second_joint[2500 * 200:3000 * 200], bins)
    plt.xlabel("Difference in positions\nimmediately following each other")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "action_diff_middle_500.png")
        plt.close(True)

    plt.figure(3)
    plt.hist(first_joint[4500 * 200:], bins)
    plt.hist(second_joint[4500 * 200:], bins)
    plt.xlabel("Difference in positions\nimmediately following each other")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "action_diff_last_500.png")
        plt.close(True)

    if not label:
        plt.show()


def get_experiment_summed_rewards(experiment_rewards):
    ep_rewards = []
    for episode in experiment_rewards:
        s = np.sum(episode)
        ep_rewards.append(s)
    return ep_rewards


def plot_episode_reward(path, smoothing=False, window=49, plot_linear_fit=False, label=None):
    rewards_r = load_reward_history(path)

    ep_rewards = get_experiment_summed_rewards(rewards_r)
    linear_fit = np.polyfit([i for i in range(len(ep_rewards))], ep_rewards, 1)

    plt.plot([i for i in range(len(ep_rewards))], ep_rewards if not smoothing else sps.savgol_filter(ep_rewards, window, 1), label="data")
    if plot_linear_fit:
        plt.plot([i for i in range(len(ep_rewards))], [linear_fit[0] * i + linear_fit[1] for i in range(len(ep_rewards))], label="line approx.")

    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward per episode")
    plt.subplots_adjust(bottom=0.2, left=0.2)

    if not label:
        plt.show()
    else:
        dir_name = "./plots/" + label + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(dir_name + "cumulative_reward.png")
        plt.close(True)


def plot_max_min_avg_reward(path, smoothing=False, window=49, label=None):
    rewards_mma = load_reward_history(path)

    max_rewards = []
    min_rewards = []
    avg_rewards = []
    for episode in rewards_mma:
        max_rewards.append(np.max(episode))
        min_rewards.append(np.min(episode))
        avg_rewards.append(np.mean(episode))

    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(max_rewards))], max_rewards if not smoothing else sps.savgol_filter(max_rewards, window, 1), label="max")
    ax.plot([i for i in range(len(min_rewards))], min_rewards if not smoothing else sps.savgol_filter(min_rewards, window, 1), label="min")
    ax.plot([i for i in range(len(avg_rewards))], avg_rewards if not smoothing else sps.savgol_filter(avg_rewards, window, 1), label="avg")

    ax.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Reward per episode")
    plt.subplots_adjust(bottom=0.2, left=0.2)

    if not label:
        plt.show()
    else:
        dir_name = "./plots/" + label + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(dir_name + "max_min_avg.png")
        plt.close(True)


def plot_average_episode_reward(path, count=10, smoothing=False):
    all_ep_rewards = []
    for i, name in enumerate(os.listdir(path)):
        if i >= count:
            break
        if os.path.isdir(os.path.join(path, name)):
            rewards = load_reward_history(path + name + "/")
            all_ep_rewards.append(get_experiment_summed_rewards(rewards))
        print("Loaded {} reward histories.".format(len(all_ep_rewards)))

    averaged_ep_rewards = np.average(np.array(all_ep_rewards), 0)
    plt.plot([i for i in range(len(averaged_ep_rewards))], averaged_ep_rewards if not smoothing else sps.savgol_filter(averaged_ep_rewards, 99, 1))
    plt.show()


def pendulum_position_histogram(path, bins=51, label=None):
    states_p = load_state_history(path)

    all_states = []
    upper_states = []
    upper_low_vel_states = []
    for i in range(len(states_p)):
        for j in range(len(states_p[i])):
            if states_p[i][j][0] > 2 * np.pi:
                all_states.append(states_p[i][j][0] - 2 * np.pi * np.floor(states_p[i][j][0] / (2 * np.pi)))
            else:
                all_states.append(states_p[i][j][0])

            all_states[-1] = all_states[-1] - np.pi

            if abs(states_p[i][j][0]) <= 1/4 * np.pi:
                upper_states.append(all_states[-1])
                if abs(states_p[i][j][1]) < 2 * np.pi:
                    upper_low_vel_states.append(all_states[-1])

    dir_name = "./"
    if label:
        dir_name = "./plots/" + label + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    if label:
        with open(dir_name + "pctg_upper.txt", "w") as f:
            f.write("Percentage in upper region: {}\n".format(len(upper_states) / len(all_states)))
            f.write("Percentage in upper region with low velocity: {}\n".format(len(upper_low_vel_states) / len(all_states)))

    plt.figure(0)
    plt.hist(all_states, bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2)
    if label:
        plt.savefig(dir_name + "pend_pos_all.png")
        plt.close(True)

    plt.figure(1)
    plt.hist(all_states[:500*200], bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2)
    if label:
        plt.savefig(dir_name + "pend_pos_first_500.png")
        plt.close(True)

    plt.figure(2)
    plt.hist(all_states[2500*200:3000*200], bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2)
    if label:
        plt.savefig(dir_name + "pend_pos_middle_500.png")
        plt.close(True)

    plt.figure(3)
    plt.hist(all_states[4500*200:], bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2)
    if label:
        plt.savefig(dir_name + "pend_pos_last_500.png")
        plt.close(True)

    if not label:
        plt.show()


def motor_position_histogram(path, bins=51, label=None):
    states_m = load_state_history(path)

    all_states_0 = []
    all_states_1 = []
    for i in range(len(states_m)):
        for j in range(len(states_m[i])):
            # lower motor
            if states_m[i][j][2] > 2 * np.pi:
                all_states_0.append(states_m[i][j][2] - 2 * np.pi * np.floor(states_m[i][j][2] / (2 * np.pi)))
            else:
                all_states_0.append(states_m[i][j][2])

            all_states_0[-1] = all_states_0[-1] - np.pi

            # upper motor
            if states_m[i][j][4] > 2 * np.pi:
                all_states_1.append(states_m[i][j][4] - 2 * np.pi * np.floor(states_m[i][j][4] / (2 * np.pi)))
            else:
                all_states_1.append(states_m[i][j][4])

            all_states_1[-1] = all_states_1[-1] - np.pi

    dir_name = "./"
    if label:
        dir_name = "./plots/" + label + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    plt.figure(0)
    plt.hist(all_states_0, bins)
    plt.hist(all_states_1, bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "mot_pos_all.png")
        plt.close(True)

    plt.figure(1)
    plt.hist(all_states_0[:500*200], bins)
    plt.hist(all_states_1[:500*200], bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "mot_pos_first_500.png")
        plt.close(True)

    plt.figure(2)
    plt.hist(all_states_0[2500*200:3000*200], bins)
    plt.hist(all_states_1[2500*200:3000*200], bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "mot_pos_middle_500.png")
        plt.close(True)

    plt.figure(3)
    plt.hist(all_states_0[4500*200:], bins)
    plt.hist(all_states_1[4500*200:], bins)
    plt.xlabel("Angular position in radians\n(0 as upright position)")
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.2, left=0.2)
    if label:
        plt.savefig(dir_name + "mot_pos_last_500.png")
        plt.close(True)

    if not label:
        plt.show()


if __name__ == "__main__":
    # experiment used for reward/done function 0
    reward_0_dir = "../normal-index-0/22-01-2018_23-45-20_42bf3e51-61a9-4e8b-a788-951ae88a40fb/"

    # experiment used for reward/done function 0
    reward_1_dir = "../normal-index-1//"

    # experiment used for reward/done function 0
    reward_2_dir = "../normal-index-2//"

    random_dir = "../random-index-0//"

    dirs = [reward_0_dir, reward_1_dir, reward_2_dir, random_dir]

    for i, dir in enumerate(dirs):
        if i > 0:
            break
        max_q_over_time(dir, smoothing=True, label=("reward_" + str(i) if i < 3 else "random_0"))
        motor_position_histogram(dir, label=("reward_" + str(i) if i < 3 else "random_0"))
        plot_episode_reward(dir, False, label=("reward_" + str(i) if i < 3 else "random_0"))
        plot_max_min_avg_reward(dir, True, label=("reward_" + str(i) if i < 3 else "random_0"))
        action_difference_histogram(dir, label=("reward_" + str(i) if i < 3 else "random_0"))
        pendulum_position_histogram(dir, label=("reward_" + str(i) if i < 3 else "random_0"))



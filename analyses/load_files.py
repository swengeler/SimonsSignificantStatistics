import csv
import ast
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def load_reward_history(directory_path):
    file_name = directory_path + "reward.csv"

    reward_history = []
    line_counter = 0
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            print(line_counter)
            episode_reward_history = []
            for col in row:
                if col is '':
                    continue
                col = ast.literal_eval(col)
                episode_reward_history.append(col)
            reward_history.append(episode_reward_history)
            line_counter += 1

    return reward_history


def load_episode_reward_history(directory_path, episode_idx):
    file_name = directory_path + "reward.csv"

    line_counter = 0
    episode_reward_history = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if line_counter == episode_idx:
                episode_reward_history = []
                for col in row:
                    if col is '':
                        continue
                    col = ast.literal_eval(col)
                    episode_reward_history.append(col)
                break
            else:
                line_counter += 1

    return episode_reward_history


def load_action_history(directory_path):
    file_name = directory_path + "action.csv"

    action_history = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            episode_action_history = []
            for col in row:
                if col is '':
                    continue
                col = ast.literal_eval(col)
                episode_action_history.append(col)
            action_history.append(episode_action_history)

    return action_history


def load_episode_action_history(directory_path, episode_idx):
    file_name = directory_path + "action.csv"

    line_counter = 0
    episode_action_history = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if line_counter == episode_idx:
                episode_action_history = []
                for col in row:
                    if col is '':
                        continue
                    col = ast.literal_eval(col)
                    episode_action_history.append(col)
                break
            else:
                line_counter += 1

    return episode_action_history


def load_max_q_history(directory_path):
    file_name = directory_path + "max-q.csv"

    max_q_history = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            episode_max_q_history = []
            for col in row:
                if col is '':
                    continue
                elif col is 'nan':
                    col = np.nan
                else:
                    col = ast.literal_eval(col)
                episode_max_q_history.append(col)
            max_q_history.append(episode_max_q_history)

    return max_q_history


def load_episode_max_q_history(directory_path, episode_idx):
    file_name = directory_path + "max-q.csv"

    line_counter = 0
    episode_max_q_history = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if line_counter == episode_idx:
                episode_max_q_history = []
                for col in row:
                    if col is '':
                        continue
                    elif col is 'nan':
                        col = np.nan
                    else:
                        col = ast.literal_eval(col)
                    episode_max_q_history.append(col)
                break
            else:
                line_counter += 1

    return episode_max_q_history


def load_state_history(directory_path):
    file_name = directory_path + "state.csv"

    state_history = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            episode_state_history = []
            for col in row:
                if col is '':
                    continue
                col = ast.literal_eval(col)
                episode_state_history.append(col)
            state_history.append(episode_state_history)

    return state_history


def load_episode_state_history(directory_path, episode_idx):
    file_name = directory_path + "state.csv"

    line_counter = 0
    episode_state_history = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if line_counter == episode_idx:
                episode_state_history = []
                for col in row:
                    if col is '':
                        continue
                    col = ast.literal_eval(col)
                    episode_state_history.append(col)
                break
            else:
                line_counter += 1

    return episode_state_history


def load_model(directory_path, episode_idx, action_size):
    if episode_idx % 50 != 0:
        print("Can only load weights for every 50 episodes.")
        return None

    model = Sequential()
    model.add(Dense(20, input_dim=6, activation="tanh"))

    for i in range(1, 2):
        model.add(Dense(20, activation="tanh"))

    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001, decay=0.0))

    file_path = directory_path + "weights-ep-{}".format(episode_idx)
    model.load_weights(file_path)

    return model


if __name__ == "__main__":
    test_directory = "../test/"
    rewards = load_action_history(test_directory)
    print(rewards[0])
import pickle
import os
import json
import time

def load_all_files(load_few=False) -> list:
    filenames = get_all_filenames()
    if load_few and len(filenames) > 10:
        filenames = filenames[0:10]

    objects = [None for _ in range(0, len(filenames))]

    for idx, file in enumerate(filenames):
        if (idx + 1) % 100 == 0:
            print("loaded {} files".format(idx + 1))

        experiment = import_experiment(file)
        parameter_names = experiment['parameters']
        parameter_names['epsilon_decay_episodes_required'] = parameter_names.pop('epsilon_decay_per_step')

        rewards = experiment['rewards']
        for i in range(len(rewards)):
            for j in range(len(rewards[i])):
                rewards[i][j] = rewards[i][j]/10

        objects[idx] = experiment

    return objects


def import_experiment(fn):
    with open(fn, 'r') as jsonf:
        return json.load(jsonf)


def get_all_filenames():
    path = "../data/"
    return [path + file for file in os.listdir("../data/") if ".json" in file]


if __name__ == '__main__':
    print("loading experiments")
    ct = time.time()
    experiments = load_all_files(True)
    print("finished loading in {} s".format(time.time() - ct))

    obj = {}
    obj["experiments"] = experiments

    with open("experiments.pickle", "wb") as f:
        pickle.dump(experiments, f, protocol=pickle.HIGHEST_PROTOCOL)

import pickle
import os
import json
import time
from json import JSONDecodeError
from numpy import pi


def get_env_json():
    obj = {}
    obj['description'] = "simulation"
    obj['physical_parameters'] = (0.04, 0.09, 0.12, 0.03, 0.0005, 9.81)
    obj['sim_ticks_per_step'] = 10
    obj['sim_interval'] = 0.005
    obj['sim_threshold'] = 0.001
    obj['sim_max_acceleration'] = 50.0
    obj['sim_kp'] = 20.0
    obj['sim_ka'] = 3.0
    obj['sim_acceleration_control'] = False
    obj['sim_acceleration_limit'] = pi
    obj['reward_function'] = {
        'average': False,
        'index': 0,
        'parameters': (1/6 * pi, 2 * pi, 10)
    }
    return obj


def load_all_files(load_few=False) -> list:
    filenames = get_all_filenames()
    if load_few and len(filenames) > 10:
        filenames = filenames[0:10]
    objects = [None for _ in range(0, len(filenames))]

    idx_offset = 0
    for idx, file in enumerate(filenames):
        if (idx + 1) % 100 == 0:
            print("[INFO]: Loaded {} filess".format(idx + 1))

        try:
            experiment = import_experiment(file)
        except JSONDecodeError:
            print("[WARNING]: Could not load file.")
            objects.remove(None)
            idx_offset -= 1
            continue
        parameter_names = experiment['parameters']
        if 'epsilon_decay_per_step' in parameter_names.keys():
            parameter_names['epsilon_decay_episodes_required'] = parameter_names.pop('epsilon_decay_per_step')

        if parameter_names['num_episodes'] == 510:
            rewards = experiment['rewards']
            for i in range(len(rewards)):
                for j in range(len(rewards[i])):
                    rewards[i][j] = rewards[i][j]/10

            print("[WARNING]: Old experiment file.")
            objects.remove(None)
            idx_offset -= 1
            continue

        experiment['environment'] = get_env_json()

        objects[idx + idx_offset] = experiment

    return objects


def import_experiment(fn):
    with open(fn, 'r') as jsonf:
        return json.load(jsonf)


def get_all_filenames():
    path = "../data-new/"
    return [path + file for file in os.listdir(path) if ".json" in file]


if __name__ == '__main__':
    print("[INFO]: Loading experiments.")
    ct = time.time()
    experiments = load_all_files()
    print("[INFO]: Finished loading in {}s.".format(time.time() - ct))

    obj = {}
    obj["experiments"] = experiments

    with open("../experiments-new-test.pickle", "wb") as f:
        pickle.dump(experiments, f, protocol=pickle.HIGHEST_PROTOCOL)


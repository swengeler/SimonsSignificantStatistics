import pickle
import os
import json
import time
from json import JSONDecodeError


def load_all_files(load_few=False, batch_size=0) -> list:
    filenames = get_all_filenames()
    if load_few and len(filenames) > 10:
        filenames = filenames[1:100]
    objects = [None for _ in range(0, len(filenames))]

    for idx, file in enumerate(filenames):
        if (idx + 1) % 10 == 0:
            print("[INFO]: Loaded {} files.".format(idx + 1))

        try:
            experiment = import_experiment(file)
        except JSONDecodeError:
            print("[WARNING]: Could not load file.")
            continue
        objects[idx] = experiment

    return objects


def import_experiment(fn):
    with open(fn, 'r') as jsonf:
        return json.load(jsonf)


def get_all_filenames():
    path = "../data-long/"
    return [path + file for file in os.listdir(path) if ".json" in file]


if __name__ == '__main__':
    print("[INFO]: Loading experiments.")
    ct = time.time()
    experiments = load_all_files(True)
    print("[INFO]: Finished loading in {}s.".format(time.time() - ct))

    obj = {}
    obj["experiments"] = experiments

    with open("../experiments-long-0.pickle", "wb") as f:
        pickle.dump(experiments, f, protocol=pickle.HIGHEST_PROTOCOL)


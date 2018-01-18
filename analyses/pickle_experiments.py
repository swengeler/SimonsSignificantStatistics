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
            print("[INFO]: Loaded {} files.".format(idx + 1))

        experiment = import_experiment(file)
        objects[idx] = experiment

    return objects


def import_experiment(fn):
    with open(fn, 'r') as jsonf:
        return json.load(jsonf)


def get_all_filenames():
    path = "../../data-new/"
    return [path + file for file in os.listdir(path) if ".json" in file]


if __name__ == '__main__':
    print("[INFO]: Loading experiments.")
    ct = time.time()
    experiments = load_all_files()
    print("[INFO]: Finished loading in {}s.".format(time.time() - ct))

    obj = {}
    obj["experiments"] = experiments

    with open("../../experiments-new.pickle", "wb") as f:
        pickle.dump(experiments, f, protocol=pickle.HIGHEST_PROTOCOL)

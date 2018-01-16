import time
import pickle
import matplotlib.pyplot as plot

def get_parameter_name(experiments):
    experiment = experiments[0]
    parameters = experiment["parameters"]
    names = list(parameters.keys())
    return names


def get_parameter(experiments, parameter_name):
    values = []
    for experiment in experiments:
        parameters = experiment["parameters"]
        value = parameters[parameter_name]
        values.append(value)
    return values

def create_histogram(values, parameter_name):
    plot.hist(values, bins='auto')
    plot.title("histogram of {}".format(parameter_name))
    plot.show()


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

    names = get_parameter_name(experiments)
    for name in names:
        create_histogram(get_parameter(experiments, name), name)
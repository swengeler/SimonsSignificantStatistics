import numpy as np
from load_files import *

if __name__ == "__main__":
    directory_name = "../test-analysis/23-01-2018_02-16-10_05a2090c-06e1-4d55-9d94-c188b6607b6e/"
    for i in range(0, 5000, 50):
        model = load_model(directory_name, i, 81)
        print(np.argmax(model.predict(np.array((0, 0, np.pi, 0, np.pi, 0)).reshape([1, 6]))))
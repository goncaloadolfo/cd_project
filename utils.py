'''
Auxiliar functions to other modules.
'''

# libs
import numpy as np


def print_return_variable(prefix, value):
    print(prefix + str(value))
    return value


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

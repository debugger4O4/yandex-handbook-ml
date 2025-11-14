import numpy as np

def construct_matrix(first_array, second_array):
    return np.dstack([first_array, second_array]).reshape([-1,2])
import random
import numpy as np

class sampling_func():

    def choice(options: list):
        return random.choice(options)

    def uniform(bounds: list):
        if len(bounds) != 2:
            raise Exception("Please provide a upper and lower bound for uniform.")
        return np.random.uniform(bounds[0], bounds[1])

    def validate_value_range(function, arguments):
        assert(len(arguments.keys()) == 1)
        assert(len(function.__code__.co_varnames) == 1)
        expected_value_range = function.__code__.co_varnames[0]
        actual_value_range = next(iter(arguments.keys()))
        assert(expected_value_range == actual_value_range)
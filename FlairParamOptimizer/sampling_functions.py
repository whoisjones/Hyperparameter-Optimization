import random
import numpy as np

class sampling_func():

    def choice(options: list):
        return random.choice(options)

    def uniform(bounds: list):
        if len(bounds) != 2:
            raise Exception("Please provide a upper and lower bound for uniform.")
        return np.random.uniform(bounds[0], bounds[1])

    @classmethod
    def validate_value_range(cls, function, arguments: dict):
        assert(len(arguments.keys()) == 1)
        assert(len(function.__code__.co_varnames) == 1)
        expected_function_parameter = function.__code__.co_varnames[0]
        actual_function_parameter = next(iter(arguments.keys()))
        assert(expected_function_parameter == actual_function_parameter)

    @classmethod
    def extract_value_range(cls, function, general_argument: dict):
        function_specific_argument = function.__code__.co_varnames[0]
        return general_argument.get(function_specific_argument)
from enum import Enum

from .parameter_groups import EMBEDDING_SPECIFIC_PARAMETERS
from .sampling_functions import sampling_func

class ParameterCollection():

    def __init__(self):
        pass

    def add(self, parameter_name: str, sampling_function: sampling_func, value_range: list, embedding_key : str  = "GeneralParameters"):
        parameter = {"sampling_function": sampling_function,
                     "value_range": value_range}
        if hasattr(self, embedding_key):
            self._append_to_existing_embedding_key(embedding_key, parameter_name, parameter)
        else:
            self._create_new_embedding_key(embedding_key)
            self._append_to_existing_embedding_key(embedding_key, parameter_name, parameter)

    def _create_new_embedding_key(self, parameter_name: str):
        setattr(self, parameter_name, {})

    def _append_to_existing_embedding_key(self, embedding_key: str, parameter_name: str, parameter: dict):
        getattr(self, embedding_key)[parameter_name] = parameter


class Configuration():

    def __init__(self):
        pass
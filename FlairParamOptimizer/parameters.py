import itertools
from .sampling_functions import sampling_func
from .parameter_listings.parameter_groups import EMBEDDINGS

class ParameterStorage():

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


class TrainingConfigurations():

    def __init__(self, parameter_storage: ParameterStorage):
        self.parameter_storage = parameter_storage

    def make_configurations(self):
        available_embeddings =  self.parameter_storage.__dict__.keys() & EMBEDDINGS
        for embedding_type in available_embeddings:
            embedding_parameters = getattr(self.parameter_storage, embedding_type)
            keys = embedding_parameters.keys()

    def _get_cartesian_product(self, parameters: tuple):
        parameter_keys, parameter_options = parameters
        all_configurations = []
        for configuration in itertools.product(*parameter_options):
            all_configurations.append(dict(zip(parameter_keys, configuration)))
        return all_configurations
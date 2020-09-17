import itertools
from .parameter_listings.parameter_groups import EMBEDDINGS
from flair.embeddings import DocumentRNNEmbeddings, TransformerDocumentEmbeddings, DocumentPoolEmbeddings

class ParameterStorage():

    def __init__(self):
        pass

    def add(self, parameter_name: str, value_range: list, embedding_key : str  = "GeneralParameters"):
        if hasattr(self, embedding_key):
            self._append_to_existing_embedding_key(embedding_key, parameter_name, value_range)
        else:
            self._create_new_embedding_key(embedding_key)
            self._append_to_existing_embedding_key(embedding_key, parameter_name, value_range)

    def _create_new_embedding_key(self, parameter_name: str):
        if parameter_name in EMBEDDINGS:
            setattr(self, parameter_name, {"document_embeddings":[eval(parameter_name)]})
        else:
            setattr(self, parameter_name, {})

    def _append_to_existing_embedding_key(self, embedding_key: str, parameter_name: str, parameter: dict):
        getattr(self, embedding_key)[parameter_name] = parameter


class TrainingConfigurations():

    def __init__(self):
        self.configurations = []

    def make_grid_configurations(self, parameter_storage: ParameterStorage):
        parameters_tuple = self._get_parameters_tuple(parameter_storage)
        self._make_cartesian_product(parameters_tuple)

    def _get_parameters_tuple(self, parameter_storage: ParameterStorage):
        embedding_specific_keys_in_parameter_storage, general_parameters = self._get_parameter_keys(parameter_storage)
        if embedding_specific_keys_in_parameter_storage:
            parameter_tuples = self._make_embedding_specific_tuples(embedding_specific_keys_in_parameter_storage,
                                                                    general_parameters,
                                                                    parameter_storage)
        else:
            parameter_tuples = self._make_tuples(general_parameters, parameter_storage)
        return parameter_tuples

    def _get_parameter_keys(self, parameter_storage: ParameterStorage):
        embedding_specific_keys_in_parameter_storage = parameter_storage.__dict__.keys() & EMBEDDINGS
        general_parameter_keys = parameter_storage.__dict__.keys() - EMBEDDINGS
        return embedding_specific_keys_in_parameter_storage, general_parameter_keys

    def _make_embedding_specific_tuples(self, embedding_keys: set, general_keys: set, parameter_storage: ParameterStorage):
        tuples = []
        for embedding_key, general_key in itertools.product(embedding_keys, general_keys):
            embedding_specific_parameters = getattr(parameter_storage, embedding_key)
            general_parameters = getattr(parameter_storage, general_key)
            complete_parameters_per_embedding = {**embedding_specific_parameters, **general_parameters}
            tuples.append(complete_parameters_per_embedding)
        return tuples

    def _make_tuples(self, general_keys: set, parameter_storage: ParameterStorage):
        tuples = []
        general_key = general_keys.pop()
        general_parameters = getattr(parameter_storage, general_key)
        tuples.append(general_parameters)
        return tuples

    def _make_cartesian_product(self, parametersList: list):
        for parameters in parametersList:
            keys, values = zip(*parameters.items())
            training_configurations = itertools.product(*values)
            for configuration in training_configurations:
                self.configurations.append(dict(zip(keys, configuration)))

    def _make_evolutionary_configurations(self, parameter_storage: ParameterStorage):
        pass
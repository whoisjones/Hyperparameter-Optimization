import itertools
from abc import abstractmethod
import random
import numpy as np
from random import randrange

from FlairParamOptimizer.search_spaces import SearchSpace

"""
The ParamOptimizer object acts as the optimizer instance in flair's hyperparameter optimization.
We are currently supporting three types of optimization:
    GeneticOptimizer        using evolutionary algorithms
    GridSearchOptimizer     standard grid search optimization
    RandomSearchOptimizer   random order grid search optimization
    
The optimizers take an search space object as input, please see the documentation of search spaces for further
information.

Depending on the optimizer type the respective configurations of your hyperparameter optimization will be calculated.

Apart from optimizer specific functions, if you want to add a new optimization procedure, following functions
have to be overwritten:
    _get_configurations()                           wrapper function which returns a list of all configurations.
                                                    depending on whether embedding specific parameters are set,
                                                    call respective functions.
    _get_embedding_specific_configurations()        If we have embedding specific parameters.
    _get_standard_configurations()                  If we don't have embedding specific parameters.
    _get_configurations_for_single_embedding()      returns a list of configurations for one embedding.
"""


class ParamOptimizer(object):

    def __init__(self, search_space: SearchSpace):
        self.results = {}
        self.document_embedding_specific_parameters = search_space.document_embedding_specific_parameters
        search_space._check_mandatory_parameters_are_set(optimizer_type=self.__class__.__name__)

    @abstractmethod
    def _make_configurations(self,
                             parameter: dict,
                             shuffle: bool,
                             embedding_specific: bool):
        pass

    @abstractmethod
    def _make_embedding_specific_configurations(self, parameters: dict):
        pass

    @abstractmethod
    def _make_configurations_for_single_embedding(self, parameters: dict):
        pass


class GridSearchOptimizer(ParamOptimizer):

    def __init__(self, search_space: SearchSpace, shuffle: bool = False):
        super().__init__(search_space)
        search_space.configurations = self._make_configurations(parameters=search_space.parameters, shuffle=shuffle,
                                                                embedding_specific=search_space.document_embedding_specific_parameters)

    def _make_configurations(self,
                             parameters: dict,
                             shuffle: bool,
                             embedding_specific: bool):
        if embedding_specific:
            configurations = self._make_embedding_specific_configurations(parameters)
        else:
            configurations = self._make_configurations_for_single_embedding(parameters)

        if shuffle:
            random.shuffle(configurations)
        return configurations

    def _make_embedding_specific_configurations(self, parameters: dict):
        all_configurations = []
        for document_embedding, embedding_parameters in parameters.items():
            all_configurations.append(self._make_configurations_for_single_embedding(embedding_parameters))
        all_configurations = self._flatten_grid(all_configurations)
        return all_configurations

    def _make_configurations_for_single_embedding(self, parameters: dict):
        parameters_chosen_from_options, uniformly_sampled_parameters = self._split_up_parameters_based_on_sampling(parameters)
        all_configurations = self._get_cartesian_product(parameters_chosen_from_options)
        # Since dicts are not sorted, uniformly sampled configurations have to be added later
        if uniformly_sampled_parameters:
            all_configurations = self._add_uniformly_sampled_parameters(uniformly_sampled_parameters,
                                                                        all_configurations)
        return all_configurations

    def _split_up_parameters_based_on_sampling(self, parameters: dict):
        parameter_options = []
        parameter_keys = []
        uniformly_sampled_parameters = {}
        for parameter_key in parameters.keys():
            config = parameters.get(parameter_key)
            sampling_type = config['method'].__code__.co_varnames[0]
            if sampling_type == "options":
                parameter_options.append(config['options'])
                parameter_keys.append(parameter_key)
            else:
                uniformly_sampled_parameters[parameter_key] = config
        parameters_chosen_from_options = (parameter_keys, parameter_options)
        return parameters_chosen_from_options, uniformly_sampled_parameters

    def _get_cartesian_product(self, parameters: tuple):
        parameter_keys, parameter_options = parameters
        all_configurations = []
        for configuration in itertools.product(*parameter_options):
            all_configurations.append(dict(zip(parameter_keys, configuration)))
        return all_configurations

    def _add_uniformly_sampled_parameters(self, bounds: dict, all_configurations: list):
        for item in all_configurations:
            for parameter_name, configuration in bounds.items():
                func = configuration['method']
                item[parameter_name] = func(configuration['bounds'])
        return all_configurations

    def _flatten_grid(self, all_configurations: list):
        return [item for config in all_configurations for item in config]


class RandomSearchOptimizer(GridSearchOptimizer):

    def __init__(self, search_space: SearchSpace):
        super().__init__(search_space, shuffle=True)

class GeneticOptimizer(ParamOptimizer):

    def __init__(
            self,
            search_space: SearchSpace,
            population_size: int = 8,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.01,
    ):
        super().__init__(search_space)
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        search_space.configurations = self._make_configurations(parameters=search_space.parameters,
                                                                embedding_specific=search_space.document_embedding_specific_parameters)
        search_space._set_additional_budget_parameters(population_size = population_size)

    def _make_configurations(self, parameters: dict, embedding_specific: str):
        if embedding_specific:
            configurations = self._make_embedding_specific_configurations(parameters)
        else:
            configurations = self._make_configurations_for_single_embedding(parameters)
        random.shuffle(configurations)
        return configurations

    def _make_embedding_specific_configurations(self, parameters: dict):
        amount_individuals_per_embedding = self._get_equal_amount_of_individuals_per_embedding(len(parameters))
        configurations = self._make_configurations_for_each_embedding(parameters, amount_individuals_per_embedding)
        return configurations

    def _get_equal_amount_of_individuals_per_embedding(self, length_of_different_embeddings: int) -> list:
        individuals_per_embedding = [self.population_size // length_of_different_embeddings +
                                     (1 if x < self.population_size % length_of_different_embeddings else 0)
                                     for x in range(length_of_different_embeddings)]
        return individuals_per_embedding

    def _make_configurations_for_each_embedding(self, parameters: dict, individuals_per_embedding: list):
        configurations = []
        for (embedding_type, parameters_for_embedding), individuals_single_embedding in zip(parameters.items(), individuals_per_embedding):
            for _ in range(individuals_single_embedding):
                configurations_for_single_embedding = self._make_configurations_for_single_embedding(parameters_for_embedding)
                configurations.append(configurations_for_single_embedding)
        return configurations

    def _make_configurations_for_single_embedding(self, parameters: dict):
        configuration = {}
        for parameter_name, value_range in parameters.items():
            parameter_value = self._sample_parameter_from(**value_range)
            configuration[parameter_name] = parameter_value
        return configuration

    def _sample_parameter_from(self, **kwargs):
        sampling_function = kwargs.get('method')
        accepted_attribute_from_function = sampling_function.__code__.co_varnames[0]
        value_range = kwargs.get(accepted_attribute_from_function)
        training_parameter = sampling_function(value_range)
        return training_parameter

    def _evolve_required(self, current_run: int):
        if current_run % (self.population_size) == (self.population_size - 1):
            return True
        else:
            return False

    def _evolve(self, search_space: SearchSpace, current_results: dict):
        parent_population = self._get_parent_population(search_space.configurations)
        selected_population = self._select(search_space.configurations, current_results)
        for child in selected_population:
            child = self._crossover(child, parent_population)
            child = self._mutate(child)
            self.configurations.append(child)

    def _get_parent_population(self, configurations: list) -> dict:
        parent_population = {}
        for embedding in configurations[-self.population_size:]:
            embedding_key = self._get_embedding_key(embedding)
            embedding_value = embedding
            if embedding_key in parent_population:
                parent_population[embedding_key].append(embedding_value)
            else:
                parent_population[embedding_key] = [embedding_value]
        return parent_population

    def _get_embedding_key(self, embedding: dict):
        if self.document_embedding_specific_parameters == True:
            embedding_key = embedding['document_embeddings'].__name__
        else:
            embedding_key = "universal_embeddings"
        return embedding_key

    def _select(self, configurations: list, current_results: dict) -> np.array:
        evo_probabilities = self._get_fitness(current_results)
        return np.random.choice(configurations, size=self.population_size, replace=True, p=evo_probabilities)

    def _get_fitness(self, results: dict):
        fitness = np.asarray([individual['result'] for individual in results.values()])
        probabilities = fitness / (sum([individual['result'] for individual in results.values()]))
        return probabilities

    def _crossover(self, child: dict, parent_population: dict):
        child_type = self._get_embedding_key(child)
        population_size = len(parent_population[child_type])
        DNA_size = len(child)
        if np.random.rand() < self.cross_rate:
            i_ = randrange(population_size)  # select another individual from pop
            parent = parent_population[child_type][i_]
            cross_points = np.random.randint(0, 2, DNA_size).astype(np.bool)  # choose crossover points
            for (parameter, value), replace in zip(child.items(), cross_points):
                if replace:
                    child[parameter] = parent[parameter]  # mating and produce one child
        return child

    def _mutate(self, child: dict):
        child_type = self._get_embedding_key(child)
        for parameter in child.keys():
            if np.random.rand() < self.mutation_rate:
                func = self.all_configurations[child_type][parameter]['method']
                if self.all_configurations[child_type][parameter].get("options") is not None:
                    child[parameter] = func(self.all_configurations[child_type][parameter]['options'])
                elif self.all_configurations[child_type][parameter].get("bounds") is not None:
                    child[parameter] = func(self.all_configurations[child_type][parameter]['bounds'])
        return child
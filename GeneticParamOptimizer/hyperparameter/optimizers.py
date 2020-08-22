import itertools
from abc import abstractmethod
import random
import logging as log
import numpy as np
from random import randrange
from .parameters import *

from GeneticParamOptimizer.hyperparameter.search_spaces import SearchSpace

class ParamOptimizer():
    """
    Parent class for all hyperparameter optimizers.

    Attributes:
        search_space: Search Space object
    """

    def __init__(
            self,
            search_space: SearchSpace,
    ):
        """
        Parent class for optimizers which stores budget and parameters from search space object

        :rtype: object
        :param search_space: the search space from which to get parameters and budget from
        """
        search_space._check_mandatory_parameters_are_set()

        self.type = search_space.__class__.__name__
        self.budget = search_space.budget
        self.parameters = search_space.parameters
        self.optimization_value = search_space.optimization_value
        self.evaluation_metric = search_space.evaluation_metric
        self.max_epochs_training  = search_space.max_epochs_training

        #TODO rename
        @abstractmethod
        #Wrapper function to get configurations
        def _get_configurations(self):
            pass

        @abstractmethod
        # If there are document embedding specific parameters
        def _get_embedding_specific_configurations(self):
            pass

        @abstractmethod
        # For standard parameters without document embeddings
        def _get_standard_configurations(self):
            pass

        @abstractmethod
        # returns all configurations for one embedding (either 1 document embedding or all parameter)
        def _get_configurations_for_single_embedding(self):
            pass


class GridSearchOptimizer(ParamOptimizer):
    """A class for grid search hyperparameter optimization."""

    def __init__(
            self,
            search_space: SearchSpace,
            shuffle: bool = False,
    ):
        """
        Creates a grid search object with all possible configurations from search space (cartesian product)

        :rtype: object
        :param search_space: the search space from which to get parameters and budget from
        :param shuffled: if true, returns a shuffled list of parameter configuration.
        """
        super().__init__(
            search_space
        )

        self.configurations = self._get_configurations(
                                parameters=search_space.parameters,
                                shuffle=shuffle,
                                embedding_specific=search_space.document_embedding_specific)

    def _get_configurations(
            self,
            parameters : dict,
            shuffle : bool,
            embedding_specific: bool
    ):
        """
        Wrapper function which does the cartesian product of provided configurations depending on search space type

        :param shuffled: if true, a shuffled list of configurations is returned
        :param parameters: a dict which contains parameters as keywords with its possible configurations as values
        :return: a list of parameters configuration
        :rtype: list
        """

        if embedding_specific:
            configurations = self._get_embedding_specific_configurations(parameters, shuffle)
        else:
            configurations = self._get_standard_configurations(parameters, shuffle)

        return configurations

    def _get_embedding_specific_configurations(self,
                                     parameters: dict,
                                     shuffle: bool):
        """
        Returns all configurations and check embedding specific parameters,
        i.e. for the text classification downstream task
        :param parameters: Dict containing all parameters as key value pairs
        :param shuffled: Bool - if true, shuffle the grid
        :return: list of all configurations
        """

        all_configurations = []
        for document_embedding, embedding_parameters in parameters.items():
            all_configurations.append(self._get_configurations_for_single_embedding(embedding_parameters))

        all_configurations = self._flatten_grid(all_configurations)

        if shuffle:
            random.shuffle(all_configurations)

        return all_configurations

    def _get_standard_configurations(self, parameters: dict, shuffle: bool):
        """
        Returns all configurations for the sequence labeling downstream task
        :param parameters: Dict containing all parameters as key value pairs
        :param shuffled: Bool - if true, shuffle the grid
        :return: list of all configurations
        """

        all_configurations = self._get_configurations_for_single_embedding(parameters)

        if shuffle:
            random.shuffle(all_configurations)

        return all_configurations

    def _get_configurations_for_single_embedding(self, parameters: dict):
        """
        Returns the cartesian product for all configurations provided. Adds uniformly sampled data in the second step.
        :param parameters:
        :return:
        """

        option_parameters, uniformly_sampled_parameters = self._split_up_configurations(parameters)

        all_configurations = self._get_cartesian_product(option_parameters)

        # Since dicts are not sorted, uniformly sampled configurations have to be added later
        if uniformly_sampled_parameters:
            all_configurations = self._add_uniformly_sampled_parameters(uniformly_sampled_parameters, all_configurations)

        return all_configurations

    def _split_up_configurations(self, parameters: dict):
        """
        Splits the parameters based on whether to choose from options or take a uniform sample from a distribution.
        :param parameters: Dict containing the parameters
        :return: parameters from options as tuple, uniformly sampled parameters as dict
        """
        parameter_options = []
        parameter_keys = []
        uniformly_sampled_parameters = {}

        for parameter_name, configuration in parameters.items():
            try:
                parameter_options.append(configuration['options'])
                parameter_keys.append(parameter_name)
            except:
                uniformly_sampled_parameters[parameter_name] = configuration

        return (parameter_keys, parameter_options), uniformly_sampled_parameters

    def _get_cartesian_product(self, parameters: tuple):
        """
        Returns the cartesian product of provided parameters. Takes two list (keys, values) in form of a tuple
        as input.
        :param parameters: tuple (list, list) containing keys and values of parameters
        :return: list of all configurations
        """
        parameter_keys, parameter_options = parameters
        all_configurations = []
        for configuration in itertools.product(*parameter_options):
            all_configurations.append(dict(zip(parameter_keys, configuration)))

        return all_configurations

    def _add_uniformly_sampled_parameters(self, bounds: dict, all_configurations: list):
        """
        Adds to each configuration a uniform sample of respective parameters.
        :param bounds: dict containing the parameters which should be uniformly sampled
        :param all_configurations: list of all configurations to which append a uniform sample parameter
        :return: list of all configurations with a uniformly sampled parameter
        """
        for item in all_configurations:
            for parameter_name, configuration in bounds.items():
                func = configuration['method']
                item[parameter_name] = func(configuration['bounds'])

        return all_configurations

    def _flatten_grid(self, all_configurations: list):
        """
        Flattens the list of all configurations for further processing.
        :param all_configurations: list of all configurations
        :return: flat list of all configurations
        """
        return [item for subgrid in all_configurations for item in subgrid]

class RandomSearchOptimizer(GridSearchOptimizer):
    """A class for random search hyperparameter optimization"""

    def __init__(
            self,
            search_space: SearchSpace,
    ):
        """
        Initializes a RandomSearchOptimizer object

        :param search_space: the search space from which to get parameters and budget from
        :rtype: object
        """
        super().__init__(
            search_space,
            shuffle=True
        )


class GeneticOptimizer(ParamOptimizer):
    """A class for hyperparameter optimization using evolutionary algorithms."""

    def __init__(
            self,
            search_space: SearchSpace,
            population_size: int = 32,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.01,
    ):
        """
        Initializes a GeneticOptimizer for hyperparameter optimization using evolutionary algorithms

        :param search_space: the search space from which to get parameters and budget from
        :param population_size: number of configurations per generation
        :param cross_rate: percentage of crossover during recombination of configurations
        :param mutation_rate: probability of mutation of configurations
        :rtype: object
        """
        super().__init__(
            search_space
        )

        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.all_parameters = search_space.parameters

        self.search_grid = self._get_search_grid(search_space.parameters, population_size, type=self.type)

    def _get_search_grid(
            self,
            parameters : dict,
            population_size : int,
            type: str,
    ):
        """
        returns a generation of parameter configurations

        :param parameters: a dict which contains parameters as keywords with its possible configurations as values
        :return: a list of configurations
        :rtype: list
        """
        if type == "TextClassifierSearchSpace":
            return self._get_TC_search_grid(parameters, population_size)
        elif type == "SequenceTaggerSearchSpace":
            return self._get_ST_search_grid(parameters, population_size)
        else:
            raise Exception()

    def _get_TC_search_grid(self, parameters, population_size):

        grid = []

        div = len(parameters)
        nested_population_size = [population_size // div + (1 if x < population_size % div else 0)  for x in range (div)]

        for (nested_key, nested_parameters), individuals_per_group in zip(parameters.items(), nested_population_size):
            grid.append(self._get_individuals(nested_parameters, population_size=individuals_per_group))
        flat_grid = [item for subgrid in grid for item in subgrid]

        shuffle(flat_grid)

        return flat_grid

    def _get_ST_search_grid(self, parameters, population_size):

        return self._get_individuals(parameters, population_size)

    def _get_individuals(self, parameters, population_size):
        individuals = []
        for idx in range(population_size):
            individual = {}
            for parameter_name, configuration in parameters.items():
                parameter_value = self.get_parameter_from(**configuration)
                individual[parameter_name] = parameter_value
            individuals.append(individual)

        return individuals

    def get_parameter_from(self, **kwargs):
        """
        Helper function to extract either a choice from list or a parameter value from a uniform distribution

        :param kwargs: a tuple of a function and values / bounds
        :return: float or int depending on function provided
        """
        func = kwargs.get('method')
        if kwargs.get('options') != None:
            parameter = func(kwargs.get('options'))
        elif kwargs.get('bounds') != None:
            parameter = func(kwargs.get('bounds'))
        else:
            raise Exception("Please provide either bounds or options as arguments to the search space depending on your function.")
        return parameter

    def _evolve(self, current_population: list):
        parent_population = self._get_formatted_population(current_population)
        selected_population = self._select(current_population)
        new_generation = []
        for child in selected_population:
            child = self._crossover(child, parent_population)
            child = self._mutate(child)
            new_generation.append(child)
        return new_generation

    def _get_formatted_population(self, current_population: list):
        formatted = {}
        for embedding in current_population:
            embedding_key = embedding['params']['document_embeddings'].__name__
            embedding_value = embedding['params']
            if embedding_key in formatted:
                formatted[embedding_key].append(embedding_value)
            else:
                formatted[embedding_key] = [embedding_value]
        return formatted


    def _select(self, current_population: list):
        evo_probabilities = self._get_fitness(current_population)
        return np.random.choice(current_population, size=self.population_size, replace=True, p=evo_probabilities)


    def _get_fitness(self, current_population: list):
        fitness = [individual['result'] for individual in current_population]
        probabilities = fitness / (sum([x['result'] for x in current_population]))
        return probabilities


    def _crossover(self, child: dict, parent_population: dict):
        child_type = child['params']['document_embeddings'].__name__
        population_size = len(parent_population[child_type])
        DNA_size = len(child['params'])
        if np.random.rand() < self.cross_rate:
            i_ = randrange(population_size)  # select another individual from pop
            parent = parent_population[child_type][i_]
            cross_points = np.random.randint(0, 2, DNA_size).astype(np.bool)  # choose crossover points
            for (parameter, value), replace in zip(child['params'].items(), cross_points):
                if replace:
                    child['params'][parameter] = parent[parameter] # mating and produce one child
        return child

    def _mutate(self, child: dict):
        child_type = child['params']['document_embeddings'].__name__
        for parameter in child['params']:
            if np.random.rand() < self.mutation_rate:
                func = self.all_parameters[child_type][parameter]['method']
                child[parameter] = func(self.all_parameters[child_type][parameter]['options'])
        return child


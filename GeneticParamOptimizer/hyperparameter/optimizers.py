import itertools
from abc import abstractmethod
from random import shuffle
import logging as log
import numpy as np
from .parameters import *

from GeneticParamOptimizer.hyperparameter.search_spaces import SearchSpace

class ParamOptimizer():
    """Parent class for all hyperparameter optimizers."""

    def __init__(
            self,
            search_space: SearchSpace,
    ):
        """
        Parent class for optimizers which stores budget and parameters from search space object

        :rtype: object
        :param search_space: the search space from which to get parameters and budget from
        """
        if not all([search_space.budget, search_space.parameters, search_space.optimization_value, search_space.evaluation_metric]):
            raise Exception("Please provide a budget, parameters, a optimization value and a evaluation metric for an optimizer.")

        log.info('Initializing optimizer...')

        self.type = search_space.__class__.__name__
        self.budget = search_space.budget
        self.parameters = search_space.parameters
        self.optimization_value = search_space.optimization_value
        self.evaluation_metric = search_space.evaluation_metric
        self.max_epochs_training  = search_space.max_epochs_training

        @abstractmethod
        def _get_search_grid(self):
            pass

        @abstractmethod
        def _get_TC_search_grid(self):
            pass

        @abstractmethod
        def _get_ST_search_grid(self):
            pass

        @abstractmethod
        def _get_individuals(self):
            pass


class GridSearchOptimizer(ParamOptimizer):
    """A class for grid search hyperparameter optimization."""

    def __init__(
            self,
            search_space: SearchSpace,
            shuffled: bool = False,
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

        self.search_grid = self._get_search_grid(search_space.parameters, shuffled, type=self.type)

    def _get_search_grid(
            self,
            parameters : dict,
            shuffled : bool,
            type : str,
    ):
        """
        Does the cartesian product of provided configurations.

        :param shuffled: if true, a shuffled list of configurations is returned
        :param parameters: a dict which contains parameters as keywords with its possible configurations as values
        :return: a list of parameters configuration
        :rtype: list
        """

        if type == "TextClassifierSearchSpace":
            search_grid = self._get_TC_search_grid(parameters, shuffled)
        elif type == "SequenceTaggerSearchSpace":
            search_grid = self._get_ST_search_grid(parameters, shuffled)

        return search_grid

    def _get_TC_search_grid(self, parameters, shuffled):

        grid = []

        for nested_key, nested_parameters in parameters.items():
            grid.append(self._get_individuals(nested_parameters, shuffled))
        flat_grid = [item for subgrid in grid for item in subgrid]

        if shuffled:
            shuffle(flat_grid)

        return flat_grid

    def _get_ST_search_grid(self, parameters, shuffled):

        return self._get_individuals(parameters, shuffled)

    def _get_individuals(self, parameters, shuffled):
        parameter_options = []
        parameter_keys = []

        bounds = {}  # store bounds for later since they cannot be part of cartesian product

        # filter bounds
        for parameter_name, configuration in parameters.items():
            try:
                parameter_options.append(configuration['options'])
                parameter_keys.append(parameter_name)
            except:
                bounds[parameter_name] = configuration

        # get cartesian product from all choice options
        grid = []
        for configuration in itertools.product(*parameter_options):
            grid.append(dict(zip(parameter_keys, configuration)))

        # if bounds are given in search_space, add them here
        if bounds:
            for item in grid:
                for parameter_name, configuration in bounds.items():
                    func = configuration['method']
                    item[parameter_name] = func(configuration['bounds'])

        if shuffled:
            shuffle(grid)

        return grid


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
            shuffled=True
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
        for child in selected_population:
            child = self._crossover(child, selected_population)
            child = self._mutate(child)
            parent[:] = child

    def _get_formatted_population(self, current_population):
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
        print("moin")


    def _crossover(self, child, parent_population):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.population_size, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # choose crossover points
            child[cross_points] = parent_population[i_, cross_points]  # mating and produce one child
        return child
import itertools
from random import shuffle

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
        self.search_space = search_space
        self.budget = search_space.budget
        self.parameters = search_space.parameters
        self.optimization_value = search_space.optimization_value
        self.evaluation_metric = search_space.evaluation_metric


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

        self.search_grid = self.get_search_grid(search_space.parameters, shuffled)

    def get_search_grid(
            self,
            parameters : dict,
            shuffled : bool
    ):
        """
        Does the cartesian product of provided configurations.

        :param shuffled: if true, a shuffled list of configurations is returned
        :param parameters: a dict which contains parameters as keywords with its possible configurations as values
        :return: a list of parameters configuration
        :rtype: list
        """

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

        self.DNA_size = len(search_space.parameters)
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.search_grid = self.get_search_grid(search_space.parameters, population_size)

    def get_search_grid(
            self,
            parameters : dict,
            population_size : int,
    ):
        """
        returns a generation of parameter configurations

        :param parameters: a dict which contains parameters as keywords with its possible configurations as values
        :param population_size: the size of individual configurations per generation
        :return: a list of configurations
        :rtype: list
        """
        search_grid = []
        for idx in range(population_size):
            individual = {}
            for parameter_name, configuration in parameters.items():
                parameter_value = self.get_parameter_from(**configuration)
                individual[parameter_name] = parameter_value
            search_grid.append(individual)

        return search_grid

    def get_parameter_from(self, **kwargs):
        """
        Helper function to extract either a choice from list or a parameter value from a uniform distribution

        :param kwargs: a tuple of a function and values / bounds
        :return: float or int depending on function provided
        """
        func = kwargs.get('method')
        if kwargs.get('options') != None:
            parameter = func(kwargs.get('options'))
        else:
            parameter = func(kwargs.get('bounds'))
        return parameter

    def get_fitness(self):
        pass

    def setup_new_generation(self):
        pass

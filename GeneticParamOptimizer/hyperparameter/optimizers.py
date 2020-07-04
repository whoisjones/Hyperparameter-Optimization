import itertools
from GeneticParamOptimizer.hyperparameter.multiprocessor import multiprocess
from GeneticParamOptimizer.hyperparameter.search_spaces import SearchSpace

class ParamOptimizer():
    def __init__(
            self,
            search_space: SearchSpace,
    ):
        self.search_space = search_space
        self.budget = search_space.budget
        self.parameters = search_space.parameters

class GridSearchOptimizer(ParamOptimizer):

    def __init__(
            self,
            search_space: SearchSpace,
            ):
        super().__init__(
            search_space
        )

        self.search_grid = self.get_search_grid(**search_space.parameters)

    def get_search_grid(self, **kwargs):
        options = [] # used to store all combinations for cartesian product
        keys = [] # store all keys associated to options
        bounds = {} # store bounds for later since they cannot be part of cartesian product

        # split search space in uniform and options
        for parameter_name, configuration in kwargs.items():
            try:
                options.append(configuration['options'])
                keys.append(parameter_name)
            except:
                bounds[parameter_name] = configuration

        #calculate grid from all choice options
        grid = []
        for instance in itertools.product(*options):
            grid.append(dict(zip(keys, instance)))

        # now add to each item in complete grid a uniform sample for all left parameters
        if bounds:
            for item in grid:
                for parameter_name, configuration in bounds.items():
                    func = configuration['method']
                    item[parameter_name] = func(configuration['bounds'])

        return grid


class GeneticOptimizer(ParamOptimizer):

    def __init__(
            self,
            search_space: SearchSpace,
            population_size: int = 32,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.01,
            ):
        super().__init__(
            search_space
        )

        self.DNA_size = len(search_space.parameters)
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.population = self.get_population(search_space.parameters, population_size)

    def get_population(self, parameters, population_size):
        population = []
        for idx in range(population_size):
            individual = {}
            for parameter_name, configuration in parameters.items():
                parameter_value = self.get_parameter_from(**configuration)
                individual[parameter_name] = parameter_value
            population.append(individual)

        return population

    def get_parameter_from(self, **kwargs):
        func = kwargs.get('method')
        if kwargs.get('options') != None:
            parameter = func(kwargs.get('options'))
        else:
            parameter = func(kwargs.get('bounds'))
        return parameter

    def run(self):
        multiprocess(self)

    def get_fitness(self):
        pass

    def setup_new_generation(self):
        pass
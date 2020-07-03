import itertools
import random
from GeneticParamOptimizer.hyperparameter.multiprocessor import multiprocess
from GeneticParamOptimizer.hyperparameter import SearchSpace
from flair.data import Corpus

class ParamOptimizer():
    def __init__(
            self,
            search_space: SearchSpace,
    ):
        self.search_space = search_space
        self.budget = search_space.budget
        self.parameters = search_space.parameters

class GridSearchOptimizer(ParamOptimizer):
    def __init__(self):
        pass

class RandomSearchOptimizer(ParamOptimizer):
    def __init__(self):
        pass

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
        self.population = self.get_initial_population(search_space.parameters, population_size)

    def get_initial_population(self, parameters, population_size):
        population = self.get_entire_population(**parameters)
        population = random.sample(population, population_size)
        return population

    def get_entire_population(self, **kwargs):
        entire_population = []
        keys = kwargs.keys()
        vals = kwargs.values()
        #TODO: WHY DOES APPEND LEADING ZEROS
        for instance in itertools.product(*vals):
            entire_population.append(dict(zip(keys, instance)))
        return entire_population

    def run(self):
        multiprocess(self)

    def get_fitness(self):
        pass

    def setup_new_generation(self):
        pass
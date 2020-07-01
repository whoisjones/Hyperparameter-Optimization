from GeneticParamOptimizer.hyperparameter.param_selection import SearchSpaceEvolutionary
from flair.data import Corpus
import itertools
import random

class ParamOptimizer():

    def __init__(
            self,
            corpus: Corpus,
            search_space: SearchSpaceEvolutionary,
            population_size: int = 32,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.01,
            ):

        self.corpus = corpus
        self.DNA_size = len(search_space.parameters)
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.budget = search_space.budget

        self.population = self.get_initial_population(search_space.parameters, population_size)

    def get_initial_population(self, parameters, population_size):
        population = self.get_entire_population(**parameters)
        population = random.sample(population, population_size)
        return population

    def get_entire_population(self, **kwargs):
        entire_population = []
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            entire_population.append(dict(zip(keys, instance)))
        return entire_population


class TextClassificationOptimizer(ParamOptimizer):
    def __init__(self, search_space):
        super().__init__(search_space)
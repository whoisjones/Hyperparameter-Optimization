from abc import abstractmethod
import random
import numpy as np
from random import randrange

from FlairParamOptimizer.parameters import ParameterStorage
from FlairParamOptimizer.search_spaces import SearchSpace

class SearchStrategy(object):

    def __init__(self):
        self.search_strategy = self.__class__.__name__

    @abstractmethod
    def make_configurations(self, parameter_storage: ParameterStorage):
        pass


class GridSearch(SearchStrategy):

    def __init__(self, shuffle : bool = False):
        super().__init__()
        self.shuffle = shuffle

    def make_configurations(self, search_space: SearchSpace):
        search_space.check_completeness(self.search_strategy)
        search_space.training_configurations.make_grid_configurations(search_space.parameter_storage)
        if self.shuffle:
            random.shuffle(search_space.training_configurations.configurations)


class RandomSearch(GridSearch):

    def __init__(self):
        super().__init__(shuffle=True)

class EvolutionarySearch(SearchStrategy):

    def __init__(
            self,
            population_size: int = 8,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.01,
    ):
        super().__init__()
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def make_configurations(self, search_space: SearchSpace):
        search_space.check_completeness(self.search_strategy)
        search_space.training_configurations.make_evolutionary_configurations(search_space.parameter_storage)

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
        parent_population = self._get_parent_population(search_space.training_configurations)
        selected_population = self._select(search_space.training_configurations, current_results)
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
        if self.has_document_embedding_specific_parameters == True:
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
from abc import abstractmethod
import random
import numpy as np
from random import randrange

from FlairParamOptimizer.parameter_collections import ParameterStorage
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
        search_space.training_configurations.make_evolutionary_configurations(search_space.parameter_storage, self.population_size)

    def _evolve_required(self, current_run: int):
        if current_run % (self.population_size) == 0:
            return True
        else:
            return False

    def _evolve(self, search_space: SearchSpace, current_results: dict):
        parent_population = self._get_parent_population(current_results)
        selected_population = self._select(current_results)
        for child in selected_population:
            child = self._crossover(child, parent_population)
            child = self._mutate(child, search_space.parameter_storage)
            self.configurations.append(child)

    def _get_parent_population(self, results: dict) -> dict:
        parent_population = self._extract_configurations_from_results(results)
        grouped_parent_population = self._group_by_embedding_keys(parent_population)
        return grouped_parent_population

    def _extract_configurations_from_results(self, results: dict) -> list:
        configurations = []
        for configuration in results.values():
            configurations.append(configuration.get("params"))
        return configurations

    def _group_by_embedding_keys(self, parent_population: list) -> dict:
        grouped_parent_population = {}
        for embedding in parent_population:
            embedding_key = self._get_embedding_key(embedding)
            embedding_value = embedding
            if embedding_key in grouped_parent_population:
                grouped_parent_population[embedding_key].append(embedding_value)
            else:
                grouped_parent_population[embedding_key] = [embedding_value]
        return grouped_parent_population

    def _get_embedding_key(self, embedding: dict):
        if embedding.get("document_embeddings") is not None:
            embedding_key = embedding['document_embeddings'].__name__
        else:
            embedding_key = "GeneralParameters"
        return embedding_key

    def _select(self, current_results: dict) -> np.array:
        current_configurations = [result.get("params") for result in current_results.values()]
        evolution_probabilities = self._get_fitness(current_results)
        return np.random.choice(current_configurations, size=self.population_size, replace=True, p=evolution_probabilities)

    def _get_fitness(self, results: dict):
        fitness = np.asarray([configuration['result'] for configuration in results.values()])
        probabilities = fitness / (sum([configuration['result'] for configuration in results.values()]))
        return probabilities

    def _crossover(self, child: dict, parent_population: dict):
        child_type = self._get_embedding_key(child)
        configuration_with_same_embedding = len(parent_population[child_type])
        DNA_size = len(child)
        if np.random.rand() < self.cross_rate:
            random_configuration = randrange(configuration_with_same_embedding)  # select another individual from pop
            parent = parent_population[child_type][random_configuration]
            cross_points = np.random.randint(0, 2, DNA_size).astype(np.bool)  # choose crossover points
            for (parameter, value), replace in zip(child.items(), cross_points):
                if replace:
                    child[parameter] = parent[parameter]  # mating and produce one child
        return child

    def _mutate(self, child: dict, parameter_storage: ParameterStorage):
        child_type = self._get_embedding_key(child)
        for parameter in child.keys():
            if np.random.rand() < self.mutation_rate:
                child[parameter] = random.sample(getattr(parameter_storage, child_type).get(parameter), 1)
        return child
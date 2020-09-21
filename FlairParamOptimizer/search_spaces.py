import time
import logging
from datetime import datetime
from enum import Enum
from abc import abstractmethod

from .parameter_collections import ParameterStorage, TrainingConfigurations
from FlairParamOptimizer.parameter_listings.parameters_for_user_input import Budget, EvaluationMetric, OptimizationValue
from FlairParamOptimizer.parameter_listings.parameter_groups import EMBEDDINGS

log = logging.getLogger("flair")

class SearchSpace(object):

    def __init__(self, has_document_embedding_specific_parameters: bool):
        self.parameter_storage = ParameterStorage()
        self.training_configurations = TrainingConfigurations()
        self.budget = Budget()
        self.current_run = 0
        self.optimization_value = {}
        self.evaluation_metric = {}
        self.max_epochs_per_training_run = 50
        self.has_document_embedding_specific_parameters = has_document_embedding_specific_parameters

    @abstractmethod
    def add_parameter(self,
                      parameter: Enum,
                      options):
        pass

    def add_budget(self, budget: Budget, amount: int):
        self.budget.add(budget_type=budget.value, amount=amount)

    def add_optimization_value(self, optimization_value: OptimizationValue):
        self.optimization_value = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: EvaluationMetric):
        self.evaluation_metric = evaluation_metric.value

    def add_max_epochs_per_training_run(self, max_epochs: int):
        self.max_epochs_per_training_run = max_epochs

    def check_completeness(self, search_strategy: str):
        self._check_mandatory_parameters_are_set()
        self._check_budget_type_matches_search_strategy(search_strategy)

    def _check_mandatory_parameters_are_set(self):
        self._check_steering_parameters()
        if self.has_document_embedding_specific_parameters:
            self._check_embeddings_are_set()

    def _check_steering_parameters(self):
        if not all([self.budget, self.optimization_value, self.evaluation_metric]):
            raise AttributeError("Please provide a budget, parameters, a optimization value and a evaluation metric for an optimizer.")

        if self.parameter_storage.is_empty():
            raise AttributeError("Parameters haven't been set.")

    def _check_embeddings_are_set(self):
        currently_set_parameters = self.parameter_storage.__dict__.keys()
        if not any(check in currently_set_parameters for check in EMBEDDINGS):
            raise AttributeError("Embeddings are required but missing.")

        union_of_embedding_types = [embedding for embedding in currently_set_parameters if embedding in EMBEDDINGS]
        for embedding in union_of_embedding_types:
            if not bool(getattr(self.parameter_storage, embedding).get("embeddings")) and embedding != "TransformerDocumentEmbeddings":
                raise KeyError("Please set WordEmbeddings for DocumentEmbeddings.")


    def _check_budget_type_matches_search_strategy(self, search_strategy: str):
        if 'generations' in self.budget.budget_type and search_strategy != "EvolutionarySearch":
            log.info("Can't assign generations to a an Optimizer which is not a GeneticOptimizer. Switching to runs.")
            self.budget.budget_type = "runs"

    def _set_additional_budget_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TextClassifierSearchSpace(SearchSpace):

    def __init__(self, multi_label: bool = False):
        super().__init__(has_document_embedding_specific_parameters=True)
        self.multi_label = multi_label

    def add_parameter(self,
                      parameter: Enum,
                      options: list):
        embedding_key_and_value_range_arguments = self._extract_embedding_keys_and_value_range_arguments(parameter, options)
        self.parameter_storage.add(parameter_name=parameter.value, **embedding_key_and_value_range_arguments)

    def _extract_embedding_keys_and_value_range_arguments(self, parameter: Enum, options: list,) -> list:
        function_arguments = {}
        if parameter.__class__.__name__ in EMBEDDINGS:
            function_arguments["embedding_key"] = parameter.__class__.__name__
            function_arguments["value_range"] = options
        else:
            function_arguments["value_range"] = options
        return function_arguments


class SequenceTaggerSearchSpace(SearchSpace):

    def __init__(self):
        super().__init__(has_document_embedding_specific_parameters=False)
        self.tag_type = ""

    def add_tag_type(self, tag_type: str):
        self.tag_type = tag_type

    def add_parameter(self,
                      parameter: Enum,
                      options: list):
        self.parameter_storage.add(parameter_name=parameter.value, value_range=options)

class Budget(object):

    def __init__(self):
        self.internal_counter_for_generations_budget = 0
        # Will be set if EvolutionarySearch is used
        self.population_size = None

    def add(self, budget_type: str, amount: int):
        self.budget_type = budget_type
        self.amount = amount
        if budget_type == "time_in_h":
            self.start_time = time.time()

    def _is_not_used_up(self):
        if self.budget_type == 'time_in_h':
            is_used_up = self._is_time_budget_left()
        elif self.budget_type == 'runs':
            is_used_up = self._is_runs_budget_left()
        elif self.budget_type == 'generations':
            is_used_up = self._is_generations_budget_left()
        self.internal_counter_for_generations_budget += 1
        return is_used_up

    def _is_time_budget_left(self):
        time_passed_since_start = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(self.start_time)
        if (time_passed_since_start.total_seconds()) / 3600 < self.amount:
            return True
        else:
            return False

    def _is_runs_budget_left(self):
        if self.amount > 0:
            self.amount -= 1
            return True
        else:
            return False

    def _is_generations_budget_left(self):
        # Decrease generations every X iterations (X is amount of individuals per generation)
        if self.amount > 1 \
                and self.internal_counter_for_generations_budget % self.population_size == 0 \
                and self.internal_counter_for_generations_budget != 0:
            self.amount -= 1
            return True
        # If last generation, budget is used up
        elif self.amount == 1 \
                and self.internal_counter_for_generations_budget % self.population_size == 0 \
                and self.internal_counter_for_generations_budget != 0:
            self.amount -= 1
            return False
        # If enough budget, pass
        elif self.amount > 0:
            return True

    def _set_population_size(self, population_size: int):
        self.population_size = population_size
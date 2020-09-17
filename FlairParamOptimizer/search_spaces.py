import time
import logging
from datetime import datetime
from enum import Enum
from abc import abstractmethod

from .parameters import ParameterStorage, TrainingConfigurations
from FlairParamOptimizer.parameter_listings.parameters_for_user_input import Budget, EvaluationMetric, OptimizationValue
from FlairParamOptimizer.parameter_listings.parameter_groups import EMBEDDINGS

log = logging.getLogger("flair")

class SearchSpace(object):

    def __init__(self, has_document_embedding_specific_parameters: bool):
        self.parameter_storage = ParameterStorage()
        self.training_configurations = TrainingConfigurations()
        self.budget = {}
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

    def add_budget(self, budget: Budget, amount):
        self.budget[budget.value] = amount
        if budget.value == "time_in_h":
            self.start_time = time.time()

    def add_optimization_value(self, optimization_value: OptimizationValue):
        self.optimization_value = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: EvaluationMetric):
        self.evaluation_metric = evaluation_metric.value

    def add_max_epochs_per_training_run(self, max_epochs: int):
        self.max_epochs_per_training_run = max_epochs

    def check_completeness(self, search_strategy: str):
        self._check_mandatory_parameters_are_set()
        self._check_budget_type_matches_optimizer_type(search_strategy)

    def _check_mandatory_parameters_are_set(self):
        self._check_steering_parameters()
        if self.has_document_embedding_specific_parameters:
            self._check_embeddings()

    def _check_steering_parameters(self):
        if not all([self.budget, self.parameter_storage, self.optimization_value, self.evaluation_metric, self.budget]):
            raise Exception("Please provide a budget, parameters, a optimization value and a evaluation metric for an optimizer.")

    def _check_embeddings(self):
        currently_set_parameters = self.parameter_storage.__dict__.keys()
        if not any(check in currently_set_parameters for check in EMBEDDINGS):
            raise Exception("Embeddings are required but missing.")

    def _check_budget_type_matches_optimizer_type(self, search_strategy: str):
        if 'generations' in self.budget and search_strategy != "EvolutionarySearch":
            log.info("Can't assign generations to a an Optimizer which is not a GeneticOptimizer. Switching to runs.")
            self.budget["runs"] = self.budget["generations"]
            del self.budget["generations"]

    def _budget_is_not_used_up(self):
        budget_type = self._get_budget_type(self.budget)
        if budget_type == 'time_in_h':
            return self._is_time_budget_left()
        elif budget_type == 'runs':
            return self._is_runs_budget_left()
        elif budget_type == 'generations':
            return self._is_generations_budget_left()

    def _get_budget_type(self, budget: dict):
        if len(budget) == 1:
            for budget_type in budget.keys():
                return budget_type
        else:
            raise Exception('Budget has more than 1 parameter.')

    def _is_time_budget_left(self):
        time_passed_since_start = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(self.start_time)
        if (time_passed_since_start.total_seconds()) / 3600 < self.budget['time_in_h']:
            return True
        else:
            return False

    def _is_runs_budget_left(self):
        if self.budget['runs'] > 0:
            self.budget['runs'] -= 1
            return True
        else:
            return False

    def _is_generations_budget_left(self):
        #Decrease generations every X iterations (X is amount of individuals per generation)
        if self.budget['generations'] > 1 \
        and self.current_run % self.population_size == 0\
        and self.current_run != 0:
            self.budget['generations'] -= 1
            return True
        #If last generation, budget is used up
        elif self.budget['generations'] == 1 \
        and self.current_run % self.population_size == 0\
        and self.current_run != 0:
            self.budget['generations'] -= 1
            return False
        #If enough budget, pass
        elif self.budget['generations'] > 0:
            return True

    def _get_current_configuration(self, current_run: int):
        current_configuration = self.training_configurations[current_run]
        return current_configuration

    def _get_technical_training_parameters(self):
        technical_training_parameters = {}
        technical_training_parameters["max_epochs"] = self.max_epochs_per_training_run
        technical_training_parameters["optimization_value"] = self.optimization_value
        return technical_training_parameters

    def _set_additional_budget_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TextClassifierSearchSpace(SearchSpace):

    def __init__(self):
        super().__init__(has_document_embedding_specific_parameters=True)

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
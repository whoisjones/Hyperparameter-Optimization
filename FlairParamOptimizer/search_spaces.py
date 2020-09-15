import time
from datetime import datetime
from enum import Enum
from abc import abstractmethod

from .parameters import ParameterCollection
from .sampling_functions import sampling_func
from .parameters_for_user_guidance import Budget, EvaluationMetric, OptimizationValue, TextClassifier
from .task_specific_parameters import EMBEDDING_SPECIFIC_PARAMETERS

"""
The Search Space object acts as a data object containing all configurations for the hyperparameter optimization.
We currently support two types of downstream task for hyperparameter optimization:
    Text Classification
    Sequence Labeling
    
Steering parameters which have to bet set independent of downstream task:
    Steering params:                        function to use:
    A budget preventing a long runtime      add_budget()
    A evaluation metric for training        add_evaluation_metric()
    An optimization value for training      add_optimization_value()
    Max epochs per training run             add_max_epochs_training() (default: 50)
    
For text classification, please first set document embeddings since you probably add document specific embeddings

Add parameters like this:

import FlairParamOptimizer.hyperparameter.parameters as param
from FlairParamOptimizer.hyperparameter.utils import func

search_space.add_parameter(param.[TYPE OF PARAMETER TO BE SET].[CONCRETE PARAMETER],
                           func.[FUNCTION TO PICK FROM VALUE RANGE],
                           options=[LIST OF PARAMETER VALUES] or range=[BOUNDS OF PARAMETER VALUES])

Note following combinations of functions and type of parameter values are possible:

    function:   value range argument:   explanation:
    choice      options=[1,2,3]         choose from different options
    uniform     bounds=[0, 0.5]         take a uniform sample between lower and upper bound
"""

class SearchSpace(object):

    def __init__(self, document_embedding_specific_parameters: bool):
        self.parameters = ParameterCollection()
        self.configurations = []
        self.budget = {}
        self.current_run = 0
        self.optimization_value = {}
        self.evaluation_metric = {}
        self.max_epochs_per_training_run = 50
        self.document_embedding_specific_parameters = document_embedding_specific_parameters

    @abstractmethod
    def add_parameter(self,
                      embedding_key: str,
                      parameter: Enum,
                      sampling_function: sampling_func,
                      kwargs):
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

    def _check_document_embeddings_are_set(self, parameter: Enum):
        if (parameter.name != "DOCUMENT_EMBEDDINGS" and not self.parameters):
            raise Exception("Please set document embeddings first in order to assign embeddings specific parameters later.")

    def _check_mandatory_parameters_are_set(self, optimizer_type: str):
        if not all([self.budget, self.parameters, self.optimization_value, self.evaluation_metric]) \
                and self._check_budget_type_matches_optimizer_type(optimizer_type):
            raise Exception("Please provide a budget, parameters, a optimization value and a evaluation metric for an optimizer.")

    def _check_budget_type_matches_optimizer_type(self, optimizer_type: str):
        if 'generations' in self.budget and optimizer_type == "GeneticOptimizer":
            return True
        elif 'runs' in self.budget or 'time_in_h' in self.budget:
            return True
        else:
            return False

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
        current_configuration = self.configurations[current_run]
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
        super().__init__(document_embedding_specific_parameters=True)

    def add_parameter(self,
                      parameter: Enum,
                      sampling_function: sampling_func,
                      **kwargs):
        try:
            sampling_func.validate_value_range(sampling_function, kwargs)
            unprocessed_value_range = sampling_func.extract_value_range(sampling_function, kwargs)
        except:
            raise Exception("Please provide correct value ranges to your sampling function.")

        """
        try:
            self._check_document_embeddings_are_set(parameter)
        except:
            raise Exception("Document Embeddings have to be set first.")
        """

        processed_embedding_keys_and_value_ranges = self._extract_embedding_keys_and_value_ranges(parameter, unprocessed_value_range)

        for embedding_key, value_range in processed_embedding_keys_and_value_ranges:
            self.parameters.add(embedding_key=embedding_key,
                                parameter_name=parameter.value,
                                sampling_function=sampling_function,
                                value_range=value_range)

    def _extract_embedding_keys_and_value_ranges(self, parameter: Enum, unprocessed_value_range: list) -> list:
        if parameter.name == TextClassifier.DOCUMENT_EMBEDDINGS.name:
            embedding_keys = [document_embedding.__name__ for document_embedding in unprocessed_value_range]
            value_ranges = [[document_embedding_class] for document_embedding_class in unprocessed_value_range]
        elif parameter.__class__.__name__ in EMBEDDING_SPECIFIC_PARAMETERS:
            embedding_keys = [parameter.__class__.__name__]
            value_ranges = [unprocessed_value_range]
        else:
            embedding_keys = ["GeneralParameters"]
            value_ranges = [unprocessed_value_range]
        processed_embedding_keys_and_value_ranges = list(zip(embedding_keys, value_ranges))
        return processed_embedding_keys_and_value_ranges

    def _insert_document_embeddings_hierarchy(self,
                                              parameter: Enum,
                                              func: sampling_func,
                                              options):
        try:
            for embedding in options:
                self.parameters[embedding.__name__] = {parameter.value: {"options": [embedding], "method": func}}
        except:
            raise Exception("Document embeddings only takes options as arguments")

    def _insert_parameters(self,
                           parameter: Enum,
                           func: sampling_func,
                           kwargs):
        if "Document" in parameter.__class__.__name__:
            self._insert_embedding_specific_parameter(parameter, func, kwargs)
        else:
            self._insert_universal_parameter(parameter, func, kwargs)

    def _insert_embedding_specific_parameter(self,
                                             parameter: Enum,
                                             func: sampling_func,
                                             kwargs):
        try:
            for key, values in kwargs.items():
                self.parameters[parameter.__class__.__name__].update({parameter.value: {key: values, "method": func}})
        except:
            raise Exception("If your want to assign document embedding specific parameters, make sure it is included in the search space.")

    def _insert_universal_parameter(self,
                                    parameter: Enum,
                                    func: sampling_func,
                                    kwargs):
        for embedding in self.parameters:
            for key, values in kwargs.items():
                self.parameters[embedding].update({parameter.value: {key: values, "method": func}})


class SequenceTaggerSearchSpace(SearchSpace):

    def __init__(self):
        super().__init__(document_embedding_specific_parameters=False)
        self.tag_type = ""

    def add_tag_type(self, tag_type: str):
        self.tag_type = tag_type

    def add_parameter(self,
                      parameter: Enum,
                      sampling_function: sampling_func,
                      **kwargs):
        for key, values in kwargs.items():
            self.parameters.update({parameter.value : {key: values, "method": sampling_function}})
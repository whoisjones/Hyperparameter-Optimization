import time
from enum import Enum
from abc import abstractmethod

from .utils import Func
from .parameters import Budget, EvaluationMetric, OptimizationValue

class SearchSpace(object):
    """
    Search space main class.

    Attributes:
        parameters          Parameters of all configurations
        budget              Budget for the hyperparameter optimization
        optimization_value  Metric which will be optimized during training
        evaluation_metric   Metric which is used for selecting the best configuration
        max_epochs_training max. number of iterations per training for a single configuration
    """

    def __init__(self):
        self.parameters = {}
        self.budget = {}
        self.optimization_value = {}
        self.evaluation_metric = {}
        self.max_epochs_training = 50

    @abstractmethod
    def add_parameter(self, parameter, func, **kwargs):
        """
        Adds single parameter configuration to search space. Overwritten by child class.
        :param parameter: passed
        :param func: passed
        :param kwargs: passed
        :return: passed
        """
        pass

    def add_budget(self, budget: Budget, value):
        """
        Adds a budget for the entire hyperparameter optimization.
        :param budget: Type of budget which is going to be used
        :param value: Budget value - depending on budget type
        :return: none
        """
        self.budget['type'] = budget.value
        self.budget['amount'] = value
        if budget.value == "time_in_h":
            self.budget['start_time'] = time.time()

    def add_optimization_value(self, optimization_value: OptimizationValue):
        """
        Adds optimization value to the search space.
        :param optimization_value: Optimization Value from Enum class.
        :return: none
        """
        self.optimization_value = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: EvaluationMetric):
        """
        Sets evaluation metric for training
        :param evaluation_metric:
        :return:
        """
        self.evaluation_metric = evaluation_metric.value

    def add_max_epochs_training(self, max_epochs: int):
        """
        Set max iteration per training for a single configuration
        :param max_epochs:
        :return:
        """
        self.max_epochs_training = max_epochs

    def _check_function_param_match(self, **kwargs):
        """
        Checks whether options or bounds are provided as value search space.
        :param kwargs:
        :return:
        """
        if len(kwargs) != 1 and \
                not "options" in kwargs and \
                not "bounds" in kwargs:
            raise Exception("Please provide either options or bounds depending on your function.")

    def _check_document_embeddings_are_set(self, parameter):
        if not self.parameters and parameter.name != "DOCUMENT_EMBEDDINGS":
            raise Exception("Please provide first the document embeddings in order to assign model specific attributes")


class TextClassifierSearchSpace(SearchSpace):
    """
    Search space for the text classification downstream task

    Attributes:
        inherited from SearchSpace object
    """

    def __init__(self):
        super().__init__()

    def add_parameter(self, parameter: Enum, func: Func, **kwargs):
        """
        Adds configuration for a single parameter to the search space.
        :param parameter: Type of parameter
        :param func: Function how to choose values from the parameter configuration
        :param kwargs: Either options or bounds depending on the function
        :return: None
        """
        self._check_function_param_match(kwargs)

        # This needs to be checked here,
        # since we want to set document embeddings specific parameters
        self._check_document_embeddings_are_set(parameter)

        if parameter.name == "DOCUMENT_EMBEDDINGS":
            self._add_document_embeddings(parameter, func, **kwargs)
        else:
            self._add_parameters(parameter, func, kwargs)


    def _add_document_embeddings(self, parameter, func, options):
        """
        Adds document embeddings to search space.
        :param parameter: Document Embedding to be set
        :param func:
        :param options:
        :return:
        """
        try:
            for embedding in options:
                self.parameters[embedding.__name__] = {parameter.value: {"options": [embedding], "method": func}}
        except:
            raise Exception("Document embeddings only takes options as arguments")


    def _add_parameters(self, parameter, func, kwargs):
        if "Document" in parameter.__class__.__name__:
            self._add_embedding_specific_parameter(parameter, func, kwargs)
        else:
            self._add_universal_parameter(parameter, func, kwargs)

    def _add_embedding_specific_parameter(self, parameter, func, kwargs):
        try:
            for key, values in kwargs.items():
                self.parameters[parameter.__class__.__name__].update({parameter.value: {key: values, "method": func}})
        except:
            raise Exception("If your want to assign document embedding specific parameters, make sure it is included in the search space.")

    def _add_universal_parameter(self, parameter, func, kwargs):
        for embedding in self.parameters:
            for key, values in kwargs.items():
                self.parameters[embedding].update({parameter.value: {key: values, "method": func}})

class SequenceTaggerSearchSpace(SearchSpace):

    def __init__(self):
        super().__init__()

        self.tag_type = ""

    def add_tag_type(self, tag_type : str):
        self.tag_type = tag_type

    def add_parameter(self, parameter, func, **kwargs):
        for key, values in kwargs.items():
            self.parameters.update({parameter.value : {key: values, "method": func}})
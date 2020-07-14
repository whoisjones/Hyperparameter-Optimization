from abc import abstractmethod
from .parameters import Budget, EvaluationMetric, OptimizationValue

class SearchSpace():

    def __init__(self):
        self.parameters = {}
        self.budget = {}
        self.optimization_value = {}
        self.evaluation_metric = {}

    @abstractmethod
    def add_parameter(self, parameter, func, **kwargs):
        pass

    def add_budget(self, budget: Budget, value):
        self.budget[budget.value] = value

    def add_optimization_value(self, optimization_value: OptimizationValue):
        self.optimization_value = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: EvaluationMetric):
        self.evaluation_metric = evaluation_metric.value


class TextClassifierSearchSpace(SearchSpace):

    def __init__(self):
        super().__init__()


    def add_parameter(self, parameter, func, **kwargs):
        if len(kwargs) != 1 and not "options" in kwargs and not "bounds" in kwargs:
            raise Exception("Please provide either options or bounds depending on your function.")

        if parameter.name == "DOCUMENT_EMBEDDINGS":
            self._add_document_embeddings(parameter, func, **kwargs)
        else:
            self._add_parameters(parameter, func, kwargs)


    def _add_document_embeddings(self, parameter, func, options):
        try:
            for embedding in options:
                self.parameters[embedding.__name__] = {parameter.value: {"options": embedding, "method": func}}
        except:
            raise Exception("Document embeddings only takes options as arguments")


    def _add_parameters(self, parameter, func, kwargs):
        if parameter.__class__.__name__ in self.parameters:
            self._add_embedding_specific_parameter(parameter, func, kwargs)
        else:
            self._add_universal_parameter(parameter, func, kwargs)

    def _add_embedding_specific_parameter(self, parameter, func, kwargs):
        for key, values in kwargs.items():
            self.parameters[parameter.__class__.__name__].update({parameter.value: {key: values, "method": func}})

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
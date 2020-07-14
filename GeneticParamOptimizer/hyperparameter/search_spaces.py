from abc import abstractmethod

from .parameters import *

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

    def add_evaluation_metric(self, evaluation_metric: OptimizationValue):
        self.evaluation_metric = evaluation_metric.value


class TextClassifierSearchSpace(SearchSpace):

    def __init__(self):
        super().__init__()

    def add_parameter(self, parameter, func, **kwargs):
        if len(kwargs) != 1 and not "options" in kwargs and not "bounds" in kwargs:
            raise Exception("Please provide either options or bounds depending on your function.")

        if parameter.name == "DOCUMENT_EMBEDDINGS":
            try:
                embeddings = kwargs['options']
                for embedding in embeddings:
                    self.parameters[embedding.__name__] = {parameter.value: embedding, "method": func}
            except:
                raise Exception("Document embeddings only takes options as arguments")
        else:
            if parameter.__class__.__name__ in self.parameters:
                for key, value in kwargs.items():
                    self.paramters[parameter.value] = {key: value, "method": func}
            else:
                print("")
                for embedding in self.parameters:
                    for key, value in kwargs.items():
                        self.parameters[embedding].update({parameter.value: value, "method": func})


class SequenceTaggerSearchSpace(SearchSpace):

    def __init__(self):
        super().__init__()
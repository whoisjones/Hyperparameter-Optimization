from .parameters import *

class SearchSpace():

    def __init__(self):
        self.parameters = {}
        self.budget = {}
        self.optimization_value = {}
        self.evaluation_metric = {}

    def add_parameter(self, parameter, func, **kwargs):
        if len(kwargs) != 1:
            raise Exception("Please provide either options or bounds depending on your function.")

        for key, values in kwargs.items():
            self.parameters[parameter] = {key:values, "method": func}

    def add_budget(self, budget: Budget, value):
        self.budget[budget.value] = value

    def add_optimization_value(self, optimization_value: OptimizationValue):
        self.optimization_metric["value"] = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: OptimizationValue):
        self.evaluation_metric["metric"] = evaluation_metric.value
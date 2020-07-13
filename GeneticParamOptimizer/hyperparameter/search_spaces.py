from .parameters import *

class SearchSpace():

    def __init__(self):
        self.parameters = {}
        self.budget = {}
        self.optimization_value = OptimizationValue.DEV_SCORE
        self.evaluation_metric = EvaluationMetric.MICRO_F1_SCORE

    def add_parameter(self, parameter, func, **kwargs):
        if len(kwargs) != 1:
            raise Exception("Please provide either options or bounds depending on your function.")

        for key, values in kwargs.items():
            self.parameters[parameter] = {key:values, "method": func}

    def add_budget(self, budget: Budget, value):
        self.budget[budget.value] = value

    def add_optimization_value(self, optimization_value: OptimizationValue):
        self.optimization_value = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: OptimizationValue):
        self.evaluation_metric = evaluation_metric.value
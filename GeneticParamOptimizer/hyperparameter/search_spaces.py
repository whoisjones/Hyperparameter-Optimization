from .parameters import Parameter, Budget

class SearchSpace():

    def __init__(self):
        self.parameters = {}
        self.budget = {}

    def add_parameter(self, parameter: Parameter, func, **kwargs):
        if len(kwargs) != 1:
            raise Exception("Please provide either options or bounds depending on your function.")

        for key, values in kwargs.items():
            self.parameters[parameter.value] = {key:values, "method": func}

    def add_budget(self, budget: Budget, value):
        self.budget[budget.value] = value
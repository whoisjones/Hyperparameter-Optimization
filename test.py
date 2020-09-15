import unittest

from FlairParamOptimizer.search_spaces import TextClassifierSearchSpace, SequenceTaggerSearchSpace
from FlairParamOptimizer.sampling_functions import sampling_func
import FlairParamOptimizer.parameters_for_user_guidance as param

class TestSearchSpaces(unittest.TestCase):

    def testAddParameter(self):
        provided_parameters = {"sampling_function":sampling_func.choice,
                               "values":[0.1, 0.2, 0.3]}
        search_space = TextClassifierSearchSpace()
        search_space.add_parameter(param.ModelTrainer.ANNEAL_FACTOR, provided_parameters.get("sampling_function"), options=provided_parameters.get("values"))
        if search_space.document_embedding_specific_parameters == True:
            pass
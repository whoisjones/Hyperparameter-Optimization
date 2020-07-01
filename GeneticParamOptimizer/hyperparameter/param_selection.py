import logging
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Tuple, Union
import numpy as np

from hyperopt import hp, fmin, tpe

import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentPoolEmbeddings, DocumentRNNEmbeddings
#TODO: ADD BUDGET
from GeneticParamOptimizer.hyperparameter import Budget
from flair.hyperparameter import Parameter
from flair.hyperparameter.parameter import (
    SEQUENCE_TAGGER_PARAMETERS,
    TRAINING_PARAMETERS,
    DOCUMENT_EMBEDDING_PARAMETERS,
    MODEL_TRAINER_PARAMETERS,
)
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import (
    EvaluationMetric,
    log_line,
    init_output_file,
    add_file_handler,
)

#TODO: NEW SEARCH SPACE CLASS
class SearchSpaceEvolutionary(object):
    def __init__(self):
        self.parameters = {}
        self.budget = {}

    def add(self, parameter: Parameter, values):
        self.parameters[parameter.value] = values

    def add_budget(self, budget: Budget, value):
        self.budget[budget.value] = value
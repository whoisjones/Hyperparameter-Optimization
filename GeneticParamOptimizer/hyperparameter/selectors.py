from typing import Union
from pathlib import Path
from abc import abstractmethod

from GeneticParamOptimizer.hyperparameter import SearchSpace, ParamOptimizer
from GeneticParamOptimizer.hyperparameter.parameters import *

import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from flair.training_utils import (
    EvaluationMetric
)

class ParamSelector():

    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            evaluation_metric: EvaluationMetric,
            optimization_value: OptimizationValue
    ):
        if type(base_path) is str:
            base_path = Path(base_path)

        self.corpus = corpus
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.optimization_value = optimization_value

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    def _objective(self, params):
        model = self._set_up_model(params)

    def optimize(self, optimizer: ParamOptimizer):
        optimizer = optimizer
        self._objective()

class TextClassificationParamSelector(ParamSelector):
    def __init__(
            self,
            corpus: Corpus,
            multi_label: bool,
            base_path: Union[str, Path],
            document_embedding_type: str,
            evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
            optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS,
    ):
        super().__init__(
            corpus,
            base_path,
            evaluation_metric,
            optimization_value
        )

        self.multi_label = multi_label
        self.document_embedding_type = document_embedding_type

    def _set_up_model(self, params: dict):

        if self.document_embedding_type == "lstm":
            embdding_params = {
                key: params[key] for key in params if key in DOCUMENT_RNN_EMBEDDING_PARAMETERS
            }
            document_embedding = DocumentRNNEmbeddings(**embdding_params)
        else:
            embdding_params = {
                key: params[key] for key in params if key in DOCUMENT_POOL_EMBEDDING_PARAMETERS
            }
            document_embedding = DocumentPoolEmbeddings(**embdding_params)

        text_classifier: TextClassifier = TextClassifier(
            label_dictionary=self.label_dictionary,
            multi_label=self.multi_label,
            document_embeddings=document_embedding,
        )

        return text_classifier
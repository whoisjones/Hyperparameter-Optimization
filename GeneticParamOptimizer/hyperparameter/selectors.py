from typing import Union
from pathlib import Path
from abc import abstractmethod
from datetime import datetime

from GeneticParamOptimizer.hyperparameter.optimizers import ParamOptimizer
from GeneticParamOptimizer.hyperparameter.parameters import *
from GeneticParamOptimizer.hyperparameter.multiprocessor import NonDaemonPool

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
            optimization_value: OptimizationValue,
            params: dict,
            max_epochs: int = 50,
    ):
        if type(base_path) is str:
            base_path = Path(base_path)

        self.corpus_name = corpus.__class__.__name__
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.optimization_value = optimization_value
        self.max_epochs = max_epochs
        self.params = params

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    def _train(self, params):
        pass

    def _objective(self, params):

        pool = NonDaemonPool()
        results = []
        for task in params:
            results.append(pool.apply_async(self._train, args=(task,)))
        pool.close()
        pool.join()

        results = [r.get() for r in results]

    def optimize(self):
        self._objective(params=self.params)


class TextClassificationParamSelector(ParamSelector):
    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            optimizer: ParamOptimizer,
            multi_label: bool = False,
            max_epochs: int = 50,
    ):
        super().__init__(
            corpus,
            base_path,
            evaluation_metric=optimizer.evaluation_metric,
            optimization_value=optimizer.optimization_value,
            params=optimizer.search_grid,
            max_epochs=max_epochs
        )

        self.multi_label = multi_label
        if optimizer.parameters['document_embeddings'] != None:
            self.document_embedding_type = optimizer.parameters['document_embeddings']
        else:
            raise Exception("Please provide a document embedding for text classification.")
        self.label_dictionary = corpus.make_label_dictionary()

    def _set_up_model(self, params: dict):
        if self.document_embedding_type == "DocumentRNNEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_RNN_EMBEDDING_PARAMETERS
            }
            document_embedding = DocumentRNNEmbeddings(**embedding_params)
        elif self.document_embedding_type == "DocumentPoolEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_POOL_EMBEDDING_PARAMETERS
            }
            document_embedding = DocumentPoolEmbeddings(**embedding_params)
        elif self.document_embedding_type == "TransformerDocumentEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS
            }
            document_embedding = TransformerDocumentEmbeddings(**embedding_params)
        else:
            raise Exception("Please provide a flair document embedding class")

        text_classifier: TextClassifier = TextClassifier(
            label_dictionary=self.label_dictionary,
            multi_label=self.multi_label,
            document_embeddings=document_embedding,
        )

        return text_classifier

    def _train(self, params):
        corpus_class = eval(self.corpus_name)
        corpus = corpus_class()

        for sent in corpus.get_all_sentences():
            sent.clear_embeddings()

        model = self._set_up_model(params)

        training_params = {
            key: params[key] for key, value in params.items() if key in TRAINING_PARAMETERS
        }
        model_trainer_parameters = {
            key: params[key] for key, value in params.items() if key in MODEL_TRAINER_PARAMETERS
        }

        trainer: ModelTrainer = ModelTrainer(
            model, corpus, **model_trainer_parameters
        )

        path = Path(self.base_path) / str(datetime.now())

        results = trainer.train(
            path,
            max_epochs=self.max_epochs,
            param_selection_mode=True,
            **training_params,
        )

        return results

#TODO: IMPLEMENT
class SequenceTaggerParamSelector(ParamSelector):
    def __init__(
        self,
        corpus: Corpus,
        tag_type: str,
        base_path: Union[str, Path],
        max_epochs: int = 50,
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        training_runs: int = 1,
        optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS,
    ):
        """
        :param corpus: the corpus
        :param tag_type: tag type to use
        :param base_path: the path to the result folder (results will be written to that folder)
        :param max_epochs: number of epochs to perform on every evaluation run
        :param evaluation_metric: evaluation metric used during training
        :param training_runs: number of training runs per evaluation run
        :param optimization_value: value to optimize
        """
        super().__init__(
            corpus,
            base_path,
            max_epochs,
            evaluation_metric,
            training_runs,
            optimization_value,
        )

        self.tag_type = tag_type
        self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

    def _set_up_model(self, params: dict):
        sequence_tagger_params = {
            key: params[key] for key in params if key in SEQUENCE_TAGGER_PARAMETERS
        }

        tagger: SequenceTagger = SequenceTagger(
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            **sequence_tagger_params,
        )
        return tagger
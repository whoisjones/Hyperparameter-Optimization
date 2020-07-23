from typing import Union
from pathlib import Path
import os
from abc import abstractmethod
from datetime import datetime
import time
import multiprocessing

from GeneticParamOptimizer.hyperparameter.optimizers import *
from GeneticParamOptimizer.hyperparameter.multiprocessor import NonDaemonPool
from GeneticParamOptimizer.hyperparameter.helpers import *

import flair.nn
from flair.data import Corpus
from flair.datasets import *
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings, WordEmbeddings
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
            budget: Budget,
            params: dict,
            optimizer_type: str,
            optimizer: ParamOptimizer,
            max_epochs: int,
    ):

        if type(base_path) is str:
            base_path = Path(base_path)

        self.corpus_name = corpus.__name__
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.optimization_value = optimization_value
        self.budget = budget
        self.params = params
        self.optimizer_type = optimizer_type
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.best_config = {}

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    @abstractmethod
    def _train(self, params):
        pass

    @abstractmethod
    def _evaluate(self):
        pass

    @abstractmethod
    def _process_results(self):
        pass

    def optimize(self, parallel_processes : int = os.cpu_count()):

        while self._budget_is_not_used_up:
            results = self._objective(params=self.params, parallel_processes=parallel_processes)
            self._process_results(results)

        print(self.best_config)

    def _objective(self, params, parallel_processes):

        results = []
        multiprocessing.set_start_method('spawn', force=True)
        pool = NonDaemonPool(processes=parallel_processes)
        for task in params:
            results.append(pool.apply_async(self._train, args=(task,)))
        pool.close()
        pool.join()

        return [p.get() for p in results]

    @property
    def _budget_is_not_used_up(self):

        if self.optimizer_type in ["GridSearchOptimizer", "RandomSearchOptimizer"] \
                and self.budget['type'] == "runs" \
                and self.budget['amount'] > 0:
            self.params = self.params[:self.budget['amount']]
            self.budget['amount'] = 0
            return True

        if self.budget['type'] == 'runs' \
                and self.budget['amount'] > 0:
            self.budget['amount'] = self.budget['amount'] - 1
            return True
        else:
            return False

        if self.budget['type'] == 'time_in_h' \
                and time.time() - self.budget['start_time'] < self.budget['amount']:
            return True
        else:
            return False


class TextClassificationParamSelector(ParamSelector):
    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            optimizer,
            multi_label: bool = False,
    ):
        super().__init__(
            corpus,
            base_path,
            evaluation_metric=optimizer.evaluation_metric,
            optimization_value=optimizer.optimization_value,
            budget= optimizer.budget,
            params=optimizer.search_grid,
            optimizer_type=optimizer.__class__.__name__,
            optimizer=optimizer,
            max_epochs=optimizer.max_epochs_training
        )

        self.multi_label = multi_label

    def _set_up_model(self, params: dict, label_dictionary : dict):

        document_embedding = params['document_embeddings'].__name__
        if document_embedding == "DocumentRNNEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_RNN_EMBEDDING_PARAMETERS
            }
            embedding_params['embeddings'] = [WordEmbeddings(TokenEmbedding) if type(params['embeddings']) == list
                                              else WordEmbeddings(params['embeddings']) for TokenEmbedding in params['embeddings']]
            document_embedding = DocumentRNNEmbeddings(**embedding_params)

        elif document_embedding == "DocumentPoolEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_POOL_EMBEDDING_PARAMETERS
            }
            embedding_params['embeddings'] = [WordEmbeddings(TokenEmbedding) for TokenEmbedding in params['embeddings']]
            document_embedding = DocumentPoolEmbeddings(**embedding_params)

        elif document_embedding == "TransformerDocumentEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS
            }
            document_embedding = TransformerDocumentEmbeddings(**embedding_params)

        else:
            raise Exception("Please provide a flair document embedding class")

        text_classifier: TextClassifier = TextClassifier(
            label_dictionary=label_dictionary,
            multi_label=self.multi_label,
            document_embeddings=document_embedding,
        )

        return text_classifier

    def _train(self, params):

        corpus_class = eval(self.corpus_name)
        corpus = corpus_class()

        label_dict = corpus.make_label_dictionary()

        for sent in corpus.get_all_sentences():
            sent.clear_embeddings()

        model = self._set_up_model(params, label_dict)

        training_params = {
            key: params[key] for key, value in params.items() if key in TRAINING_PARAMETERS
        }
        model_trainer_parameters = {
            key: params[key] for key, value in params.items() if key in MODEL_TRAINER_PARAMETERS and key != 'model'
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
        if self.optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result':result, 'params':params}

    def _process_results(self, results: list):

        sorted_results = sorted(results, key=lambda k: k['result'], reverse=True)
        self.best_config = sorted_results[0]

        if self.optimizer_type == "GeneticOptimizer":
            self.params = self.optimizer._evolve(sorted_results)

        return

class SequenceTaggerParamSelector(ParamSelector):
    def __init__(
        self,
        corpus: Corpus,
        tag_type: str,
        base_path: Union[str, Path],
        optimizer: Optimizer,
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
            evaluation_metric=optimizer.evaluation_metric,
            optimization_value=optimizer.optimization_value,
            budget=optimizer.budget,
            params=optimizer.search_grid,
            optimizer_type=optimizer.__class__.__name__,
            optimizer=optimizer,
            max_epochs=optimizer.max_epochs_training
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

    def _train(self):
        pass

    def _process_results(self, results: list):

        if self.__class__.__name__ == "TextClassificationParamSelector":
            pass
        elif self.__class__.__name__ == "SequenceTaggerParamSelector":
            pass
from typing import Union
from pathlib import Path
import os
from abc import abstractmethod
from datetime import datetime
import time
from torch.cuda import device_count

from GeneticParamOptimizer.hyperparameter.optimizers import *
from GeneticParamOptimizer.hyperparameter.search_spaces import SearchSpace
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
    """
    The ParamSelector selects the best configuration omitted by an optimizer object
    """

    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            optimizer: ParamOptimizer,
            search_space: SearchSpace,
    ):

        if type(base_path) is str:
            base_path = Path(base_path)

        self.corpus = corpus
        self.base_path = base_path
        self.optimizer = optimizer
        self.search_space = search_space
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

    def optimize(self, train_on_multiple_gpus : bool = False):

        while self._budget_is_not_used_up():
            if train_on_multiple_gpus and self._sufficient_available_gpus():
                results = self._perform_training_on_multiple_gpus(self.params)
            else:
                results = self._perform_training(params=self.params)
            self._process_results(results)

        print(self.best_config)

    def _sufficient_available_gpus(self):
        if device_count() > 1:
            return True
        else:
            log.info("It is less than 2 GPUs available, switching to standard calculation.")

    def _perform_training(self, params):

        results = []

        for task in params:
            results.append(pool.apply_async(self._train, args=(task,)))
        pool.close()
        pool.join()

        return [p.get() for p in results]

    def _perform_training_on_multiple_gpus(self, params):
        #TODO to be implemented
        pass


    def _budget_is_not_used_up(self):

        budget_type, budget_value = self._get_budget_information(self.search_space.budget)

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

    def _get_budget_information(self, budget: dict):
        if len(budget) == 1:
            for budget_type, value in budget.items():
                return budget_type, value


class TextClassificationParamSelector(ParamSelector):
    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            optimizer: ParamOptimizer,
            search_space: SearchSpace,
            multi_label: bool = False,
    ):
        super().__init__(
            corpus,
            base_path,
            optimizer=optimizer,
            search_space=search_space,
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
        base_path: Union[str, Path],
        optimizer: Optimizer,
        search_space: SearchSpace,
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
            optimizer=optimizer,
            search_space=search_space
        )

        self.tag_type = search_space.tag_type
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
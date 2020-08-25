import time
import os
import logging
from datetime import datetime
from operator import getitem
from typing import Union
from pathlib import Path
from torch.cuda import device_count
from abc import abstractmethod

from GeneticParamOptimizer.hyperparameter.optimizers import *
from GeneticParamOptimizer.hyperparameter.search_spaces import SearchSpace
from GeneticParamOptimizer.hyperparameter.helpers import *

import flair.nn
from flair.data import Corpus
from flair.datasets import *
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings, WordEmbeddings
from flair.models import TextClassifier
from flair.training_utils import (
    EvaluationMetric
)

log = logging.getLogger("flair")

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
        self.results = {}
        self.current_run = 0

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    @abstractmethod
    def _train(self, params):
        pass

    def optimize(self, train_on_multiple_gpus : bool = False):
        while self._budget_is_not_used_up():
            current_configuration = self._get_current_configuration()
            if train_on_multiple_gpus and self._sufficient_available_gpus():
                self._perform_training_on_multiple_gpus(current_configuration)
            else:
                self._perform_training(current_configuration)

        self._log_results()

    def _perform_training(self, params):
        self.results[f"training-run-{self.current_run}"] = self._train(params)

    def _perform_training_on_multiple_gpus(self, params):
        #TODO to be implemented
        pass

    def _budget_is_not_used_up(self):

        budget_type = self._get_budget_type(self.search_space.budget)

        if budget_type == 'time_in_h':
            return self._is_time_budget_left()
        elif budget_type == 'runs':
            return self._is_runs_budget_left()
        elif budget_type == 'generations':
            return self._is_generations_budget_left()

    def _get_budget_type(self, budget: dict):
        if len(budget) == 1:
            for budget_type in budget.keys():
                return budget_type
        else:
            raise Exception('Budget has more than 1 parameter.')

    def _is_time_budget_left(self):
        already_running = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(self.search_space.start_time)
        if (already_running.total_seconds()) / 3600 < self.search_space.budget['time_in_h']:
            return True
        else:
            return False

    def _is_runs_budget_left(self):
        if self.search_space.budget['runs'] > 0:
            self.search_space.budget['runs'] -= 1
            return True
        else:
            return False

    def _is_generations_budget_left(self):
        if self.search_space.budget['generations'] > 0 \
        and self.current_run % self.optimizer.population_size == 0:
            self.search_space.budget['generations'] -= 1
            return True
        else:
            return False

    def _get_current_configuration(self):
        current_configuration = self.optimizer.configurations[self.current_run]
        self.current_run += 1
        return current_configuration

    def _sufficient_available_gpus(self):
        if device_count() > 1:
            return True
        else:
            log.info("There are less than 2 GPUs available, switching to standard calculation.")

    def _log_results(self):
        sorted_results = sorted(self.results.items(), key=lambda x: getitem(x[1], 'result'), reverse=True)[:5]
        log.info("The top 5 results are:")
        for idx, config in enumerate(sorted_results):
            log.info(50*'-')
            log.info(idx+1)
            log.info(f"{config[0]} with a score of {config[1]['result']}.")
            log.info("with following configurations:")
            for parameter, value in config[1]['params'].items():
                log.info(f"{parameter}:  {value}")



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

        corpus = self.corpus()

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

        path = Path(self.base_path) / f"training-run-{self.current_run}"

        results = trainer.train(
            path,
            max_epochs=self.search_space.max_epochs_per_training,
            param_selection_mode=True,
            **training_params,
        )

        if self.search_space.optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result': result, 'params': params}

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
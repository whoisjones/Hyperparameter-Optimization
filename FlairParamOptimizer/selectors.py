import logging
import pickle
from datetime import datetime
from operator import getitem
from typing import Union
from pathlib import Path
from torch import cuda

from FlairParamOptimizer.optimizers import *
from FlairParamOptimizer.search_spaces import SearchSpace
from FlairParamOptimizer.parameter_listings.parameter_groups import *

import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import TextClassifier, SequenceTagger
from flair.trainers import ModelTrainer

log = logging.getLogger("flair")

class ParamSelector():

    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path]
    ):
        if type(base_path) is str:
            base_path = Path(base_path)
        self.corpus = corpus
        self.base_path = base_path
        self.current_run = 0
        self.results = {}

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    @abstractmethod
    def _train(self, params: dict):
        pass

    def optimize(self, optimizer: Optimizer, search_space: SearchSpace, train_on_multiple_gpus : bool = False):
        self.technical_training_parameters = search_space._get_technical_training_parameters()
        while search_space._budget_is_not_used_up():
            current_configuration = search_space._get_current_configuration(search_space.current_run)
            if train_on_multiple_gpus and self._sufficient_available_gpus():
                self._perform_training_on_multiple_gpus(current_configuration)
            else:
                self._perform_training(current_configuration, current_run=search_space.current_run)
            if optimizer.__class__.__name__ == "GeneticOptimizer" \
            and optimizer._evolve_required(current_run=search_space.current_run):
                optimizer._evolve(search_space, self.results)
            search_space.current_run += 1
        self._log_results()

    def _perform_training(self, params: dict, current_run: int):
        self.results[f"training-run-{current_run}"] = self._train(params, self.technical_training_parameters)
        self._store_results(result=self.results[f"training-run-{current_run}"], current_run=current_run)

    def _perform_training_on_multiple_gpus(self, params: dict):
        #TODO to be implemented
        pass

    def _sufficient_available_gpus(self):
        if cuda.device_count() > 1:
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

    def _store_results(self, result: dict, current_run: int):
        result['timestamp'] = datetime.now()
        entry = f"training-run-{current_run}"
        try:
            self._load_and_pickle_results(entry, result)
        except FileNotFoundError:
            self._initialize_results_pickle(entry, result)

    def _load_and_pickle_results(self, entry: str, result: dict):
        pickle_file = open(self.base_path / "results.pkl", 'rb')
        results_dict = pickle.load(pickle_file)
        pickle_file.close()
        pickle_file = open(self.base_path / "results.pkl", 'wb')
        results_dict[entry] = result
        pickle.dump(results_dict, pickle_file)
        pickle_file.close()

    def _initialize_results_pickle(self, entry: str, result: dict):
        results_dict = {}
        pickle_file = open(self.base_path / "results.pkl", 'wb')
        results_dict[entry] = result
        pickle.dump(results_dict, pickle_file)
        pickle_file.close()


class TextClassificationParamSelector(ParamSelector):
    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            multi_label: bool = False,
    ):
        super().__init__(
            corpus,
            base_path,
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

    def _train(self, params: dict, technical_training_parameters: dict):
        corpus = self.corpus
        label_dict = corpus.make_label_dictionary()
        for sent in corpus.get_all_sentences():
            sent.clear_embeddings()
        model = self._set_up_model(params, label_dict)
        training_parameters = {
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
            max_epochs=technical_training_parameters["max_epochs"],
            param_selection_mode=True,
            **training_parameters
        )

        if technical_training_parameters["optimization_value"] == "score":
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

        embedding_types = params['embeddings']

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        sequence_tagger_params['embeddings'] = embeddings

        tagger: SequenceTagger = SequenceTagger(
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            **sequence_tagger_params,
        )

        return tagger

    def _train(self, params: dict):
        """
        trains a sequence tagger model
        :param params: dict containing the parameters
        :return: dict containing result and configuration
        """

        corpus = self.corpus

        tagger = self._set_up_model(params=params)

        training_params = {
            key: params[key] for key, value in params.items() if key in TRAINING_PARAMETERS
        }
        model_trainer_parameters = {
            key: params[key] for key, value in params.items() if key in MODEL_TRAINER_PARAMETERS and key != 'model'
        }

        trainer: ModelTrainer = ModelTrainer(
            tagger, corpus, **model_trainer_parameters
        )

        path = Path(self.base_path) / f"training-run-{self.current_run}"

        results = trainer.train(path,
                      max_epochs=self.search_space.max_epochs_per_training,
                      **training_params)

        if self.search_space.optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result': result, 'params': params}
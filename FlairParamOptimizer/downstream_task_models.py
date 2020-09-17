from abc import abstractmethod
from pathlib import Path

import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings, WordEmbeddings, \
    TransformerDocumentEmbeddings, StackedEmbeddings
from flair.models import TextClassifier, SequenceTagger
from flair.trainers import ModelTrainer

from FlairParamOptimizer.parameter_listings.parameter_groups import *


class DownstreamTaskModel(object):

    def __init__(self):
        pass

    @abstractmethod
    def _set_up_model(self, params: dict, label_dictionary) -> flair.nn.Model:
        pass

    @abstractmethod
    def _train(self, corpus: Corpus, params: dict, base_path: Path, max_epochs: int, optimization_value: str):
        pass

class TextClassification(DownstreamTaskModel):

    def __init__(self, multi_label: bool = False):
        super().__init__()
        self.multi_label = multi_label

    def _set_up_model(self, params: dict, label_dictionary):
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

    def _train(self, corpus: Corpus, params: dict, base_path: Path, max_epochs: int, optimization_value: str):
        corpus = corpus
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
        path = base_path
        results = trainer.train(
            path,
            max_epochs=max_epochs,
            param_selection_mode=True,
            **training_parameters
        )

        if optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result': result, 'params': params}

class SequenceTagging(DownstreamTaskModel):

    def __init__(self):
        super().__init__()
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
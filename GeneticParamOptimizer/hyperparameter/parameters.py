from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import inspect
from enum import Enum

"""
Parameter for configuration of hyperparameter optimization
"""
class EvaluationMetric(Enum):
    MICRO_ACCURACY = "micro-average accuracy"
    MICRO_F1_SCORE = "micro-average f1-score"
    MACRO_ACCURACY = "macro-average accuracy"
    MACRO_F1_SCORE = "macro-average f1-score"
    MEAN_SQUARED_ERROR = "mean squared error"

class OptimizationValue(Enum):
    DEV_LOSS = "loss"
    DEV_SCORE = "score"

class Budget(Enum):
    RUNS = "runs"
    GENERATIONS = "generations"
    TIME_IN_H = "time_in_h"

"""
Parameter for torch optimizer
"""
class ParameterOptimizer(Enum):
    MOMENTUM = "momentum"
    DAMPENING = "dampening"
    WEIGHT_DECAY = "weight_decay"
    NESTEROV = "nesterov"
    AMSGRAD = "amsgrad"
    BETAS = "betas"

"""
Parameter for Model Trainer class und its function train()
"""
class ParameterModelTrainer():
    OPTIMIZER = "optimizer"
    EPOCH = "epoch"
    USE_TENSORBOARD = "use_tensorboard"

class ParameterTraining():
    LEARNING_RATE = "learning_rate"
    MINI_BATCH_SIZE = "mini_batch_size"
    MINI_BATCH_CHUNK_SIZE = "mini_batch_chunk_size"
    MAX_EPOCHS = "max_epochs"
    ANNEAL_FACTOR = "anneal_factor"
    ANNEAL_WITH_RESTARTS = "anneal_with_restarts"
    PATIENCE = "patience"
    INITIAL_EXTRA_PATIENCE = "initial_extra_patience"
    MIN_LEARNING_RATE = "min_learning_rate"
    TRAIN_WITH_DEV = "train_with_dev"
    NUM_WORKERS = "num_workers"

"""
Parameter for Downstream Tasks
"""
class ParameterSequenceTagger():
    HIDDEN_SIZE = "hidden_size"
    EMBEDDINGS = "embeddings"
    USE_CRF = "use_crf"
    USE_RNN = "use_rnn"
    RNN_LAYERS = "rnn_layers"
    DROPOUT = "dropout"
    WORD_DROPOUT = "word_dropout"
    LOCKED_DROPOUT = "locked_dropout"
    REPROJECT_TO = "reproject_to"
    TRAIN_INITIAL_HIDDEN_STATE = "train_initial_hidden_state"
    BETA = "beta"

class ParameterTextClassifier():
    DOCUMENT_EMBEDDINGS = "document_embeddings"
    BETA = "beta"

"""
Parameter for text classification embeddings
"""
class ParameterDocumentRNNEmbeddings():
    EMBEDDINGS = "embeddings"
    HIDDEN_SIZE = "hidden_size"
    RNN_LAYERS = "rnn_layers"
    REPROJECT_WORDS = "reproject_words"
    REPROJECT_WORDS_DIMENSION = "reproject_words_dimension"
    BIDIRECTIONAL = "bidirectional"
    DROPOUT = "dropout"
    WORD_DROPOUT = "word_dropout"
    LOCKED_DROPOUT = "locked_dropout"
    RNN_TYPE = "rnn_type"
    FINE_TUNE = "fine_tune"

class ParameterDocumentPoolEmbeddings():
    EMBEDDINGS = "embeddings"
    FINE_TUNE_MODE = "fine_tune_mode"
    POOLING = "pooling"

class ParameterTransformerDocumentEmbeddings():
    MODEL = "model"
    FINE_TUNE = "fine_tune"
    BATCH_SIZE = "batch_size"
    LAYERS = "layers"
    USE_SCALER_MIX = "use_scalar_mix"


OPTIMIZER_PARAMETERS = [param.value for param in ParameterOptimizer]
DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS = inspect.getfullargspec(TransformerDocumentEmbeddings).args
DOCUMENT_RNN_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentRNNEmbeddings).args
DOCUMENT_POOL_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentPoolEmbeddings).args
SEQUENCE_TAGGER_PARAMETERS = inspect.getfullargspec(SequenceTagger).args
TRAINING_PARAMETERS = inspect.getfullargspec(ModelTrainer.train).args + OPTIMIZER_PARAMETERS
MODEL_TRAINER_PARAMETERS = inspect.getfullargspec(ModelTrainer).args
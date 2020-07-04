from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import inspect
from enum import Enum

class OptimizationValue(Enum):
    DEV_LOSS = "loss"
    DEV_SCORE = "score"

class Budget(Enum):
    RUNS = "runs"
    GENERATIONS = "generations"
    TIME_IN_H = "time_in_h"

class Parameter(Enum):
    EMBEDDINGS = "embeddings"
    HIDDEN_SIZE = "hidden_size"
    USE_CRF = "use_crf"
    USE_RNN = "use_rnn"
    RNN_LAYERS = "rnn_layers"
    RNN_TYPE = "rnn_type"
    DROPOUT = "dropout"
    WORD_DROPOUT = "word_dropout"
    LOCKED_DROPOUT = "locked_dropout"
    LEARNING_RATE = "learning_rate"
    MINI_BATCH_SIZE = "mini_batch_size"
    ANNEAL_FACTOR = "anneal_factor"
    ANNEAL_WITH_RESTARTS = "anneal_with_restarts"
    PATIENCE = "patience"
    REPROJECT_WORDS = "reproject_words"
    REPROJECT_WORD_DIMENSION = "reproject_words_dimension"
    BIDIRECTIONAL = "bidirectional"
    OPTIMIZER = "optimizer"
    MOMENTUM = "momentum"
    DAMPENING = "dampening"
    WEIGHT_DECAY = "weight_decay"
    NESTEROV = "nesterov"
    AMSGRAD = "amsgrad"
    BETAS = "betas"
    EPS = "eps"
    POOLING = "pooling"
    FINE_TUNE_MODE = "fine_tune_mode"
    USE_SCALAR_MIX = "use_scalar_mix"
    LAYERS = "layers"
    BATCH_SIZE = "batch_size"

DOCUMENT_RNN_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentRNNEmbeddings).args
DOCUMENT_POOL_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentPoolEmbeddings).args
SEQUENCE_TAGGER_PARAMETERS = inspect.getfullargspec(SequenceTagger).args
TRAINING_PARAMETERS = inspect.getfullargspec(ModelTrainer.train()).args
MODEL_TRAINER_PARAMETERS = inspect.getfullargspec(ModelTrainer).args
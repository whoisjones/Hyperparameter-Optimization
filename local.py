from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter, TextClassifierParamSelector, OptimizationValue

# load your corpus
corpus = TREC_6().downsample(0.1)

# define your search space
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    [ FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward') ]
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

# create the parameter selector
param_selector = TextClassifierParamSelector(
    corpus,
    False,
    'resources/results',
    'lstm',
    max_epochs=50,
    training_runs=3,
    optimization_value=OptimizationValue.DEV_SCORE
)

# start the optimization
param_selector.optimize(search_space, max_evals=100)
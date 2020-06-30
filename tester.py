from GeneticParamOptimizer import optimizers
from hyperopt import hp
from GeneticParamOptimizer.hyperparameter.param_selection import NewSearchSpace, Parameter, Budget
from flair.embeddings import WordEmbeddings, FlairEmbeddings

# define your search space
search_space = NewSearchSpace()
search_space.add(Parameter.EMBEDDINGS, [ WordEmbeddings('glove') ])
search_space.add(Parameter.HIDDEN_SIZE, [32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, [1, 2])
search_space.add(Parameter.DROPOUT, [0, 0.5])
search_space.add(Parameter.LEARNING_RATE, [0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, [8, 16, 32])
search_space.add_budget(Budget.RUNS, 50)


optimizer = optimizers.TextClassificationOptimizer()
print("")
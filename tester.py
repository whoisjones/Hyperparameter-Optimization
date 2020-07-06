from GeneticParamOptimizer.hyperparameter import selectors, search_spaces, optimizers
from GeneticParamOptimizer.hyperparameter.parameters import Parameter, Budget, OptimizationValue
from GeneticParamOptimizer.hyperparameter.utils import *
from flair.embeddings import WordEmbeddings
from flair.datasets import TREC_6

corpus = TREC_6()
search_space = search_spaces.SearchSpace()
search_space.add_parameter(Parameter.EMBEDDINGS, choice, options=[[WordEmbeddings('glove')]])
search_space.add_parameter(Parameter.BATCH_SIZE, choice, options=[4, 8, 12, 16])
search_space.add_parameter(Parameter.HIDDEN_SIZE, choice, options=[128, 256, 512])
search_space.add_parameter(Parameter.LEARNING_RATE , choice, options=[0.01, 0.05, 0.1])
search_space.add_parameter(Parameter.DROPOUT, uniform, bounds=[0, 0.5])
search_space.add_budget(Budget.GENERATIONS, 50)
optimizer = optimizers.GeneticOptimizer(search_space=search_space, population_size=6)
param_selector = selectors.TextClassificationParamSelector(corpus=corpus, multi_label=False, base_path='resources/hyperopt', document_embedding_type='lstm', max_epochs=5)
param_selector.optimize(optimizer=optimizer)
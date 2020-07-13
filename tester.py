from GeneticParamOptimizer.hyperparameter import selectors, search_spaces, optimizers
from GeneticParamOptimizer.hyperparameter.parameters import *
from GeneticParamOptimizer.hyperparameter.utils import choice, uniform
from flair.embeddings import WordEmbeddings
from flair.datasets import TREC_6

# 1.) Define your corpus
corpus = TREC_6()

# 2.) create an search space
search_space = search_spaces.SearchSpace()

# 3.) depending on your task add the respective parameters you want to optimize over

#Define your budget and optmization metric
search_space.add_budget(Budget.GENERATIONS, 50)
search_space.add_evaluation_metric(EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(OptimizationValue.DEV_SCORE)

#Depending on your downstream task, add embeddings and specify these with the respective Parameters below
search_space.add_parameter(ParameterTextClassifier.DOCUMENT_EMBEDDINGS, choice, options=[DocumentRNNEmbeddings,
                                                                                         TransformerDocumentEmbeddings,
                                                                                         DocumentPoolEmbeddings])
search_space.add_parameter(ParameterTraining.LEARNING_RATE, choice, options=[0.01, 0.05, 0.1])
search_space.add_parameter(ParameterTraining.MINI_BATCH_SIZE, choice, options=[16, 32])

#Define parameters for document embeddings RNN
search_space.add_parameter(ParameterDocumentRNNEmbeddings.HIDDEN_SIZE, choice, options=[128, 256, 512])
search_space.add_parameter(ParameterDocumentRNNEmbeddings.DROPOUT, uniform, bounds=[0, 0.5])
search_space.add_parameter(ParameterDocumentRNNEmbeddings.EMBEDDINGS, choice, options=[WordEmbeddings('glove'), WordEmbeddings('en')])

#Define parameters for document embeddings Pool
search_space.add_parameter(ParameterDocumentPoolEmbeddings.EMBEDDINGS, choice, options=[WordEmbeddings('glove'), WordEmbeddings('en')])
search_space.add_parameter(ParameterDocumentPoolEmbeddings.POOLING, choice, options=['mean', 'max', 'min'])

#Define parameters for Transformers
search_space.add_parameter(ParameterTransformerDocumentEmbeddings.MODEL, choice, options=["bert-based-uncased", "distilbert-base-uncased"])
search_space.add_parameter(ParameterTransformerDocumentEmbeddings.BATCH_SIZE, choice, options=[32, 64])

#Pass the search space to the optimizer object
optimizer = optimizers.GeneticOptimizer(search_space=search_space, population_size=2)

#Create parameter selector object and optimize by passing the optimizer object to the function
param_selector = selectors.TextClassificationParamSelector(corpus=corpus, base_path='resources/hyperopt', optimizer=optimizer)
param_selector.optimize()
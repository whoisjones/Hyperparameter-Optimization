from GeneticParamOptimizer.hyperparameter import selectors, search_spaces, optimizers
import GeneticParamOptimizer.hyperparameter.parameters as param
from GeneticParamOptimizer.hyperparameter.utils import choice, uniform
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings, TransformerDocumentEmbeddings
from flair.datasets import TREC_6

# 1.) Define your corpus
#corpus = TREC_6()

# 2.) create an search space
search_space = search_spaces.TextClassifierSearchSpace()

# 3.) depending on your task add the respective parameters you want to optimize over

#Define your budget and optmization metric
search_space.add_budget(param.Budget.RUNS, 50)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)

#Depending on your downstream task, add embeddings and specify these with the respective Parameters below
search_space.add_parameter(param.TextClassifier.DOCUMENT_EMBEDDINGS, choice, options=[DocumentRNNEmbeddings,
                                                                                      DocumentPoolEmbeddings,
                                                                                      TransformerDocumentEmbeddings])
search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, choice, options=[0.01, 0.05, 0.1])
search_space.add_parameter(param.ModelTrainer.MINI_BATCH_SIZE, choice, options=[16, 32])

#Define parameters for document embeddings RNN
search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, choice, options=[128, 256, 512])
search_space.add_parameter(param.DocumentRNNEmbeddings.DROPOUT, uniform, bounds=[0, 0.5])
search_space.add_parameter(param.DocumentRNNEmbeddings.EMBEDDINGS, choice, options=[WordEmbeddings('glove'), WordEmbeddings('en')])

#Define parameters for document embeddings Pool
search_space.add_parameter(param.DocumentPoolEmbeddings.EMBEDDINGS, choice, options=[WordEmbeddings('glove'), WordEmbeddings('en')])
search_space.add_parameter(param.DocumentPoolEmbeddings.POOLING, choice, options=['mean', 'max', 'min'])

#Define parameters for Transformers
search_space.add_parameter(param.TransformerDocumentEmbeddings.MODEL, choice, options=["bert-based-uncased", "distilbert-base-uncased"])
search_space.add_parameter(param.TransformerDocumentEmbeddings.BATCH_SIZE, choice, options=[32, 64])

#Pass the search space to the optimizer object
optimizer = optimizers.GeneticOptimizer(search_space=search_space)

#Create parameter selector object and optimize by passing the optimizer object to the function
param_selector = selectors.TextClassificationParamSelector(corpus=corpus, base_path='resources/hyperopt', optimizer=optimizer)
param_selector.optimize()
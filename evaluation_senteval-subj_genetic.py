from FlairParamOptimizer import optimizers, search_spaces, selectors
import FlairParamOptimizer.parameters_for_user_guidance as param
from FlairParamOptimizer.sampling_functions import sampling_func
from flair.embeddings import DocumentPoolEmbeddings, DocumentRNNEmbeddings, TransformerDocumentEmbeddings
from flair.datasets import SENTEVAL_SUBJ
from torch.optim import SGD, Adam

# 1.) Define your corpus
corpus = SENTEVAL_SUBJ()

# 2.) create an search space
search_space = search_spaces.TextClassifierSearchSpace()

# 3.) depending on your task add the respective parameters you want to optimize over

#Define your budget and optmization metric
search_space.add_budget(param.Budget.TIME_IN_H, 24)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(15)

#Depending on your downstream task, add embeddings and specify these with the respective Parameters below
search_space.add_parameter(param.TextClassifier.DOCUMENT_EMBEDDINGS, sampling_func.choice, options=[DocumentRNNEmbeddings,
                                                                                                    DocumentPoolEmbeddings,
                                                                                                    TransformerDocumentEmbeddings])
search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, sampling_func.choice, options=[0.01, 0.05, 0.1])
search_space.add_parameter(param.ModelTrainer.MINI_BATCH_SIZE, sampling_func.choice, options=[16, 32, 64])
search_space.add_parameter(param.ModelTrainer.ANNEAL_FACTOR, sampling_func.choice, options=[0.25, 0.5])
search_space.add_parameter(param.ModelTrainer.OPTIMIZER, sampling_func.choice, options=[SGD, Adam])
search_space.add_parameter(param.Optimizer.WEIGHT_DECAY, sampling_func.choice, options=[1e-2, 0])


#Define parameters for document embeddings RNN
search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, sampling_func.choice, options=[128, 256, 512])
search_space.add_parameter(param.DocumentRNNEmbeddings.DROPOUT, sampling_func.uniform, bounds=[0, 0.5])
search_space.add_parameter(param.DocumentRNNEmbeddings.REPROJECT_WORDS, sampling_func.choice, options=[True, False])
search_space.add_parameter(param.DocumentRNNEmbeddings.WORD_EMBEDDINGS, sampling_func.choice, options=[['glove'], ['en'], ['en', 'glove']])

#Define parameters for document embeddings Pool
search_space.add_parameter(param.DocumentPoolEmbeddings.WORD_EMBEDDINGS, sampling_func.choice, options=[['glove'], ['en'], ['en', 'glove']])
search_space.add_parameter(param.DocumentPoolEmbeddings.POOLING, sampling_func.choice, options=['mean', 'max', 'min'])

#Define parameters for Transformers
search_space.add_parameter(param.TransformerDocumentEmbeddings.MODEL, sampling_func.choice, options=["bert-base-uncased", "distilbert-base-uncased"])
search_space.add_parameter(param.TransformerDocumentEmbeddings.BATCH_SIZE, sampling_func.choice, options=[16, 32, 64])

#Pass the search space to the optimizer object
optimizer = optimizers.GeneticOptimizer(search_space=search_space, population_size=8)

#Create parameter selector object and optimize by passing the optimizer object to the function
param_selector = selectors.TextClassificationParamSelector(corpus=corpus,
                                                           base_path='resources/evaluation-senteval-subj-genetic',
                                                           optimizer=optimizer,
                                                           search_space=search_space)
param_selector.optimize()
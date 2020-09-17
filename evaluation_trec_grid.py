from FlairParamOptimizer import search_strategies, search_spaces, orchestrator
import FlairParamOptimizer.parameter_listings.parameters_for_user_input as param
from flair.datasets import TREC_6
from torch.optim import SGD, Adam

# 1.) Define your corpus
corpus = TREC_6()

# 2.) create an search space
search_space = search_spaces.TextClassifierSearchSpace()
search_strategy = search_strategies.EvolutionarySearch(population_size=4, mutation_rate=1, cross_rate=1)

# 3.) depending on your task add the respective parameters you want to optimize over
search_space.add_budget(param.Budget.GENERATIONS, 3)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(1)

#Depending on your downstream task, add embeddings and specify these with the respective Parameters below
search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
search_space.add_parameter(param.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32, 64])
search_space.add_parameter(param.ModelTrainer.ANNEAL_FACTOR, options=[0.25, 0.5])
search_space.add_parameter(param.ModelTrainer.OPTIMIZER, options=[SGD, Adam])
search_space.add_parameter(param.Optimizer.WEIGHT_DECAY, options=[1e-2, 0])

#Define parameters for document embeddings RNN
search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
search_space.add_parameter(param.DocumentRNNEmbeddings.DROPOUT, options=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
search_space.add_parameter(param.DocumentRNNEmbeddings.REPROJECT_WORDS, options=[True, False])
search_space.add_parameter(param.DocumentRNNEmbeddings.WORD_EMBEDDINGS, options=[['glove'], ['en'], ['en', 'glove']])

#Define parameters for document embeddings Pool
search_space.add_parameter(param.DocumentPoolEmbeddings.WORD_EMBEDDINGS, options=[['glove'], ['en'], ['en', 'glove']])
search_space.add_parameter(param.DocumentPoolEmbeddings.POOLING, options=['mean', 'max', 'min'])

#Define parameters for Transformers
#search_space.add_parameter(param.TransformerDocumentEmbeddings.MODEL, sampling_func.choice, options=["bert-base-uncased", "distilbert-base-uncased"])
#search_space.add_parameter(param.TransformerDocumentEmbeddings.BATCH_SIZE, sampling_func.choice, options=[16, 32, 64])

search_strategy.make_configurations(search_space)

orchestrator = orchestrator.Orchestrator(corpus=corpus,
                                         base_path='resources/evaluation-trec-grid-DRAFT',
                                         search_space=search_space,
                                         search_strategy=search_strategy)

orchestrator.optimize()
from FlairParamOptimizer import search_strategies, search_spaces, orchestrator
import FlairParamOptimizer.parameter_listings.parameters_for_user_input as param
from FlairParamOptimizer.sampling_functions import sampling_func
from flair.embeddings import WordEmbeddings

from flair.data import Corpus
from flair.datasets import UD_ENGLISH

corpus: Corpus = UD_ENGLISH().downsample(0.5)

search_space = search_spaces.SequenceTaggerSearchSpace()

search_space.add_tag_type("pos")

search_space.add_budget(param.Budget.TIME_IN_H, 24)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(20)

search_space.add_parameter(param.SequenceTagger.WORD_EMBEDDINGS, sampling_func.choice, options=[
                                                                            [WordEmbeddings('glove')],
                                                                            [WordEmbeddings('en')],
                                                                            [WordEmbeddings('en'),
                                                                             WordEmbeddings('glove')]
                                                                            ])
search_space.add_parameter(param.SequenceTagger.HIDDEN_SIZE, sampling_func.choice, options=[128, 256, 512])
search_space.add_parameter(param.SequenceTagger.DROPOUT, sampling_func.uniform, bounds=[0, 0.5])
search_space.add_parameter(param.SequenceTagger.WORD_DROPOUT, sampling_func.choice, options=[0, 0.01, 0.05, 0.1])
search_space.add_parameter(param.SequenceTagger.RNN_LAYERS, sampling_func.choice, options=[2, 3, 4, 5])
search_space.add_parameter(param.SequenceTagger.USE_RNN, sampling_func.choice, options=[True, False])
search_space.add_parameter(param.SequenceTagger.REPROJECT_EMBEDDINGS, sampling_func.choice, options=[True, False])

optimizer = search_strategies.RandomSearch(search_space=search_space)

param_selector = orchestrator.SequenceTaggerOrchestrator(corpus=corpus,
                                                         base_path="resources/evaluation_ud-english_random",
                                                         search_space=search_space,
                                                         optimizer=optimizer)

param_selector.optimize()

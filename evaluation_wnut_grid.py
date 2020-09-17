from FlairParamOptimizer import search_strategies, search_spaces, selectors
import FlairParamOptimizer.parameter_listings.parameters_for_user_input as param
from flair.embeddings import WordEmbeddings

from flair.data import Corpus
from flair.datasets import WNUT_17

corpus: Corpus = WNUT_17().downsample(0.1)

search_space = search_spaces.SequenceTaggerSearchSpace()
search_strategy = search_strategies.RandomSearch()

search_space.add_tag_type("ner")

search_space.add_budget(param.Budget.TIME_IN_H, 1)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(1)

search_space.add_parameter(param.SequenceTagger.HIDDEN_SIZE, options=[128, 256, 512])
search_space.add_parameter(param.SequenceTagger.DROPOUT, options=[0, 0.5])
search_space.add_parameter(param.SequenceTagger.WORD_DROPOUT, options=[0, 0.01, 0.05, 0.1])
search_space.add_parameter(param.SequenceTagger.RNN_LAYERS, options=[2, 3, 4, 5])
search_space.add_parameter(param.SequenceTagger.USE_RNN, options=[True, False])
search_space.add_parameter(param.SequenceTagger.REPROJECT_EMBEDDINGS, options=[True, False])
search_space.add_parameter(param.SequenceTagger.WORD_EMBEDDINGS, options=[[WordEmbeddings('glove')],
                                                                          [WordEmbeddings('en')],
                                                                          [WordEmbeddings('en'), WordEmbeddings('glove')]])

search_strategy.make_configurations(search_space)

param_selector = selectors.SequenceTaggerParamSelector(corpus=corpus,
                                                       base_path="resources/evaluation_wnut_grid",
                                                       search_space=search_space,
                                                       optimizer=optimizer)

param_selector.optimize()

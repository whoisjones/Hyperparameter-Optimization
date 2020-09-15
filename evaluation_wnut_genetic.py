from FlairParamOptimizer import optimizers, search_spaces, selectors
import FlairParamOptimizer.parameters_for_user_guidance as param
from FlairParamOptimizer.sampling_functions import sampling_func
from flair.embeddings import WordEmbeddings

from flair.data import Corpus
from flair.datasets import WNUT_17

corpus: Corpus = WNUT_17()

search_space = search_spaces.SequenceTaggerSearchSpace()

search_space.add_tag_type("ner")

search_space.add_budget(param.Budget.TIME_IN_H, 24)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(25)

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

optimizer = optimizers.GeneticOptimizer(search_space=search_space)

param_selector = selectors.SequenceTaggerParamSelector(corpus=corpus,
                                                       base_path="resources/evaluation_wnut_genetic",
                                                       search_space=search_space,
                                                       optimizer=optimizer)

param_selector.optimize()

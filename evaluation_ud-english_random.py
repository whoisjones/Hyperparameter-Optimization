from GeneticParamOptimizer.hyperparameter import selectors, search_spaces, optimizers
import GeneticParamOptimizer.hyperparameter.parameters as param
from GeneticParamOptimizer.hyperparameter.sampling_functions import func
from flair.embeddings import WordEmbeddings, FlairEmbeddings

from flair.data import Corpus
from flair.datasets import UD_ENGLISH

corpus: Corpus = UD_ENGLISH().downsample(0.5)

search_space = search_spaces.SequenceTaggerSearchSpace()

search_space.add_tag_type("pos")

search_space.add_budget(param.Budget.TIME_IN_H, 24)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training(25)

search_space.add_parameter(param.SequenceTagger.WORD_EMBEDDINGS, func.choice, options=[
                                                                            [WordEmbeddings('glove')],
                                                                            [WordEmbeddings('en')],
                                                                            [WordEmbeddings('en'),
                                                                             WordEmbeddings('glove')]
                                                                            ])
search_space.add_parameter(param.SequenceTagger.HIDDEN_SIZE, func.choice, options=[128, 256, 512])
search_space.add_parameter(param.SequenceTagger.DROPOUT, func.uniform, bounds=[0, 0.5])
search_space.add_parameter(param.SequenceTagger.WORD_DROPOUT, func.choice, options=[0, 0.01, 0.05, 0.1])
search_space.add_parameter(param.SequenceTagger.RNN_LAYERS, func.choice, options=[2,3,4,5])
search_space.add_parameter(param.SequenceTagger.USE_RNN, func.choice, options=[True, False])
search_space.add_parameter(param.SequenceTagger.REPROJECT_EMBEDDINGS, func.choice, options=[True, False])

optimizer = optimizers.RandomSearchOptimizer(search_space=search_space)

param_selector = selectors.SequenceTaggerParamSelector(corpus=corpus,
                                                       base_path="resources/evaluation_ud-english_random",
                                                       search_space=search_space,
                                                       optimizer=optimizer)

param_selector.optimize()

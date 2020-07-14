from GeneticParamOptimizer.hyperparameter import selectors, search_spaces, optimizers
import GeneticParamOptimizer.hyperparameter.parameters as param
from GeneticParamOptimizer.hyperparameter.utils import choice, uniform
from flair.embeddings import WordEmbeddings, FlairEmbeddings

from flair.data import Corpus
from flair.datasets import UD_ENGLISH

search_space = search_spaces.SequenceTaggerSearchSpace()

#corpus: Corpus = UD_ENGLISH().downsample(0.1)

search_space.add_tag_type("pos")

search_space.add_budget(param.Budget.RUNS, 50)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)

search_space.add_parameter(param.SequenceTagger.EMBEDDINGS, choice, options=[
                                                                            [WordEmbeddings('glove')],
                                                                            [WordEmbeddings('en')],
                                                                            [WordEmbeddings('glove'),
                                                                             FlairEmbeddings('news-forward'),
                                                                             FlairEmbeddings('news-backward')]
                                                                            ])
search_space.add_parameter(param.SequenceTagger.HIDDEN_SIZE, choice, options=[128, 256, 512])
search_space.add_parameter(param.SequenceTagger.DROPOUT, uniform, bounds=[0, 0.5])
search_space.add_parameter(param.SequenceTagger.RNN_LAYERS, choice, options=[2,3,4])

optimizer = optimizers.GridSearchOptimizer(search_space=search_space)

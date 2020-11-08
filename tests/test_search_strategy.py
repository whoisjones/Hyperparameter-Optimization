
import collections
from FlairParamOptimizer import search_spaces, search_strategies
import FlairParamOptimizer.parameter_listings.parameters_for_user_input as param

def build_correct_setup():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.RandomSearch()
    search_space.add_budget(param.Budget.GENERATIONS, 10)
    search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
    search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_parameter(param.DocumentRNNEmbeddings.WORD_EMBEDDINGS, options=[['glove'], ['en'], ['en', 'glove']])
    return search_space, search_strategy

def build_correct_evolutionary_setup():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.EvolutionarySearch()
    search_space.add_budget(param.Budget.GENERATIONS, 10)
    search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
    search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])

    search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_parameter(param.DocumentRNNEmbeddings.WORD_EMBEDDINGS,
                               options=[['glove'], ['en'], ['en', 'glove']])

    search_space.add_parameter(param.DocumentPoolEmbeddings.WORD_EMBEDDINGS,
                               options=[['glove'], ['en'], ['en', 'glove']])
    search_space.add_parameter(param.DocumentPoolEmbeddings.POOLING, options=['mean', 'max', 'min'])
    return search_space, search_strategy

def test_type_of_make_configurations():
    search_space, search_strategy = build_correct_setup()
    search_strategy.make_configurations(search_space)
    assert type(search_space.training_configurations.configurations) == list
    assert len(search_space.training_configurations.configurations) == 27

def test_evolutionary_configuration():
    search_space, search_strategy = build_correct_evolutionary_setup()
    search_strategy.make_configurations(search_space)
    assert len(search_space.training_configurations.configurations) == 12
    list_of_embeddings_from_configurations = [item.get("document_embeddings").__name__ for item in search_space.training_configurations.configurations]
    occurrences = collections.Counter(list_of_embeddings_from_configurations)
    assert list(occurrences.values()) == [6,6]
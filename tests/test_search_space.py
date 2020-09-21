import pytest
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

def test_parameters_are_missing():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.RandomSearch()
    search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
    search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_budget(param.Budget.RUNS, 5)
    with pytest.raises(AttributeError):
        search_space.check_completeness(search_strategy.search_strategy_name)

def test_budget_is_missing():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.RandomSearch()
    search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
    search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
    with pytest.raises(AttributeError):
        search_space.check_completeness(search_strategy.search_strategy_name)

def test_optimization_value_is_missing():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.RandomSearch()
    search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
    search_space.add_budget(param.Budget.RUNS, 10)
    with pytest.raises(AttributeError):
        search_space.check_completeness(search_strategy.search_strategy_name)

def test_evaluation_metric_is_missing():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.RandomSearch()
    search_space.add_budget(param.Budget.RUNS, 10)
    search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
    with pytest.raises(AttributeError):
        search_space.check_completeness(search_strategy.search_strategy_name)

def test_word_embeddings_are_missing():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.RandomSearch()
    search_space.add_budget(param.Budget.GENERATIONS, 10)
    search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
    search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    with pytest.raises(KeyError):
        search_space.check_completeness(search_strategy.search_strategy_name)

def test_budget_type_switch_if_search_strategy_does_not_match():
    search_space = search_spaces.TextClassifierSearchSpace()
    search_strategy = search_strategies.RandomSearch()
    search_space.add_budget(param.Budget.GENERATIONS, 10)
    search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
    search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_parameter(param.DocumentRNNEmbeddings.WORD_EMBEDDINGS, options=[['glove'], ['en'], ['en', 'glove']])
    search_space.check_completeness(search_strategy.search_strategy_name)
    assert search_space.budget.budget_type == param.Budget.RUNS.value

def test_parameter_storage():
    search_space, search_strategy = build_correct_setup()
    assert hasattr(search_space.parameter_storage, "DocumentRNNEmbeddings")
    assert hasattr(search_space.parameter_storage, "GeneralParameters")
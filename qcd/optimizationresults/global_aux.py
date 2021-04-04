from typing import List
from . import OptimizationResult
from ..typings.configurations import OptimalConfigurations


def build_optimization_result(
        optimal_configurations: OptimalConfigurations) -> OptimizationResult:
    """ Create a Optimization Result subtype class based on its structure """

    configurations = optimal_configurations.get('configurations')
    if configurations is None:
        raise ValueError('configurations MUST be specified')

    return OptimizationResult(optimal_configurations)


def get_max_result(results: List[OptimalConfigurations]) -> OptimalConfigurations:
    """ Return the greater probability from a given list of results """
    if results is None or len(results) <= 0:
        raise ValueError('results required at least one result')
    max_results = results.pop()
    if len(results) == 1:
        return max_results
    return check_if_the_next_element_is_greater_than_current_max(max_results, results)


def check_if_the_next_element_is_greater_than_current_max(
        max_result: OptimalConfigurations,
        results: List[OptimalConfigurations]) -> OptimalConfigurations:
    if len(results) <= 1:
        return max_result

    next_result = results.pop()
    tmp_max = max_result
    for idx, (prob_max, prob_next) in enumerate(zip(tmp_max['probabilities'], next_result['probabilities'])):
        if prob_max < prob_next:
            tmp_max['best_algorithm'][idx] = next_result['best_algorithm'][idx]
            tmp_max['probabilities'][idx] = next_result['probabilities'][idx]
            tmp_max['configurations'][idx] = next_result['configurations'][idx]
            tmp_max['number_calls_made'][idx] = next_result['number_calls_made'][idx]
    return check_if_the_next_element_is_greater_than_current_max(tmp_max, results)

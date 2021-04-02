from . import OptimizationResult
from ..typings.configurations import OptimalConfigurations, TheoreticalOptimalConfigurations
from ..typings import TheoreticalOptimizationSetup
from ..optimizations import TheoreticalOneShotOptimization
from typing import Union, List, cast, Dict
from .aux import _build_probabilities_matrix, _build_probabilities_matrix_legacy


class OneShotOptimizationResult(OptimizationResult):
    """ Representation of the One Shot Optimization Result """

    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOptimalConfigurations:
        """ Returns the theoretical results for a One Shot Channel """
        theoretical_optimization = TheoreticalOneShotOptimization(optimization_setup)
        return theoretical_optimization.compute_theoretical_optimal_results()

    def _build_probabilities_matrix(self,
                                    result: Union[OptimalConfigurations,
                                                  TheoreticalOptimalConfigurations]) -> List[List[float]]:
        if isinstance(result['eta_pairs'][0][0], float):
            return _build_probabilities_matrix(cast(OptimalConfigurations, result))
        if isinstance(result['eta_pairs'][0][0], str):
            return cast(List[List[float]], _build_probabilities_matrix_legacy(cast(Dict, result)))
        raise ValueError('Bad input results')

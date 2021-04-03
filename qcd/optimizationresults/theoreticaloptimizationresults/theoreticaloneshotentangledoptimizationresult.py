from . import TheoreticalOneShotOptimizationResult
from ...typings.configurations import OptimalConfigurations, TheoreticalOneShotEntangledOptimalConfigurations
from ...typings import TheoreticalOptimizationSetup
from ...optimizations import TheoreticalOneShotEntangledOptimization
from typing import List, cast
from ..aux import _init_matrix
import numpy as np


class TheoreticalOneShotEntangledOptimizationResult(TheoreticalOneShotOptimizationResult):
    """ Representation of the theoretical One Shot Entangled Optimization Result """

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        super().__init__(optimal_configurations)
        self._improvement_matrix = self._build_improvement_matrix(
            cast(TheoreticalOneShotEntangledOptimalConfigurations, self._theoretical_result))

    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOneShotEntangledOptimalConfigurations:
        """ Returns the theoretical results for a One Shot Entangled Channel """
        theoretical_optimization = TheoreticalOneShotEntangledOptimization(optimization_setup)
        return theoretical_optimization.compute_theoretical_optimal_results()

    def _build_improvement_matrix(
            self,
            optimal_configurations: TheoreticalOneShotEntangledOptimalConfigurations) -> List[List[float]]:
        """ Returns the correspondent matrix of the improvement values """
        sorted_etas, matrix = _init_matrix(optimal_configurations)
        self._assign_improvement(optimal_configurations, sorted_etas, matrix)
        return matrix

    def _assign_improvement(self,
                            result: TheoreticalOneShotEntangledOptimalConfigurations,
                            sorted_etas: List[float],
                            matrix: np.array):
        for idx, improvement in enumerate(result['improvements']):
            ind_0 = sorted_etas.index(result['eta_pairs'][idx][0])
            ind_1 = (len(sorted_etas) - 1) - sorted_etas.index(result['eta_pairs'][idx][1])
            matrix[ind_1, ind_0] = improvement

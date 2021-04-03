from . import TheoreticalOneShotOptimizationResult
from ...typings.configurations import TheoreticalOneShotEntangledOptimalConfigurations
from ...typings import TheoreticalOptimizationSetup
from ...optimizations import TheoreticalOneShotEntangledOptimization
from ..aux import build_improvement_matrix


class TheoreticalOneShotEntangledOptimizationResult(TheoreticalOneShotOptimizationResult):
    """ Representation of the theoretical One Shot Entangled Optimization Result """

    def __init__(self, number_etas: int) -> None:
        super().__init__(number_etas)
        self._improvement_matrix = build_improvement_matrix(self.__theoretical_result)

    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOneShotEntangledOptimalConfigurations:
        """ Returns the theoretical results for a One Shot Entangled Channel """
        theoretical_optimization = TheoreticalOneShotEntangledOptimization(optimization_setup)
        self.__theoretical_result = theoretical_optimization.compute_theoretical_optimal_results()
        return self.__theoretical_result

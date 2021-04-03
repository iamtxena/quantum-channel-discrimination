from . import TheoreticalOneShotOptimizationResult
from ..typings.configurations import TheoreticalOneShotEntangledOptimalConfigurations
from ..typings import TheoreticalOptimizationSetup
from ..optimizations import TheoreticalOneShotEntangledOptimization


class TheoreticalOneShotEntangledOptimizationResult(TheoreticalOneShotOptimizationResult):
    """ Representation of the theoretical One Shot Entangled Optimization Result """

    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOneShotEntangledOptimalConfigurations:
        """ Returns the theoretical results for a One Shot Entangled Channel """
        theoretical_optimization = TheoreticalOneShotEntangledOptimization(optimization_setup)
        return theoretical_optimization.compute_theoretical_optimal_results()

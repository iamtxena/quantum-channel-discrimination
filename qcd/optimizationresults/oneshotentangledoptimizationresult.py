from . import OneShotOptimizationResult
from ..typings.configurations import TheoreticalOptimalConfigurations
from ..typings import TheoreticalOptimizationSetup
from ..optimizations import TheoreticalOneShotEntangledOptimization


class OneShotEntangledOptimizationResult(OneShotOptimizationResult):
    """ Representation of the One Shot Entangled Optimization Result """

    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOptimalConfigurations:
        """ Returns the theoretical results for a One Shot Entangled Channel """
        theoretical_optimization = TheoreticalOneShotEntangledOptimization(optimization_setup)
        return theoretical_optimization.compute_theoretical_optimal_results()

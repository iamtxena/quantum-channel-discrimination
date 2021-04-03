from . import TheoreticalOptimizationResult
from ...typings import TheoreticalOptimizationSetup
from ...optimizations import TheoreticalOneShotOptimization
from ...typings.configurations import TheoreticalOneShotOptimalConfigurations


class TheoreticalOneShotOptimizationResult(TheoreticalOptimizationResult):
    """ Representation of the theoretical optimization result for a One Shot damping channel """

    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOneShotOptimalConfigurations:
        """ Returns the theoretical results for a One Shot Channel """
        theoretical_optimization = TheoreticalOneShotOptimization(optimization_setup)
        return theoretical_optimization.compute_theoretical_optimal_results()

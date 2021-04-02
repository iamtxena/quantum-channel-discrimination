from . import OptimizationResults
from typing import Optional
from ..typings.configurations import OptimalConfigurations, TheoreticalOptimalConfigurations
from ..typings import TheoreticalOptimizationSetup
from ..optimizations import TheoreticalOneShotOptimization


class OneShotOptimizationResults(OptimizationResults):
    """ Representation of the One Shot Optimization Results """

    def __init__(self, optimal_configurations: Optional[OptimalConfigurations] = None) -> None:
        super().__init__(optimal_configurations)

    def _compute_theoretical_optimal_results(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOptimalConfigurations:
        """ Returns the theoretical results for a One Shot Channel """
        theoretical_optimization = TheoreticalOneShotOptimization(optimization_setup)
        return theoretical_optimization.compute_theoretical_optimal_results()

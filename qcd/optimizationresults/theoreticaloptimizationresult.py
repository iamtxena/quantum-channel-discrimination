from abc import abstractmethod
from ..typings.configurations import OptimalConfigurations
from ..typings import TheoreticalOptimizationSetup
from . import OptimizationResult


class TheoreticalOptimizationResult(OptimizationResult):
    """ Generic class acting as an interface for the theoretical optimization result of any channel """

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        theoretical_result = self._prepare_etas_and_compute_theoretical_result(optimal_configurations)
        super().__init__(theoretical_result)

    def _prepare_etas_and_compute_theoretical_result(
            self,
            optimal_configurations: OptimalConfigurations) -> OptimalConfigurations:
        eta_pairs = optimal_configurations.get('eta_pairs')
        if eta_pairs is None:
            raise ValueError('eta_pairs is mandatory')
        theoretical_result = self._compute_theoretical_optimal_result(
            {'eta_pairs': eta_pairs})
        return theoretical_result

    @abstractmethod
    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> OptimalConfigurations:
        pass

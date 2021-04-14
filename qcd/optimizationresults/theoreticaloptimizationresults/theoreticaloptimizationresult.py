from abc import abstractmethod
from qcd.optimizationresults.aux import get_theoretical_optimization_setup_from_number_of_etas
from .. import OptimizationResult
from ...typings import TheoreticalOptimizationSetup
from ...typings.configurations import OptimalConfigurations


class TheoreticalOptimizationResult(OptimizationResult):
    """ Generic class acting as an interface for the theoretical optimization result of any channel """

    def __init__(self, number_etas: int) -> None:
        self._theoretical_result = self._prepare_etas_and_compute_theoretical_result(
            number_etas=number_etas)
        super().__init__(self._theoretical_result)

    def _prepare_etas_and_compute_theoretical_result(
            self,
            number_etas: int) -> OptimalConfigurations:
        theoretical_result = self._compute_theoretical_optimal_result(
            get_theoretical_optimization_setup_from_number_of_etas(eta_partitions=number_etas))
        return theoretical_result

    @abstractmethod
    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> OptimalConfigurations:
        pass

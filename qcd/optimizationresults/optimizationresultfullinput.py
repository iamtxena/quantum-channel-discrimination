from . import OptimizationResult
from typing import Any, List
from ..typings.configurations import OptimalConfigurations
from .aux import build_probabilities_matrix


class OptimizationResultFullInput(OptimizationResult):
    """ Generic class acting as an interface for any Optimization Result """

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        self._two_eta_configurations = self._convert_input_group_eta_configurations_into_two_etas(
            optimal_configurations)
        self._probabilities_matrices = [build_probabilities_matrix(two_eta_configuration)
                                        for two_eta_configuration in self._two_eta_configurations]
        self._amplitudes_matrices: List[Any] = []

    @property
    def amplitudes_matrix(self) -> List[List[float]]:
        raise ValueError('No amplitudes matrix in a Full Input Optimization Results')

    @ property
    def amplitudes_matrices(self) -> List[List[List[float]]]:
        raise ValueError('No amplitudes matrix in a Full Input Optimization Results')

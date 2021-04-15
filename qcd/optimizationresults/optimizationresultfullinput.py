from . import OptimizationResult
from typing import List
from ..typings.configurations import OptimalConfigurations
from .aux import build_probabilities_matrix


class OptimizationResultFullInput(OptimizationResult):
    """ Generic class acting as an interface for any Optimization Result """

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        self._probabilities_matrix = build_probabilities_matrix(optimal_configurations)
        self._amplitudes_matrix = []

    @ property
    def probabilities_matrix(self) -> List[List[float]]:
        return self._probabilities_matrix

    @ property
    def amplitudes_matrix(self) -> List[List[float]]:
        raise ValueError('No amplitudes matrix in a Full Input Optimization Results')

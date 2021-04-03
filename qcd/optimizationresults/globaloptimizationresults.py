from abc import ABC
from typing import Optional, List, Union, cast
from ..typings.configurations import OptimalConfigurations
from .aux import (load_result_from_file, plot_one_result)
from .global_aux import build_optimization_result
from . import TheoreticalResult, TheoreticalOneShotOptimizationResult, TheoreticalOneShotEntangledOptimizationResult


class GlobalOptimizationResults(ABC):
    """ Global class to load, process and plot any Optimization Results """

    @staticmethod
    def load_results(file_names: Union[str, List[str]], path: Optional[str] = ""):
        """
          1. Load results from file for all file names
          2. build probability matrices for each loaded result
          3. build amplitude matrix for each loaded result
        """
        names = file_names
        if not isinstance(names, List):
            names = [cast(str, file_names)]

        results = [load_result_from_file(file_name, path) for file_name in names]
        return GlobalOptimizationResults(results)

    def __init__(self, optimal_configurations: Union[OptimalConfigurations, List[OptimalConfigurations]]) -> None:
        optimal_results = optimal_configurations
        if not isinstance(optimal_configurations, List):
            optimal_results = [optimal_configurations]
        optimal_results = cast(List[OptimalConfigurations], optimal_results)
        self._optimization_results = [build_optimization_result(optimal_result) for optimal_result in optimal_results]
        self._build_all_theoretical_optimizations_results(optimal_results[0])

    def _build_all_theoretical_optimizations_results(self, optimal_configurations: OptimalConfigurations) -> None:
        """ Build the theoretical optimization result for each damping channel supported """
        self._theoretical_results: TheoreticalResult = {
            "one_shot": TheoreticalOneShotOptimizationResult(optimal_configurations),
            "one_shot_side_entanglement": TheoreticalOneShotEntangledOptimizationResult(optimal_configurations),
        }

    def plot_probabilities(self,
                           results_index: int,
                           title: str = 'Probabilities from simulation',
                           bar_label: str = 'Probabilities value',
                           vmin: float = 0.0,
                           vmax: float = 1.0) -> None:
        """ Plot probabilities analysis """
        plot_one_result(self._optimization_results[results_index].probabilities_matrix, title, bar_label, vmin, vmax)

    def plot_theoretical_probabilities(self,
                                       title: str = 'Probabilities from theory',
                                       bar_label: str = 'Probabilities value',
                                       vmin: float = 0.0,
                                       vmax: float = 1.0) -> None:
        """ Plot theoretical probabilities analysis """
        plot_one_result(self._theoretical_results['one_shot'], title, bar_label, vmin, vmax)

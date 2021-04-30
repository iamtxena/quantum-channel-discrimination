from qcd.circuits.aux import set_only_eta_groups
from typing import Optional, List, Union, cast, Dict
from ...typings.configurations import OptimalConfigurations
from ..aux import load_result_from_file
from .global_aux import build_optimization_result_full_input
from . import GlobalOptimizationResults


class GlobalOptimizationResultsFullInput(GlobalOptimizationResults):
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
        return GlobalOptimizationResultsFullInput(results)

    def __init__(self, optimal_configurations: Union[OptimalConfigurations, List[OptimalConfigurations]]) -> None:
        self._optimal_configurations = optimal_configurations if isinstance(
            optimal_configurations, List) else [optimal_configurations]
        self._optimal_configurations = set_only_eta_groups(cast(List[Dict], self._optimal_configurations))
        self._optimization_results = [build_optimization_result_full_input(
            optimal_result) for optimal_result in self._optimal_configurations]
        self._build_all_theoretical_optimizations_results(len(self._optimization_results[0].probabilities_matrix))

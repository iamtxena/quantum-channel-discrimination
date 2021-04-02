from abc import ABC
from typing import Optional, List, Union, cast
from ..typings.configurations import OptimalConfigurations
from .aux import load_result_from_file
from .global_aux import build_optimization_result


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

from abc import ABC
from typing import Optional, List, Union, cast
from ..typings.configurations import OptimalConfigurations
from .aux import load_result_from_file


class GlobalOptimizationResultsFullUniversal(ABC):
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
        return GlobalOptimizationResultsFullUniversal(results)

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        self._optimal_configurations = self._update_optimal_configurations(optimal_configurations)

    @property
    def optimal_configurations(self):
        return self._optimal_configurations

    def _update_optimal_configurations(self, configurations: OptimalConfigurations) -> OptimalConfigurations:
        """ Runs the circuit with the given optimal configurations computing the success average probability
            for each eta (and also the global), the selected eta for each measured state and finally the 
            upper and lower bound fidelities
        """
        return configurations


"""
class OptimalConfigurations(TypedDict, total=False):
    eta_groups: List[List[float]]
    best_algorithm: List[str]
    probabilities: List[float]
    configurations: List[ChannelConfiguration]
    number_calls_made: List[int]
    legacy: bool
    validated_probabilities: List[float]
    eta_probabilities: List[Tuple[float, float, float]]
    measured_states_eta_assignment: List[MeasuredStatesEtaAssignment]
    fidelities: List[Fidelities]
"""

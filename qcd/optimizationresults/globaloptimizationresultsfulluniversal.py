from abc import ABC
from qcd.dampingchannels.oneshotentangledfulluniversal import OneShotEntangledFullUniversalDampingChannel
from typing import Optional
from ..typings.configurations import OptimalConfigurations
from .aux import load_result_from_file


class GlobalOptimizationResultsFullUniversal(ABC):
    """ Global class to load, process and plot any Optimization Results """

    @staticmethod
    def load_results(file_name: str, str, path: Optional[str] = ""):
        """
          1. Load results from file for all file names
          2. build probability matrices for each loaded result
          3. build amplitude matrix for each loaded result
        """

        results = load_result_from_file(file_name, path)
        return GlobalOptimizationResultsFullUniversal(results)

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        self._optimal_configurations = optimal_configurations

    @property
    def optimal_configurations(self):
        return self._optimal_configurations

    @property
    def validated_optimal_configurations(self):
        return self._validated_optimal_configurations

    def validate_optimal_configurations(self,
                                        optimal_configurations: OptimalConfigurations,
                                        plays: Optional[int] = 10000) -> None:
        """ Runs the circuit with the given optimal configurations computing the success average probability
            for each eta (and also the global), the selected eta for each measured state and finally the
            upper and lower bound fidelities
        """
        validated_configurations = optimal_configurations
        validated_configurations['validated_probabilities'] = []
        validated_configurations['eta_probabilities'] = []
        validated_configurations['measured_states_eta_assignment'] = []
        validated_configurations['fidelities'] = []

        for configuration in optimal_configurations['configurations']:
            validated_configuration = OneShotEntangledFullUniversalDampingChannel.validate_optimal_configuration(
                configuration, plays
            )
            validated_configurations['validated_probabilities'].append(validated_configuration['validated_probability'])
            validated_configurations['eta_probabilities'].append(validated_configuration['etas_probability'])
            validated_configurations['measured_states_eta_assignment'].append(
                validated_configuration['measured_states_eta_assignment'])
            validated_configurations['fidelities'].append(validated_configuration['fidelities'])
        self._validated_optimal_configurations = validated_configurations

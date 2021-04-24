from abc import ABC
from typing import List
from ..typings.configurations import OptimalConfigurations
from .aux import build_probabilities_matrix, build_amplitudes_matrix, get_number_eta_pairs


class OptimizationResult(ABC):
    """ Generic class acting as an interface for any Optimization Result """

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:

        self._two_eta_configurations = self._convert_input_group_eta_configurations_into_two_etas(
            optimal_configurations)
        self._probabilities_matrices = [build_probabilities_matrix(two_eta_configuration)
                                        for two_eta_configuration in self._two_eta_configurations]
        self._amplitudes_matrices = [build_amplitudes_matrix(two_eta_configuration)
                                     for two_eta_configuration in self._two_eta_configurations]

    @property
    def probabilities_matrix(self) -> List[List[float]]:
        return self._probabilities_matrices[0]

    @property
    def amplitudes_matrix(self) -> List[List[float]]:
        return self._amplitudes_matrices[0]

    @property
    def probabilities_matrices(self) -> List[List[List[float]]]:
        return self._probabilities_matrices

    @property
    def amplitudes_matrices(self) -> List[List[List[float]]]:
        return self._amplitudes_matrices

    def _convert_input_group_eta_configurations_into_two_etas(self,
                                                              optimal_configurations: OptimalConfigurations
                                                              ) -> List[OptimalConfigurations]:
        eta_groups_length = len(optimal_configurations['eta_groups'][0])

        if eta_groups_length > 3:
            raise ValueError('Eta groups support only 2 or 3 elements')
        if eta_groups_length == 2:
            return [optimal_configurations]

        number_eta_pairs, _ = get_number_eta_pairs(optimal_configurations['eta_groups'])
        return self._get_two_eta_configurations(optimal_configurations, number_eta_pairs, eta_groups_length)

    def _get_two_eta_configurations(self, optimal_configurations: OptimalConfigurations,
                                    number_eta_pairs: int,
                                    eta_groups_length: int) -> List[OptimalConfigurations]:

        number_third_channels = int(len(optimal_configurations['eta_groups']) / number_eta_pairs)

        list_configs = []
        for i in range(number_eta_pairs):
            new_config = OptimalConfigurations({
                'eta_groups': optimal_configurations['eta_groups']
                [i * number_third_channels:(i + 1) * number_third_channels],
                'best_algorithm': optimal_configurations['best_algorithm']
                [i * number_third_channels:(i + 1) * number_third_channels],
                'probabilities': optimal_configurations['probabilities']
                [i * number_third_channels:(i + 1) * number_third_channels],
                'configurations': optimal_configurations['configurations']
                [i * number_third_channels:(i + 1) * number_third_channels],
                'number_calls_made': optimal_configurations['number_calls_made']
                [i * number_third_channels:(i + 1) * number_third_channels]
            })
            list_configs.append(new_config)

        return list_configs

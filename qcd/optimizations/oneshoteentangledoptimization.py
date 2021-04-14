from qcd.configurations.oneshotentangledconfiguration import OneShotEntangledConfiguration
from qcd.configurations.configuration import ChannelConfiguration
from qcd.typings.configurations import OptimalConfigurations
from typing import List, cast
from qcd.circuits import OneShotEntangledCircuit
from qcd.typings import OptimizationSetup
from . import OneShotOptimization
import math


class OneShotEntangledOptimization(OneShotOptimization):
    """ Representation of the One Shot EntangledChannel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)
        self._one_shot_circuit = OneShotEntangledCircuit()

    def _set_default_optimal_configurations(self, eta_group_idx_to_skip: List[int]) -> OptimalConfigurations:
        """ Return the optimal configurations set to default values for the indexes to be skipped """
        elements_to_skip = len(eta_group_idx_to_skip)

        configurations: List[ChannelConfiguration] = []
        for eta_group_idx in eta_group_idx_to_skip:
            one_configuration = OneShotEntangledConfiguration({'state_probability': 0,
                                                               'angle_rx': 0,
                                                               'angle_ry': 0,
                                                               'eta_group': self._eta_groups[eta_group_idx]})
            configurations.append(cast(ChannelConfiguration, one_configuration))

        return {'eta_groups': [],
                'best_algorithm': ['NA'] * elements_to_skip,
                'probabilities': [0] * elements_to_skip,
                'configurations': configurations,
                'number_calls_made': [0] * elements_to_skip}

    def _best_configuration_to_print(self, best_configuration: List[float]) -> str:
        return ("Parameters Found: state_probability = " + str(best_configuration[0]) +
                ", " + u"\u03D5" + "rx = " + str(int(math.degrees(best_configuration[1]))) + u"\u00B0" +
                ", " + u"\u03D5" + "ry = " + str(int(math.degrees(best_configuration[2]))) + u"\u00B0" +
                ''.join([", " + u"\u03B7" + f'{idx}' + " = " + str(int(math.degrees(global_eta))) + u"\u00B0"
                         for idx, global_eta in enumerate(self._global_eta_group)]))

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_group: List[float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotEntangledConfiguration({
            'state_probability': configuration[0],
            'angle_rx': configuration[1],
            'angle_ry': configuration[2],
            'eta_group': eta_group})

    def _cost_function(self, params: List[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
            defined in the optimization setup.
            Cost is computed as 1 (perfect probability) - average success probability for
            all the plays with the given configuration
            Returns the Cost (error probability).
        """
        configuration = OneShotEntangledConfiguration({
            'state_probability': params[0],
            'angle_rx': params[1],
            'angle_ry': params[2],
            'eta_group': self._global_eta_group})

        return - self._one_shot_circuit.compute_average_success_probability(configuration=configuration,
                                                                            plays=self._setup['plays'])

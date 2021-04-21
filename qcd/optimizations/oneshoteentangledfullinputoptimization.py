from qcd.configurations.oneshotentangledfullinputconfiguration import OneShotEntangledFullInputConfiguration
from qcd.configurations.configuration import ChannelConfiguration
from qcd.typings.configurations import OptimalConfigurations
from typing import List, Tuple, cast
from qcd.circuits import OneShotEntangledFullInputCircuit
from qcd.typings import OptimizationSetup
from . import OneShotEntangledOptimization
import numpy as np
import math


class OneShotEntangledFullInputOptimization(OneShotEntangledOptimization):
    """ Representation of the One Shot Entangled Full Input Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)
        self._add_initial_parameters_and_variable_bounds_to_optimization_setup()
        self._one_shot_circuit = OneShotEntangledFullInputCircuit()

    def _add_initial_parameters_and_variable_bounds_to_optimization_setup(self) -> None:
        """ Update the optimization setup with initial parameters and variable bounds """
        self._setup['initial_parameters'] = [0] * 8
        variable_bounds = [(0, 2 * np.pi)
                           for i in range(8)]
        self._setup['variable_bounds'] = cast(List[Tuple[float, float]], variable_bounds)

    def _set_default_optimal_configurations(self, eta_group_idx_to_skip: List[int]) -> OptimalConfigurations:
        """ Return the optimal configurations set to default values for the indexes to be skipped """
        elements_to_skip = len(eta_group_idx_to_skip)

        configurations: List[ChannelConfiguration] = []
        for eta_group_idx in eta_group_idx_to_skip:
            one_configuration = OneShotEntangledFullInputConfiguration({
                'angle_rx_input0': 0,
                'angle_ry_input0': 0,
                'angle_rx_input1': 0,
                'angle_ry_input1': 0,
                'angle_rx1': 0,
                'angle_ry1': 0,
                'angle_rx0': 0,
                'angle_ry0': 0,
                'eta_group': self._eta_groups[eta_group_idx]})
            configurations.append(cast(ChannelConfiguration, one_configuration))

        return {'eta_groups': [],
                'best_algorithm': ['NA'] * elements_to_skip,
                'probabilities': [0] * elements_to_skip,
                'configurations': configurations,
                'number_calls_made': [0] * elements_to_skip}

    def _best_configuration_to_print(self, best_configuration: List[float]) -> str:
        return ("Parameters Found: " +
                u"\u03D5" + "rx_input0 = " + str(int(math.degrees(best_configuration[0]))) + u"\u00B0" +
                ", " + u"\u03D5" + "ry_input0 = " + str(int(math.degrees(best_configuration[1]))) + u"\u00B0" +
                ", " + u"\u03D5" + "rx_input1 = " + str(int(math.degrees(best_configuration[2]))) + u"\u00B0" +
                ", " + u"\u03D5" + "ry_input1 = " + str(int(math.degrees(best_configuration[3]))) + u"\u00B0" +
                ", " + u"\u03D5" + "rx1 = " + str(int(math.degrees(best_configuration[4]))) + u"\u00B0" +
                ", " + u"\u03D5" + "ry1 = " + str(int(math.degrees(best_configuration[5]))) + u"\u00B0" +
                ", " + u"\u03D5" + "rx0 = " + str(int(math.degrees(best_configuration[6]))) + u"\u00B0" +
                ", " + u"\u03D5" + "ry0 = " + str(int(math.degrees(best_configuration[7]))) + u"\u00B0" +
                ''.join([", " + u"\u03B7" + f'{idx}' + " = " + str(int(math.degrees(global_eta))) + u"\u00B0"
                         for idx, global_eta in enumerate(self._global_eta_group)]))

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_group: List[float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotEntangledFullInputConfiguration({
            'angle_rx_input0': configuration[0],
            'angle_ry_input0': configuration[1],
            'angle_rx_input1': configuration[2],
            'angle_ry_input1': configuration[3],
            'angle_rx1': configuration[4],
            'angle_ry1': configuration[5],
            'angle_rx0': configuration[6],
            'angle_ry0': configuration[7],
            'eta_group': eta_group})

    def _cost_function(self, params: List[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
            defined in the optimization setup.
            Cost is computed as 1 (perfect probability) - average success probability for
            all the plays with the given configuration
            Returns the Cost (error probability).
        """
        configuration = OneShotEntangledFullInputConfiguration({
            'angle_rx_input0': params[0],
            'angle_ry_input0': params[1],
            'angle_rx_input1': params[2],
            'angle_ry_input1': params[3],
            'angle_rx1': params[4],
            'angle_ry1': params[5],
            'angle_rx0': params[6],
            'angle_ry0': params[7],
            'eta_group': self._global_eta_group})

        return - self._one_shot_circuit.compute_average_success_probability(configuration=configuration,
                                                                            plays=self._setup['plays'])

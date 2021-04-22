from qcd.configurations import OneShotEntangledUniversalConfiguration
from qcd.configurations.configuration import ChannelConfiguration
from qcd.typings.configurations import OptimalConfigurations
from typing import List, Tuple, cast
from qcd.circuits import OneShotEntangledUniversalCircuit
from qcd.typings import OptimizationSetup
from . import OneShotEntangledFullInputOptimization
import numpy as np
import math


class OneShotEntangledUniversalOptimization(OneShotEntangledFullInputOptimization):
    """ Representation of the One Shot Entangled Full Input Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)
        self._add_initial_parameters_and_variable_bounds_to_optimization_setup()
        self._one_shot_circuit = OneShotEntangledUniversalCircuit()

    def _add_initial_parameters_and_variable_bounds_to_optimization_setup(self) -> None:
        """ Update the optimization setup with initial parameters and variable bounds """
        self._setup['initial_parameters'] = [0] * (4 + 8 * 3)
        variable_bounds = [(0, 2 * np.pi)
                           for i in range(4)]
        for i in range(8):
            variable_bounds += [(0, np.pi)]  # theta
            variable_bounds += [(0, 2 * np.pi)]  #  phi
            variable_bounds += [(0, 2 * np.pi)]  # lambda
        self._setup['variable_bounds'] = cast(List[Tuple[float, float]], variable_bounds)

    def _set_default_optimal_configurations(self, eta_group_idx_to_skip: List[int]) -> OptimalConfigurations:
        """ Return the optimal configurations set to default values for the indexes to be skipped """
        elements_to_skip = len(eta_group_idx_to_skip)

        configurations: List[ChannelConfiguration] = []
        for eta_group_idx in eta_group_idx_to_skip:
            one_configuration = OneShotEntangledUniversalConfiguration({
                'angle_rx_input0': 0,
                'angle_ry_input0': 0,
                'angle_rx_input1': 0,
                'angle_ry_input1': 0,
                'theta0': 0,
                'phi0': 0,
                'lambda0': 0,
                'theta1': 0,
                'phi1': 0,
                'lambda1': 0,
                'theta2': 0,
                'phi2': 0,
                'lambda2': 0,
                'theta3': 0,
                'phi3': 0,
                'lambda3': 0,
                'theta4': 0,
                'phi4': 0,
                'lambda4': 0,
                'theta5': 0,
                'phi5': 0,
                'lambda5': 0,
                'theta6': 0,
                'phi6': 0,
                'lambda6': 0,
                'theta7': 0,
                'phi7': 0,
                'lambda7': 0,
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
                ", " + u"\u03D5" + "theta0 = " + str(int(math.degrees(best_configuration[4]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi0 = " + str(int(math.degrees(best_configuration[5]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda0 = " + str(int(math.degrees(best_configuration[6]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta1 = " + str(int(math.degrees(best_configuration[7]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi1 = " + str(int(math.degrees(best_configuration[8]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda1 = " + str(int(math.degrees(best_configuration[9]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta2 = " + str(int(math.degrees(best_configuration[10]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi2 = " + str(int(math.degrees(best_configuration[11]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda2 = " + str(int(math.degrees(best_configuration[12]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta3 = " + str(int(math.degrees(best_configuration[13]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi3 = " + str(int(math.degrees(best_configuration[14]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda3 = " + str(int(math.degrees(best_configuration[15]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta4 = " + str(int(math.degrees(best_configuration[16]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi4 = " + str(int(math.degrees(best_configuration[17]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda4 = " + str(int(math.degrees(best_configuration[18]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta5 = " + str(int(math.degrees(best_configuration[19]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi5 = " + str(int(math.degrees(best_configuration[20]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda5 = " + str(int(math.degrees(best_configuration[21]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta6 = " + str(int(math.degrees(best_configuration[22]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi6 = " + str(int(math.degrees(best_configuration[23]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda6 = " + str(int(math.degrees(best_configuration[24]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta7 = " + str(int(math.degrees(best_configuration[25]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi7 = " + str(int(math.degrees(best_configuration[26]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda7 = " + str(int(math.degrees(best_configuration[27]))) + u"\u00B0" +
                ''.join([", " + u"\u03B7" + f'{idx}' + " = " + str(int(math.degrees(global_eta))) + u"\u00B0"
                         for idx, global_eta in enumerate(self._global_eta_group)]))

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_group: List[float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotEntangledUniversalConfiguration({
            'angle_rx_input0': configuration[0],
            'angle_ry_input0': configuration[1],
            'angle_rx_input1': configuration[2],
            'angle_ry_input1': configuration[3],
            'theta0': configuration[4],
            'phi0': configuration[5],
            'lambda0': configuration[6],
            'theta1': configuration[7],
            'phi1': configuration[8],
            'lambda1': configuration[9],
            'theta2': configuration[10],
            'phi2': configuration[11],
            'lambda2': configuration[12],
            'theta3': configuration[13],
            'phi3': configuration[14],
            'lambda3': configuration[15],
            'theta4': configuration[16],
            'phi4': configuration[17],
            'lambda4': configuration[18],
            'theta5': configuration[19],
            'phi5': configuration[20],
            'lambda5': configuration[21],
            'theta6': configuration[22],
            'phi6': configuration[23],
            'lambda6': configuration[24],
            'theta7': configuration[25],
            'phi7': configuration[26],
            'lambda7': configuration[27],
            'eta_group': eta_group})

    def _cost_function(self, params: List[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
            defined in the optimization setup.
            Cost is computed as 1 (perfect probability) - average success probability for
            all the plays with the given configuration
            Returns the Cost (error probability).
        """
        configuration = OneShotEntangledUniversalConfiguration({
            'angle_rx_input0': params[0],
            'angle_ry_input0': params[1],
            'angle_rx_input1': params[2],
            'angle_ry_input1': params[3],
            'theta0': params[4],
            'phi0': params[5],
            'lambda0': params[6],
            'theta1': params[7],
            'phi1': params[8],
            'lambda1': params[9],
            'theta2': params[10],
            'phi2': params[11],
            'lambda2': params[12],
            'theta3': params[13],
            'phi3': params[14],
            'lambda3': params[15],
            'theta4': params[16],
            'phi4': params[17],
            'lambda4': params[18],
            'theta5': params[19],
            'phi5': params[20],
            'lambda5': params[21],
            'theta6': params[22],
            'phi6': params[23],
            'lambda6': params[24],
            'theta7': params[25],
            'phi7': params[26],
            'lambda7': params[27],
            'eta_group': self._global_eta_group})

        return - self._one_shot_circuit.compute_average_success_probability(configuration=configuration,
                                                                            plays=self._setup['plays'])

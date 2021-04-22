from qcd.configurations import OneShotEntangledFullUniversalConfiguration
from qcd.configurations.configuration import ChannelConfiguration
from qcd.typings.configurations import OptimalConfigurations
from typing import List, Tuple, cast
from qcd.circuits import OneShotEntangledFullUniversalCircuit
from qcd.typings import OptimizationSetup
from . import OneShotEntangledUniversalOptimization
import numpy as np
import math


class OneShotEntangledFullUniversalOptimization(OneShotEntangledUniversalOptimization):
    """ Representation of the One Shot Entangled Full Input Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)
        self._add_initial_parameters_and_variable_bounds_to_optimization_setup()
        self._one_shot_circuit = OneShotEntangledFullUniversalCircuit()

    def _add_initial_parameters_and_variable_bounds_to_optimization_setup(self) -> None:
        """ Update the optimization setup with initial parameters and variable bounds """
        self._setup['initial_parameters'] = [0] * (8 * 3 + 8 * 3)
        variable_bounds = []
        for i in range(16):
            variable_bounds += [(0, np.pi)]  # theta
            variable_bounds += [(0, 2 * np.pi)]  #  phi
            variable_bounds += [(0, 2 * np.pi)]  # lambda
        self._setup['variable_bounds'] = cast(List[Tuple[float, float]], variable_bounds)

    def _set_default_optimal_configurations(self, eta_group_idx_to_skip: List[int]) -> OptimalConfigurations:
        """ Return the optimal configurations set to default values for the indexes to be skipped """
        elements_to_skip = len(eta_group_idx_to_skip)

        configurations: List[ChannelConfiguration] = []
        for eta_group_idx in eta_group_idx_to_skip:
            one_configuration = OneShotEntangledFullUniversalConfiguration({
                'input_theta0': 0,
                'input_phi0': 0,
                'input_lambda0': 0,
                'input_theta1': 0,
                'input_phi1': 0,
                'input_lambda1': 0,
                'input_theta2': 0,
                'input_phi2': 0,
                'input_lambda2': 0,
                'input_theta3': 0,
                'input_phi3': 0,
                'input_lambda3': 0,
                'input_theta4': 0,
                'input_phi4': 0,
                'input_lambda4': 0,
                'input_theta5': 0,
                'input_phi5': 0,
                'input_lambda5': 0,
                'input_theta6': 0,
                'input_phi6': 0,
                'input_lambda6': 0,
                'input_theta7': 0,
                'input_phi7': 0,
                'input_lambda7': 0,
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
                u"\u03D5" + "input_theta0 = " + str(int(math.degrees(best_configuration[0]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi0 = " + str(int(math.degrees(best_configuration[1]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda0 = " + str(int(math.degrees(best_configuration[2]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_theta1 = " + str(int(math.degrees(best_configuration[3]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi1 = " + str(int(math.degrees(best_configuration[4]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda1 = " + str(int(math.degrees(best_configuration[5]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_theta2 = " + str(int(math.degrees(best_configuration[6]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi2 = " + str(int(math.degrees(best_configuration[7]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda2 = " + str(int(math.degrees(best_configuration[8]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_theta3 = " + str(int(math.degrees(best_configuration[9]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi3 = " + str(int(math.degrees(best_configuration[10]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda3 = " + str(int(math.degrees(best_configuration[11]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_theta4 = " + str(int(math.degrees(best_configuration[12]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi4 = " + str(int(math.degrees(best_configuration[13]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda4 = " + str(int(math.degrees(best_configuration[14]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_theta5 = " + str(int(math.degrees(best_configuration[15]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi5 = " + str(int(math.degrees(best_configuration[16]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda5 = " + str(int(math.degrees(best_configuration[17]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_theta6 = " + str(int(math.degrees(best_configuration[18]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi6 = " + str(int(math.degrees(best_configuration[19]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda6 = " + str(int(math.degrees(best_configuration[20]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_theta7 = " + str(int(math.degrees(best_configuration[21]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_phi7 = " + str(int(math.degrees(best_configuration[22]))) + u"\u00B0" +
                ", " + u"\u03D5" + "input_lambda7 = " + str(int(math.degrees(best_configuration[23]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta0 = " + str(int(math.degrees(best_configuration[24]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi0 = " + str(int(math.degrees(best_configuration[25]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda0 = " + str(int(math.degrees(best_configuration[26]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta1 = " + str(int(math.degrees(best_configuration[27]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi1 = " + str(int(math.degrees(best_configuration[28]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda1 = " + str(int(math.degrees(best_configuration[29]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta2 = " + str(int(math.degrees(best_configuration[30]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi2 = " + str(int(math.degrees(best_configuration[31]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda2 = " + str(int(math.degrees(best_configuration[32]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta3 = " + str(int(math.degrees(best_configuration[33]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi3 = " + str(int(math.degrees(best_configuration[34]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda3 = " + str(int(math.degrees(best_configuration[35]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta4 = " + str(int(math.degrees(best_configuration[36]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi4 = " + str(int(math.degrees(best_configuration[37]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda4 = " + str(int(math.degrees(best_configuration[38]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta5 = " + str(int(math.degrees(best_configuration[39]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi5 = " + str(int(math.degrees(best_configuration[40]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda5 = " + str(int(math.degrees(best_configuration[41]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta6 = " + str(int(math.degrees(best_configuration[42]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi6 = " + str(int(math.degrees(best_configuration[43]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda6 = " + str(int(math.degrees(best_configuration[44]))) + u"\u00B0" +
                ", " + u"\u03D5" + "theta7 = " + str(int(math.degrees(best_configuration[45]))) + u"\u00B0" +
                ", " + u"\u03D5" + "phi7 = " + str(int(math.degrees(best_configuration[46]))) + u"\u00B0" +
                ", " + u"\u03D5" + "lambda7 = " + str(int(math.degrees(best_configuration[47]))) + u"\u00B0" +
                ''.join([", " + u"\u03B7" + f'{idx}' + " = " + str(int(math.degrees(global_eta))) + u"\u00B0"
                         for idx, global_eta in enumerate(self._global_eta_group)]))

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_group: List[float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotEntangledFullUniversalConfiguration({
            'input_theta0': configuration[0],
            'input_phi0': configuration[1],
            'input_lambda0': configuration[2],
            'input_theta1': configuration[3],
            'input_phi1': configuration[4],
            'input_lambda1': configuration[5],
            'input_theta2': configuration[6],
            'input_phi2': configuration[7],
            'input_lambda2': configuration[8],
            'input_theta3': configuration[9],
            'input_phi3': configuration[10],
            'input_lambda3': configuration[11],
            'input_theta4': configuration[12],
            'input_phi4': configuration[13],
            'input_lambda4': configuration[14],
            'input_theta5': configuration[15],
            'input_phi5': configuration[16],
            'input_lambda5': configuration[17],
            'input_theta6': configuration[18],
            'input_phi6': configuration[19],
            'input_lambda6': configuration[20],
            'input_theta7': configuration[21],
            'input_phi7': configuration[22],
            'input_lambda7': configuration[23],
            'theta0': configuration[24],
            'phi0': configuration[25],
            'lambda0': configuration[26],
            'theta1': configuration[27],
            'phi1': configuration[28],
            'lambda1': configuration[29],
            'theta2': configuration[30],
            'phi2': configuration[31],
            'lambda2': configuration[32],
            'theta3': configuration[33],
            'phi3': configuration[34],
            'lambda3': configuration[35],
            'theta4': configuration[36],
            'phi4': configuration[37],
            'lambda4': configuration[38],
            'theta5': configuration[39],
            'phi5': configuration[40],
            'lambda5': configuration[41],
            'theta6': configuration[42],
            'phi6': configuration[43],
            'lambda6': configuration[44],
            'theta7': configuration[45],
            'phi7': configuration[46],
            'lambda7': configuration[47],
            'eta_group': eta_group})

    def _cost_function(self, params: List[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
            defined in the optimization setup.
            Cost is computed as 1 (perfect probability) - average success probability for
            all the plays with the given configuration
            Returns the Cost (error probability).
        """
        configuration = OneShotEntangledFullUniversalConfiguration({
            'input_theta0': params[0],
            'input_phi0': params[1],
            'input_lambda0': params[2],
            'input_theta1': params[3],
            'input_phi1': params[4],
            'input_lambda1': params[5],
            'input_theta2': params[6],
            'input_phi2': params[7],
            'input_lambda2': params[8],
            'input_theta3': params[9],
            'input_phi3': params[10],
            'input_lambda3': params[11],
            'input_theta4': params[12],
            'input_phi4': params[13],
            'input_lambda4': params[14],
            'input_theta5': params[15],
            'input_phi5': params[16],
            'input_lambda5': params[17],
            'input_theta6': params[18],
            'input_phi6': params[19],
            'input_lambda6': params[20],
            'input_theta7': params[21],
            'input_phi7': params[22],
            'input_lambda7': params[23],
            'theta0': params[24],
            'phi0': params[25],
            'lambda0': params[26],
            'theta1': params[27],
            'phi1': params[28],
            'lambda1': params[29],
            'theta2': params[30],
            'phi2': params[31],
            'lambda2': params[32],
            'theta3': params[33],
            'phi3': params[34],
            'lambda3': params[35],
            'theta4': params[36],
            'phi4': params[37],
            'lambda4': params[38],
            'theta5': params[39],
            'phi5': params[40],
            'lambda5': params[41],
            'theta6': params[42],
            'phi6': params[43],
            'lambda6': params[44],
            'theta7': params[45],
            'phi7': params[46],
            'lambda7': params[47],
            'eta_group': self._global_eta_group})

        return - self._one_shot_circuit.compute_average_success_probability(configuration=configuration,
                                                                            plays=self._setup['plays'])

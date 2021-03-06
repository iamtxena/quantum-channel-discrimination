from qcd.configurations.oneshotentangledconfiguration import OneShotEntangledConfiguration
from . import TheoreticalOneShotOptimization
from ..typings.configurations import (TheoreticalOneShotEntangledOptimalConfiguration,
                                      TheoreticalOneShotEntangledOptimalConfigurations)
import numpy as np


class TheoreticalOneShotEntangledOptimization(TheoreticalOneShotOptimization):
    """ Representation of the theoretical One Shot Optimization """

    def compute_theoretical_optimal_results(self) -> TheoreticalOneShotEntangledOptimalConfigurations:
        """ Finds out the theoretical optimal entangled configuration for each pair of attenuation levels """
        probabilities = []
        list_theoretical_amplitude = []
        best_algorithms = []
        configurations = []
        list_number_calls_made = []
        improvements = []

        for eta_group in self._eta_groups:
            self._global_eta_group = eta_group
            result = self._compute_theoretical_best_configuration()
            best_algorithms.append(result['best_algorithm'])
            probabilities.append(result['best_probability'])
            configurations.append(result['best_configuration'])
            list_number_calls_made.append(result['number_calls_made'])
            list_theoretical_amplitude.append(result['best_theoretical_amplitude'])
            improvements.append(result['improvement'])

        return {'eta_groups': self._eta_groups,
                'best_algorithm': best_algorithms,
                'probabilities': probabilities,
                'configurations': configurations,
                'number_calls_made': list_number_calls_made,
                'list_theoretical_amplitude': list_theoretical_amplitude,
                'improvements': improvements}

    def _compute_theoretical_best_configuration(self) -> TheoreticalOneShotEntangledOptimalConfiguration:
        """ Find out the theoretical entangled best configuration with a global pair of etas (channels) """
        gamma = np.cos(self._global_eta_group[1]) + np.cos(self._global_eta_group[0])
        best_probability = 0
        theoretical_y = (gamma - 1) / (gamma - 2)
        improvement = 0

        if theoretical_y > 0:
            best_probability = 1 / 4 * (np.cos(self._global_eta_group[1]) - np.cos(self._global_eta_group[0])) * \
                ((1 - theoretical_y) * gamma + np.sqrt(
                    (1 - theoretical_y) * (4 * theoretical_y + (1 - theoretical_y) * gamma * gamma))) + 1 / 2
            improvement = 1 / 4 * (np.cos(self._global_eta_group[1]) - np.cos(self._global_eta_group[0])) * \
                ((1 - theoretical_y) * gamma + np.sqrt((1 - theoretical_y) * (
                    4 * theoretical_y + (1 - theoretical_y) * gamma * gamma))) - 1 / 2 * gamma * \
                (np.cos(self._global_eta_group[1]) - np.cos(self._global_eta_group[0]))
        else:
            best_probability = 1 / 2 * gamma * \
                (np.cos(self._global_eta_group[1]) - np.cos(self._global_eta_group[0])) + 1 / 2
            improvement = 0

        return {'best_algorithm': 'One-Shot Side Entanglement Theory',
                'best_probability': best_probability,
                'best_configuration': OneShotEntangledConfiguration({'eta_group': self._global_eta_group,
                                                                     'state_probability': 0,
                                                                     'angle_rx0': 0,
                                                                     'angle_ry0': 0,
                                                                     'angle_rx1': 0,
                                                                     'angle_ry1': 0}),
                'number_calls_made': 1,
                'best_theoretical_amplitude': max(0, theoretical_y),
                'improvement': improvement}

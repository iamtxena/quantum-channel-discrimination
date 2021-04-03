from . import TheoreticalOneShotOptimization
from ..typings.configurations import (TheoreticalOneShotEntangledOptimalConfiguration,
                                      TheoreticalOneShotEntangledOptimalConfigurations)
from ..configurations.configuration import ChannelConfiguration
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

        for eta_pair in self._eta_pairs:
            self._global_eta_pair = eta_pair
            result = self._compute_theoretical_best_configuration()
            best_algorithms.append(result['best_algorithm'])
            probabilities.append(result['best_probability'])
            configurations.append(result['best_configuration'])
            list_number_calls_made.append(result['number_calls_made'])
            list_theoretical_amplitude.append(result['best_theoretical_amplitude'])
            improvements.append(result['improvement'])

        return {'eta_pairs': self._eta_pairs,
                'best_algorithm': best_algorithms,
                'probabilities': probabilities,
                'configurations': configurations,
                'number_calls_made': list_number_calls_made,
                'list_theoretical_amplitude': list_theoretical_amplitude,
                'improvements': improvements}

    def _compute_theoretical_best_configuration(self) -> TheoreticalOneShotEntangledOptimalConfiguration:
        """ Find out the theoretical entangled best configuration with a global pair of etas (channels) """
        gamma = np.cos(self._global_eta_pair[1]) + np.cos(self._global_eta_pair[0])
        best_probability = 0
        theoretical_y = (gamma - 1) / (gamma - 2)
        improvement = 0

        if theoretical_y > 0:
            best_probability = 1 / 4 * (np.cos(self._global_eta_pair[1]) - np.cos(self._global_eta_pair[0])) * \
                ((1 - theoretical_y) * gamma + np.sqrt(
                    (1 - theoretical_y) * (4 * theoretical_y + (1 - theoretical_y) * gamma * gamma))) + 1 / 2
            improvement = 1 / 4 * (np.cos(self._global_eta_pair[1]) - np.cos(self._global_eta_pair[0])) * \
                ((1 - theoretical_y) * gamma + np.sqrt((1 - theoretical_y) * (
                    4 * theoretical_y + (1 - theoretical_y) * gamma * gamma))) - 1 / 2 * gamma * \
                (np.cos(self._global_eta_pair[1]) - np.cos(self._global_eta_pair[0]))
        else:
            best_probability = 1 / 2 * gamma * \
                (np.cos(self._global_eta_pair[1]) - np.cos(self._global_eta_pair[0])) + 1 / 2
            improvement = 0

        return {'best_algorithm': 'One-Shot Side Entanglement Theory',
                'best_probability': best_probability,
                'best_configuration': ChannelConfiguration({'eta_pair': self._global_eta_pair}),
                'number_calls_made': 1,
                'best_theoretical_amplitude': max(0, theoretical_y),
                'improvement': improvement}

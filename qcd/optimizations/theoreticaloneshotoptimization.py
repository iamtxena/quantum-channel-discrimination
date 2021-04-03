from . import TheoreticalOptimization
from ..typings.configurations import (TheoreticalOneShotOptimalConfiguration,
                                      TheoreticalOneShotOptimalConfigurations)
from ..configurations.configuration import ChannelConfiguration
import numpy as np


class TheoreticalOneShotOptimization(TheoreticalOptimization):
    """ Representation of the theoretical One Shot Optimization """

    def compute_theoretical_optimal_results(self) -> TheoreticalOneShotOptimalConfigurations:
        """ Finds out the theoretical optimal configuration for each pair of attenuation levels """
        probabilities = []
        list_theoretical_amplitude = []
        best_algorithms = []
        configurations = []
        list_number_calls_made = []

        for eta_pair in self._eta_pairs:
            self._global_eta_pair = eta_pair
            result = self._compute_theoretical_best_configuration()
            best_algorithms.append(result['best_algorithm'])
            probabilities.append(result['best_probability'])
            configurations.append(result['best_configuration'])
            list_number_calls_made.append(result['number_calls_made'])
            list_theoretical_amplitude.append(result['best_theoretical_amplitude'])

        return {'eta_pairs': self._eta_pairs,
                'best_algorithm': best_algorithms,
                'probabilities': probabilities,
                'configurations': configurations,
                'number_calls_made': list_number_calls_made,
                'list_theoretical_amplitude': list_theoretical_amplitude}

    def _compute_theoretical_best_configuration(self) -> TheoreticalOneShotOptimalConfiguration:
        """ Find out the theoretical best configuration with a global pair of etas (channels) """
        gamma = np.cos(self._global_eta_pair[1]) + np.cos(self._global_eta_pair[0])
        best_probability = 0
        best_theoretical_amplitude = 0
        if gamma < 1 / np.sqrt(2):
            best_probability = 1 / 2 + 1 / 4 * \
                (np.cos(self._global_eta_pair[1]) - np.cos(self._global_eta_pair[0])) / np.sqrt(1 - gamma * gamma)
            best_theoretical_amplitude = np.sqrt(1 / (2 - 2 * gamma * gamma))
        else:
            best_probability = 1 / 2 * (np.sin(self._global_eta_pair[0]) * np.sin(
                self._global_eta_pair[0]) + np.cos(self._global_eta_pair[1]) * np.cos(self._global_eta_pair[1]))
            best_theoretical_amplitude = 1

        return {'best_algorithm': 'One-Shot Theory',
                'best_probability': best_probability,
                'best_configuration': ChannelConfiguration({'eta_pair': self._global_eta_pair}),
                'number_calls_made': 1,
                'best_theoretical_amplitude': best_theoretical_amplitude}
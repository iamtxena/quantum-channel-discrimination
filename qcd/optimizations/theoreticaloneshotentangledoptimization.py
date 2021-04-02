from . import TheoreticalOneShotOptimization
from ..typings.configurations import (TheoreticalOptimalConfigurations,
                                      TheoreticalOneShotEntangledOptimalConfiguration,
                                      TheoreticalOneShotEntangledOptimalConfigurations)
import numpy as np


class TheoreticalOneShotEntangledOptimization(TheoreticalOneShotOptimization):
    """ Representation of the theoretical One Shot Optimization """

    def compute_theoretical_optimal_results(self) -> TheoreticalOptimalConfigurations:
        """ Finds out the theoretical optimal entangled configuration for each pair of attenuation levels """
        probabilities = []
        list_theoretical_amplitude = []
        improvements = []

        for eta_pair in self._eta_pairs:
            self._global_eta_pair = eta_pair
            result = self._compute_theoretical_best_configuration()
            probabilities.append(result['best_probability'])
            list_theoretical_amplitude.append(result['best_theoretical_amplitude'])
            improvements.append(result['improvement'])

        return TheoreticalOneShotEntangledOptimalConfigurations({
            'eta_pairs': self._eta_pairs,
            'probabilities': probabilities,
            'list_theoretical_amplitude': list_theoretical_amplitude,
            'improvements': improvements})

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

        return {'best_probability': best_probability,
                'best_theoretical_amplitude': max(0, theoretical_y),
                'improvement': improvement}

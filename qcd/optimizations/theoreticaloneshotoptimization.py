from . import TheoreticalOptimization
from ..typings import TheoreticalOptimizationSetup
from ..typings.configurations import OptimalConfiguration
from ..configurations import OneShotConfiguration
import numpy as np


class TheoreticalOneShotOptimization(TheoreticalOptimization):
    """ Representation of the theoretical One Shot Optimization """

    def __init__(self, optimization_setup: TheoreticalOptimizationSetup):
        super().__init__(optimization_setup)

    def _compute_theoretical_best_configuration(self) -> OptimalConfiguration:
        """ Find out the theoretical best configuration with a global pair of etas (channels) """
        gamma = np.cos(self._global_eta_pair[1]) + np.cos(self._global_eta_pair[0])
        best_probability = 0
        best_theta = 0
        if gamma < 1 / np.sqrt(2):
            best_probability = 1 / 2 + 1 / 4 * \
                (np.cos(self._global_eta_pair[1]) - np.cos(self._global_eta_pair[0])) / np.sqrt(1 - gamma * gamma)
            best_theta = np.sqrt(1 / (2 - 2 * gamma * gamma))
        else:
            best_probability = 1 / 2 * (np.sin(self._global_eta_pair[0]) * np.sin(
                self._global_eta_pair[0]) + np.cos(self._global_eta_pair[1]) * np.cos(self._global_eta_pair[1]))
            best_theta = 1
        one_shot_configuration = OneShotConfiguration({
            'theta': best_theta,
            'angle_rx': 0,
            'angle_ry': 0,
            'eta_pair': self._global_eta_pair})
        return OptimalConfiguration(
            best_algorithm="THEORY",
            best_probability=best_probability,
            best_configuration=one_shot_configuration,
            number_calls_made=1)

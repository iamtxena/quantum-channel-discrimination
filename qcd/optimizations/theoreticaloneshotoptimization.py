from . import TheoreticalOptimization
from ..typings import TheoreticalOptimizationSetup
from ..typings.configurations import TheoreticalOptimalConfiguration
import numpy as np


class TheoreticalOneShotOptimization(TheoreticalOptimization):
    """ Representation of the theoretical One Shot Optimization """

    def __init__(self, optimization_setup: TheoreticalOptimizationSetup):
        super().__init__(optimization_setup)

    def _compute_theoretical_best_configuration(self) -> TheoreticalOptimalConfiguration:
        """ Find out the theoretical best configuration with a global pair of etas (channels) """
        gamma = np.cos(self._global_eta_pair[1]) + np.cos(self._global_eta_pair[0])
        best_probability = 0
        best_theoric_x = 0
        if gamma < 1 / np.sqrt(2):
            best_probability = 1 / 2 + 1 / 4 * \
                (np.cos(self._global_eta_pair[1]) - np.cos(self._global_eta_pair[0])) / np.sqrt(1 - gamma * gamma)
            best_theoric_x = np.sqrt(1 / (2 - 2 * gamma * gamma))
        else:
            best_probability = 1 / 2 * (np.sin(self._global_eta_pair[0]) * np.sin(
                self._global_eta_pair[0]) + np.cos(self._global_eta_pair[1]) * np.cos(self._global_eta_pair[1]))
            best_theoric_x = 1

        return TheoreticalOptimalConfiguration(
            best_probability=best_probability,
            best_theoric_x=best_theoric_x)

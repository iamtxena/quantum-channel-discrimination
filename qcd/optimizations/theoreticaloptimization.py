from abc import ABC, abstractmethod
from ..typings import TheoreticalOptimizationSetup
from ..typings.configurations import TheoreticalOptimalConfiguration, TheoreticalOptimalConfigurations
from .aux import parse_eta_pairs


class TheoreticalOptimization(ABC):
    """ Generic class acting as an interface for any Theoretical Channel Optimization """

    def __init__(self, optimization_setup: TheoreticalOptimizationSetup):
        self._eta_pairs = parse_eta_pairs(optimization_setup['eta_pairs'])
        self._global_eta_pair = (0.0, 0.0)

    @abstractmethod
    def _compute_theoretical_best_configuration(self) -> TheoreticalOptimalConfiguration:
        """ Find out the theoretical best configuration with a global pair of etas (channels) """
        pass

    @abstractmethod
    def compute_theoretical_optimal_results(self) -> TheoreticalOptimalConfigurations:
        """ Finds out the theoretical optimal configuration for each pair of attenuation levels """
        pass

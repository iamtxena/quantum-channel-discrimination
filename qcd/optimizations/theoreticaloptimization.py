from abc import ABC, abstractmethod
from ..typings import TheoreticalOptimizationSetup
from ..typings.configurations import OptimalConfiguration, OptimalConfigurations


class TheoreticalOptimization(ABC):
    """ Generic class acting as an interface for any Theoretical Channel Optimization """

    def __init__(self, optimization_setup: TheoreticalOptimizationSetup):
        self._eta_groups = optimization_setup['eta_groups']
        self._global_eta_group = [0.0] * optimization_setup['number_channels_to_discriminate']

    @abstractmethod
    def compute_theoretical_optimal_results(self) -> OptimalConfigurations:
        """ Finds out the theoretical optimal configuration for each pair of attenuation levels """
        pass

    @abstractmethod
    def _compute_theoretical_best_configuration(self) -> OptimalConfiguration:
        """ Find out the theoretical best configuration with a global pair of etas (channels) """
        pass

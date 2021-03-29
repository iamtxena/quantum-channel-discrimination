from abc import ABC, abstractmethod
from ..optimizations import OptimalConfigurations
from ..typings import OptimizationSetup


class Optimization(ABC):
    """ Generic class acting as an interface for any Channel Optimization """

    @abstractmethod
    def find_optimal_configurations(self,
                                    optimization_setup: OptimizationSetup) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        pass

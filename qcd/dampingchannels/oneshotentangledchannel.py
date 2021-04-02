from . import OneShotDampingChannel
from ..typings import OptimizationSetup
from ..optimizations import OneShotEntangledOptimization
from ..typings.configurations import OptimalConfigurations


class OneShotEntangledDampingChannel(OneShotDampingChannel):
    """ Representation of the One Shot Two Qubit Entangled Quantum Damping Channel """

    def plot_first_channel(self):
        return self._circuits[0][0].draw('')

    def find_optimal_configurations(self,
                                    optimization_setup: OptimizationSetup) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
          using the configured optimization algorithm for an Entangled channel """

        return OneShotEntangledOptimization(optimization_setup).find_optimal_configurations()

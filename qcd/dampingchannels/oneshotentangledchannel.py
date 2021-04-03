from typing import Optional
from . import OneShotDampingChannel
from ..typings import CloneSetup, OptimizationSetup
from ..optimizations import OneShotEntangledOptimization
from ..typings.configurations import OptimalConfigurations
from qcd import save_object_to_disk


class OneShotEntangledDampingChannel(OneShotDampingChannel):
    """ Representation of the One Shot Two Qubit Entangled Quantum Damping Channel """

    def plot_first_channel(self):
        return self._circuits[0][0].draw('')

    @staticmethod
    def find_optimal_configurations(optimization_setup: OptimizationSetup,
                                    clone_setup: Optional[CloneSetup]) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
          using the configured optimization algorithm for an Entangled channel """

        optimal_configurations = OneShotEntangledOptimization(
            optimization_setup).find_optimal_configurations(clone_setup)
        if clone_setup is not None and clone_setup['file_name'] is not None:
            save_object_to_disk(optimal_configurations,
                                f"{clone_setup['file_name']}_{clone_setup['id_clone']}", clone_setup['path'])
        return optimal_configurations

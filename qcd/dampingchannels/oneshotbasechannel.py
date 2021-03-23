from . import DampingChannel
from typing import Optional
from ..backends import DeviceBackend
from ..configurations import SetupConfiguration
from ..results import ExecutionResults
from ..optimizations import OptimizationSetup, OptimalConfigurations


class OneShotDampingChannel(DampingChannel):
    """ Representation of the One Shot Quantum Damping Channel """

    def set_backend(self, backend: DeviceBackend) -> None:
        """ Defines a provider backend to execute the experiments """
        self._backend = backend

    def setup_channel(self, channel_setup_configuration: SetupConfiguration) -> None:
        """ Defines the parameters to be used to setup the channel as a base configuration """
        self._channel_setup_configuration = channel_setup_configuration

    def setup_optimization(self, optimization_setup: OptimizationSetup) -> None:
        """ Defines the optimization parameters to be used to find the optimal configuration values """
        self._optimization_setup = optimization_setup

    def create(self) -> None:
        """ Builder function to create all circuits parametrizing a quantum channel with the given configuration """

    def run(self) -> ExecutionResults:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        raise NotImplementedError('Method not implemented')

    def find_optimal_configurations(self) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        raise NotImplementedError('Method not implemented')

    def plot_surface_probabilities(self):
        """ Displays the output probabilities for all circuits in a 3D plot """
        raise NotImplementedError('Method not implemented')

    def plot_wireframe_blochs(self, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel  """
        raise NotImplementedError('Method not implemented')

    def plot_wireframe_blochs_one_lambda(self, one_lambda: int, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel
            using only the provided attenuation level (lambda) """
        raise NotImplementedError('Method not implemented')

    def plot_fidelity(self):
        """ Displays the channel fidelity for 11 discrete attenuation levels ranging from
            0 (minimal attenuation) to 1 (maximal attenuation) """
        raise NotImplementedError('Method not implemented')

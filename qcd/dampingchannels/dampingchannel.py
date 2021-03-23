from abc import ABC, abstractmethod
from typing import Optional
from ..backends import DeviceBackend
from ..configurations import SetupConfiguration
from ..results import ExecutionResults
from ..optimizations import OptimizationSetup, OptimalConfigurations


class DampingChannel(ABC):
    """ Generic class acting as an interface for any damping channel
        using a provided discrimantion strategy """

    @abstractmethod
    def set_backend(self, backend: DeviceBackend) -> None:
        """ Defines a provider backend to execute the experiments """
        pass

    @abstractmethod
    def setup_channel(self, setup: SetupConfiguration) -> None:
        """ Defines the parameters to be used to setup the channel as a base configuration """
        pass

    @abstractmethod
    def setup_optimization(self, setup: OptimizationSetup) -> None:
        """ Defines the optimization parameters to be used to find the optimal configuration values """
        pass

    @abstractmethod
    def create(self) -> None:
        """ Builder function to create all circuits parametrizing a quantum channel with the given configuration """
        pass

    @abstractmethod
    def run(self) -> ExecutionResults:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        pass

    @abstractmethod
    def find_optimal_configurations(self) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        pass

    @abstractmethod
    def plot_surface_probabilities(self):
        """ Displays the output probabilities for all circuits in a 3D plot """
        pass

    @abstractmethod
    def plot_wireframe_blochs(self, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel  """
        pass

    @abstractmethod
    def plot_wireframe_blochs_one_lambda(self, one_lambda: int, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel
            using only the provided attenuation level (lambda) """
        pass

    @abstractmethod
    def plot_fidelity(self):
        """ Displays the channel fidelity for 11 discrete attenuation levels ranging from
            0 (minimal attenuation) to 1 (maximal attenuation) """
        pass

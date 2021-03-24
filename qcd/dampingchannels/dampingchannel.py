from abc import ABC, abstractmethod
from typing import Optional
from ..backends import DeviceBackend
from ..configurations import SetupConfiguration
from ..results import ExecutionResults
from ..optimizations import OptimizationSetup, OptimalConfigurations


class DampingChannel(ABC):
    """ Generic class acting as an interface for any damping channel
        using a provided discrimantion strategy """

    def __init__(self, backend: DeviceBackend,
                 channel_setup_configuration: SetupConfiguration,
                 optimization_setup: Optional[OptimizationSetup] = None) -> None:
        self._backend = backend
        self._channel_setup_configuration = channel_setup_configuration
        self._optimization_setup = optimization_setup

    def setup_optimization(self, optimization_setup: OptimizationSetup) -> None:
        """ Defines the optimization parameters to be used to find the optimal configuration values """
        if self._optimization_setup is not None:
            raise AttributeError("Optimization setup already defined")
        self._optimization_setup = optimization_setup

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

    @abstractmethod
    def plot_first_channel(self):
        """ Draws the first created channel """
        pass

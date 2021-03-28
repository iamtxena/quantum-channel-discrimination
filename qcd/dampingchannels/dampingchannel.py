from abc import ABC, abstractmethod
from typing import Optional, List, Union
from ..backends import DeviceBackend
from ..configurations import SetupConfiguration
from ..executions import Execution
from ..optimizations import OptimizationSetup, OptimalConfigurations


class DampingChannel(ABC):
    """ Generic class acting as an interface for any damping channel
        using a provided discrimantion strategy """

    def __init__(self,
                 channel_setup_configuration: SetupConfiguration,
                 optimization_setup: Optional[OptimizationSetup] = None) -> None:
        self._channel_setup_configuration = channel_setup_configuration
        self._optimization_setup = optimization_setup

    def setup_optimization(self, optimization_setup: OptimizationSetup) -> None:
        """ Defines the optimization parameters to be used to find the optimal configuration values """
        if self._optimization_setup is not None:
            raise AttributeError("Optimization setup already defined")
        self._optimization_setup = optimization_setup

    @abstractmethod
    def run(self, backend: Union[DeviceBackend, List[DeviceBackend]],
            iterations: Optional[int] = 1024, timeout: Optional[float] = None) -> Union[Execution, List[Execution]]:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        pass

    @ abstractmethod
    def find_optimal_configurations(self) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        pass

    @ abstractmethod
    def plot_first_channel(self):
        """ Draws the first created channel """
        pass
from abc import ABC, abstractmethod
from typing import Optional, List, Union
from ..backends import DeviceBackend
from ..configurations import SetupConfiguration
from ..executions import Execution
from ..optimizationresults import OptimizationResults
from ..typings import OptimizationSetup


class DampingChannel(ABC):
    """ Generic class acting as an interface for any damping channel
        using a provided discrimantion strategy """

    def __init__(self,
                 channel_setup_configuration: Optional[SetupConfiguration] = None) -> None:
        self._channel_setup_configuration = channel_setup_configuration

    @abstractmethod
    def run(self, backend: Union[DeviceBackend, List[DeviceBackend]],
            iterations: Optional[int] = 1024, timeout: Optional[float] = None) -> Execution:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        pass

    @abstractmethod
    def find_optimal_configurations(self,
                                    optimization_setup: OptimizationSetup) -> OptimizationResults:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        pass

    @abstractmethod
    def plot_first_channel(self):
        """ Draws the first created channel """
        pass

    @abstractmethod
    def plot_fidelity(self):
        """ Displays the channel fidelity for 11 discrete attenuation levels ranging from
            0 (minimal attenuation) to 1 (maximal attenuation) """
        pass

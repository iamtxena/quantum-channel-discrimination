from abc import ABC, abstractmethod
from qcd.typings.configurations import OptimalConfigurations
from qcd.configurations.configuration import ChannelConfiguration
from qcd.circuits import Circuit
from qcd.optimizationresults import GlobalOptimizationResults
from typing import Optional, List, Union
from ..backends import DeviceBackend
from ..executions import Execution
from ..typings import CloneSetup, OptimizationSetup


class DampingChannel(ABC):
    """ Generic class acting as an interface for any damping channel
        using a provided discrimantion strategy """

    @staticmethod
    @abstractmethod
    def find_optimal_configurations(optimization_setup: OptimizationSetup,
                                    clone_setup: Optional[CloneSetup] = None) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        pass

    @staticmethod
    @abstractmethod
    def build_from_optimal_configurations(file_name: str, path: Optional[str] = ""):
        """ Builds a Quantum Damping Channel from the optimal configuration for each pair of attenuation angles """
        pass

    def __init__(self) -> None:
        self._one_shot_circuit: Circuit

    @staticmethod
    @abstractmethod
    def discriminate_channel(configuration: ChannelConfiguration, plays: Optional[int] = 100) -> float:
        """ Computes the average success probability of running a specific configuration
            for the number of plays specified.
        """
        pass

    @abstractmethod
    def run(self, backend: Union[DeviceBackend, List[DeviceBackend]],
            iterations: Optional[int] = 1024, timeout: Optional[float] = None) -> Execution:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        pass

    def one_shot_run(self, plays: Optional[int] = 100) -> GlobalOptimizationResults:
        """ Runs all the experiments using the optimal configurations and computing the success probability """
        if self._one_shot_circuit is None:
            raise ValueError('Optimal configurations MUST be provided to execute one shot runs')

        return GlobalOptimizationResults(self._one_shot_circuit.one_shot_run(plays))

    @abstractmethod
    def plot_first_channel(self):
        """ Draws the first created channel """
        pass

    @abstractmethod
    def plot_fidelity(self):
        """ Displays the channel fidelity for 11 discrete attenuation levels ranging from
            0 (minimal attenuation) to 1 (maximal attenuation) """
        pass

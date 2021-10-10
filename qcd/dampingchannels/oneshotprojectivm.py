from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qcd.configurations.configuration import ChannelConfiguration
from qcd.optimizationresults.aux import load_result_from_file, save_result_to_disk
from qcd.configurations import OneShotSetupConfiguration
from qcd.circuits import OneShotEntangledCircuit
from typing import List, Optional, Tuple
from . import OneShotDampingChannel
from ..typings import CloneSetup, OptimizationSetup, ResultStates
from ..optimizations import OneShotEntangledOptimization
from ..typings.configurations import OptimalConfigurations, ValidatedConfiguration
import numpy as np


#class OneShotEntangledDampingChannel(OneShotDampingChannel):
class OneShotDampingChannelProjectivMeasurement(OneShotDampingChannel):
    """ Representation of the One Shot Two Qubit Entangled Quantum Damping Channel """

    @staticmethod
    def build_from_optimal_configurations(file_name: str, path: Optional[str] = ""):
        """ Builds a Quantum Damping Channel from the optimal configuration for each pair of attenuation angles """
#        return OneShotEntangledDampingChannel(optimal_configurations=load_result_from_file(file_name, path))
        return OneShotDampingChannelProjectivMeasurement(optimal_configurations=load_result_from_file(file_name, path))


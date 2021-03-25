from . import DampingChannel
from typing import Optional, List
from ..backends import DeviceBackend
from ..configurations import OneShotSetupConfiguration
from ..executions import Execution
from ..optimizations import OptimizationSetup, OptimalConfigurations
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
import math


class OneShotDampingChannel(DampingChannel):
    """ Representation of the One Shot Quantum Damping Channel """

    def __init__(self,
                 channel_setup_configuration: OneShotSetupConfiguration,
                 optimization_setup: Optional[OptimizationSetup] = None) -> None:
        super().__init__(channel_setup_configuration, optimization_setup)

        self._circuits = self._create_all_circuits(channel_setup_configuration)

    def run(self, backend: List[DeviceBackend]) -> Execution:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        raise NotImplementedError('Method not implemented')

    def find_optimal_configurations(self) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        raise NotImplementedError('Method not implemented')

    def plot_first_channel(self):
        return self._circuits[0][0].draw('mpl')

    def _create_all_circuits(self, channel_setup_configuration: OneShotSetupConfiguration):
        # Create 2 qbits circuit and 1 output classical bit
        qreg_q = QuantumRegister(2, 'q')
        creg_c = ClassicalRegister(1, 'c')

        circuits = []
        # Initialize circuit with desired initial_state
        initial_states = self._prepare_initial_states(
            channel_setup_configuration.angles_theta, channel_setup_configuration.angles_phase)

        for attenuation_factor in channel_setup_configuration.attenuation_factors:
            circuit_one_attenuation_factor = []
            for index_initial_state, _ in enumerate(initial_states["zero_amplitude"]):
                circuit = QuantumCircuit(qreg_q, creg_c)
                circuit.initialize([initial_states["zero_amplitude"][index_initial_state],
                                    initial_states["one_amplitude"][index_initial_state]], qreg_q[0])
                circuit.reset(qreg_q[1])
                circuit.cry(2 * np.arcsin(np.sqrt(attenuation_factor)), qreg_q[0], qreg_q[1])
                circuit.cx(qreg_q[1], qreg_q[0])
                circuit.rx(0, qreg_q[0])  # rx set always to 0
                circuit.ry(0, qreg_q[0])  # ry set always to 0
                circuit.measure(qreg_q[0], creg_c[0])
                circuit_one_attenuation_factor.append(circuit)
            circuits.append(circuit_one_attenuation_factor)
        return circuits

    def _prepare_initial_states(self, angles_theta, angles_phase):
        """ Prepare initial states to pass through the circuit """
        # As we have to provide the state values to initialize the qreg[0] we have to do a conversion
        # from angles in the sphere to statevector amplitudes. These statevectors will be the combination of
        # Zero_Amplitude*|0> plus One_Amplitude*|1>
        initial_states_zero_Amplitude = []
        initial_states_one_amplitude = []

        for theta in angles_theta:
            for phase in angles_phase:
                initial_states_zero_Amplitude.append(math.cos(theta))
                initial_states_one_amplitude.append((math.sin(theta) * math.e**(1j * phase)))
        return {
            "zero_amplitude": initial_states_zero_Amplitude,
            "one_amplitude": initial_states_one_amplitude,
        }

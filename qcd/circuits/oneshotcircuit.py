
from qcd.circuits.aux import get_measured_value_from_counts
from qcd.configurations import OneShotConfiguration
from qcd.configurations.configuration import ChannelConfiguration
from . import Circuit
from typing import List, Tuple, cast
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class OneShotCircuit(Circuit):
    """ Representation of the One Shot Channel circuit """

    def _prepare_initial_state(self, state_probability: float) -> Tuple[complex, complex]:
        """ Prepare initial state: computing 'x' as the amplitudes """
        return (np.sqrt(1 - state_probability), np.sqrt(state_probability))

    def _guess_eta_used_one_bit_strategy(self, counts: str) -> int:
        """ Decides which eta was used on the real execution from the one bit 'counts' measured
            It is a silly guess.
            It returns the same eta index used as the measured result
        """
        if len(counts) != 1:
            raise ValueError('counts MUST be a one character length string')
        if "0" in counts:
            return 0
        return 1

    def _convert_counts_to_eta_used(self, counts_dict: dict) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        counts = get_measured_value_from_counts(counts_dict)
        return self._guess_eta_used_one_bit_strategy(counts)

    def _convert_all_counts_to_all_eta_used(self,
                                            counts_all_circuits: List[dict]) -> List[int]:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        return [self._convert_counts_to_eta_used(counts)
                for counts in counts_all_circuits]

    def _create_one_circuit(self,
                            configuration: ChannelConfiguration,
                            eta: float) -> QuantumCircuit:
        """ Creates one circuit from a given  configuration and eta """
        configuration = cast(OneShotConfiguration, configuration)
        qreg_q = QuantumRegister(2, 'q')
        creg_c = ClassicalRegister(1, 'c')
        initial_state = self._prepare_initial_state(configuration.state_probability)
        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.initialize([initial_state[0],
                            initial_state[1]], qreg_q[0])
        circuit.reset(qreg_q[1])
        circuit.cry(2 * eta, qreg_q[0], qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.rx(configuration.angle_rx, qreg_q[0])
        circuit.ry(configuration.angle_ry, qreg_q[0])
        circuit.measure(qreg_q[0], creg_c[0])
        return circuit

    def _create_one_configuration(self,
                                  configuration: ChannelConfiguration,
                                  eta_pair: Tuple[float, float]) -> OneShotConfiguration:
        """ Creates a specific configuration setting a specific eta pair """
        return OneShotConfiguration({
            'state_probability': cast(OneShotConfiguration, configuration).state_probability,
            'angle_rx': cast(OneShotConfiguration, configuration).angle_rx,
            'angle_ry': cast(OneShotConfiguration, configuration).angle_ry,
            'eta_pair': eta_pair
        })

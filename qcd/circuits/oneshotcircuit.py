from qcd.circuits.aux import get_measured_value_from_counts
from qcd.configurations import OneShotConfiguration
from qcd.typings import GuessStrategy
from qcd.configurations.configuration import ChannelConfiguration
from . import Circuit
from typing import Tuple, cast
import numpy as np
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute


class OneShotCircuit(Circuit):
    """ Representation of the One Shot Channel circuit """

    def _prepare_initial_state(self, state_probability: float) -> Tuple[complex, complex]:
        """ Prepare initial state: computing 'x' as the amplitudes """
        return (np.sqrt(1 - state_probability), np.sqrt(state_probability))
        # return (np.cos(np.arcsin(state_probability)), state_probability)

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

    def _convert_counts_to_eta_used(self,
                                    counts_dict: dict,
                                    guess_strategy: GuessStrategy) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        if guess_strategy != GuessStrategy.one_bit_same_as_measured:
            raise ValueError('Invalid Guess Strategy. Only GuessStrategy.one_bit_same_as_measured supported')

        counts = get_measured_value_from_counts(counts_dict)
        return self._guess_eta_used_one_bit_strategy(counts)

    def _compute_damping_channel(self, channel_configuration: ChannelConfiguration, eta_index: int) -> int:
        """ one-time execution of the amplitude damping circuit using the passed parameters
            Returns: the execution measured result: either 0 or 1
        """
        configuration = cast(OneShotConfiguration, channel_configuration)
        backend = Aer.get_backend('qasm_simulator') if self._backend is None else self._backend.backend
        eta = configuration.eta_pair[eta_index]
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

        counts = execute(circuit, backend, shots=1).result().get_counts(circuit)
        return self._convert_counts_to_eta_used(counts, guess_strategy=GuessStrategy.one_bit_same_as_measured)

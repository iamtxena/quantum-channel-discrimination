from qcd.circuits.aux import get_measured_value_from_counts
from qcd.configurations import OneShotConfiguration
from qcd.typings import GuessStrategy
from qcd.configurations.configuration import ChannelConfiguration
from . import OneShotCircuit
from typing import Tuple, cast
import numpy as np
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import random


class OneShotEntangledCircuit(OneShotCircuit):
    """ Representation of the One Shot Entangled Channel circuit """

    def _prepare_initial_state_entangled(self, state_probability: float) -> Tuple[complex, complex, complex, complex]:
        """ Prepare initial state: computing 'y' as the amplitudes  """
        return (0, np.sqrt(state_probability), np.sqrt(1 - state_probability), 0)

    def _guess_lambda_used_two_bit_strategy(self, counts: str) -> int:
        """ Decides which lambda was used on the real execution from the two 'counts' measured
            Setting eta0 >= eta1:
                * outcome 00 -> eta0 as the most probable (more attenuation)
                * outcome 01 -> eta1 as the most probable (less attenuation)
                * outcome 10 -> 50% chance, random choice
                * outcome 11 -> not possible, but in case we get it (from noisy simulation), 50% chance, random choice
        """
        if len(counts) != 2:
            raise ValueError('counts MUST be a two character length string')
        if counts == "00":
            return 1
        if counts == "01":
            return 0
        if counts == "10" or counts == "11":
            return random.choice([0, 1])
        raise ValueError("Accepted counts are '00', '01', '10', '11'")

    def _convert_counts_to_eta_used(self,
                                    counts_dict: dict,
                                    guess_strategy: GuessStrategy) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        if guess_strategy != GuessStrategy.two_bit_base:
            raise ValueError('Invalid Guess Strategy. Only GuessStrategy.two_bit_base supported')

        counts = get_measured_value_from_counts(counts_dict)
        return self._guess_lambda_used_two_bit_strategy(counts)

    def _compute_damping_channel(self, channel_configuration: ChannelConfiguration, eta_index: int) -> int:
        """ one-time execution of the two-qubit entangled amplitude damping circuit using the passed parameters
            Returns: the execution measured result: either 0 or 1
        """
        configuration = cast(OneShotConfiguration, channel_configuration)
        backend = Aer.get_backend('qasm_simulator') if self._backend is None else self._backend.backend
        eta = configuration.eta_pair[eta_index]
        qreg_q = QuantumRegister(3, 'q')
        creg_c = ClassicalRegister(2, 'c')

        initial_state = self._prepare_initial_state_entangled(configuration.state_probability)

        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.initialize(initial_state, [0, 1])
        circuit.reset(qreg_q[2])
        circuit.cry(2 * eta, qreg_q[1], qreg_q[2])
        circuit.cx(qreg_q[2], qreg_q[1])
        circuit.rx(configuration.angle_rx, qreg_q[1])
        circuit.ry(configuration.angle_ry, qreg_q[1])
        circuit.barrier()
        circuit.measure([0, 1], creg_c)

        counts = execute(circuit, backend, shots=1).result().get_counts(circuit)
        return self._convert_counts_to_eta_used(counts, guess_strategy=GuessStrategy.two_bit_base)

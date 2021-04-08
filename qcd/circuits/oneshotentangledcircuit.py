from qcd.configurations.configuration import ChannelConfiguration
from qcd.circuits.aux import get_measured_value_from_counts
from qcd.configurations import OneShotConfiguration
from . import OneShotCircuit
from typing import List, Tuple, cast
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import random


class OneShotEntangledCircuit(OneShotCircuit):
    """ Representation of the One Shot Entangled Channel circuit """

    def _prepare_initial_state_entangled(self, state_probability: float) -> Tuple[complex, complex, complex, complex]:
        """ Prepare initial state: computing 'y' as the amplitudes  """
        return (0, np.sqrt(state_probability), np.sqrt(1 - state_probability), 0)

    def _guess_lambda_used_two_bit_strategy(self, counts: str) -> int:
        """ Decides which eta was used on the real execution from the two 'counts' measured
            Qubits order MATTER!!!!
            "01" means that:
              the LEFTMOST bit (0) corresponds to the measurement of the qubit that goes THROUGH the channel
              and the RIGHTMOST bit (1) corresponds to the measurement of the qubit that goes OUTSIDE the channel
            Remember that we are only sending |01> + |10> entangles states
            Setting eta0 >= eta1:
                * outcome 00 -> eta0 as the most probable (more attenuation)
                * outcome 01 -> we do not know if there has been attenuation. 50% chance, random choice
                * outcome 10 -> eta1 as the most probable (less attenuation)
                * outcome 11 -> not possible, but in case we get it (from noisy simulation), 50% chance, random choice
        """
        if len(counts) != 2:
            raise ValueError('counts MUST be a two character length string')
        if counts == "00":
            return 0
        if counts == "10":
            return 1
        if counts == "01" or counts == "11":
            return random.choice([0, 1])
        raise ValueError("Accepted counts are '00', '01', '10', '11'")

    def _convert_counts_to_eta_used(self, counts_dict: dict) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        counts = get_measured_value_from_counts(counts_dict)
        return self._guess_lambda_used_two_bit_strategy(counts)

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
        return circuit

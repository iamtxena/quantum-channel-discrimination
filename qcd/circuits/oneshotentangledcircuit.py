from qcd.configurations.configuration import ChannelConfiguration
from qcd.configurations import OneShotConfiguration, OneShotEntangledConfiguration
from . import OneShotCircuit
from typing import List, Tuple, cast
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class OneShotEntangledCircuit(OneShotCircuit):
    """ Representation of the One Shot Entangled Channel circuit """

    def _prepare_initial_state_entangled(self, state_probability: float) -> Tuple[complex, complex, complex, complex]:
        """ Prepare initial state: computing 'y' as the amplitudes  """
        return (0, np.sqrt(state_probability), np.sqrt(1 - state_probability), 0)

    def _get_max_counts_distribution_for_all_channels(
            self,
            all_channel_counts: List[dict],
            counts_distribution: List[float]) -> List[float]:
        """ returns the max counts between the max counts up to that moment and the circuit counts """
        if not all_channel_counts:
            return counts_distribution

        one_channel_counts = all_channel_counts.pop()
        max_counts = [0.0, 0.0, 0.0, 0.0]
        if '00' in one_channel_counts:
            max_counts[0] = max([counts_distribution[0], one_channel_counts['00']])
        if '01' in one_channel_counts:
            max_counts[1] = max([counts_distribution[1], one_channel_counts['01']])
        if '10' in one_channel_counts:
            max_counts[2] = max([counts_distribution[2], one_channel_counts['10']])
        if '11' in one_channel_counts:
            max_counts[3] = max([counts_distribution[3], one_channel_counts['11']])

        return self._get_max_counts_distribution_for_all_channels(all_channel_counts, max_counts)

    def _guess_probability_from_counts(self,
                                       eta_counts: List[dict],
                                       plays: int,
                                       eta_group_length: int) -> float:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        counts_distribution = self._get_max_counts_distribution_for_all_channels(eta_counts, [0.0, 0.0, 0.0, 0.0])
        return (counts_distribution[0] +
                counts_distribution[1] +
                counts_distribution[2] +
                counts_distribution[3]) / (plays * eta_group_length)

    def _create_one_circuit(self,
                            configuration: ChannelConfiguration,
                            eta: float) -> QuantumCircuit:
        """ Creates one circuit from a given  configuration and eta """
        configuration = cast(OneShotEntangledConfiguration, configuration)
        qreg_q = QuantumRegister(3, 'q')
        creg_c = ClassicalRegister(2, 'c')

        initial_state = self._prepare_initial_state_entangled(configuration.state_probability)

        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.initialize(initial_state, [0, 1])
        circuit.reset(qreg_q[2])
        circuit.cry(2 * eta, qreg_q[1], qreg_q[2])
        circuit.cx(qreg_q[2], qreg_q[1])
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.rx(configuration.angle_rx1, qreg_q[1])
        circuit.ry(configuration.angle_ry1, qreg_q[1])
        circuit.rx(configuration.angle_rx0, qreg_q[0])
        circuit.ry(configuration.angle_ry0, qreg_q[0])
        circuit.measure([0, 1], creg_c)
        return circuit

    def _create_one_configuration(self,
                                  configuration: ChannelConfiguration,
                                  eta_group: List[float]) -> OneShotConfiguration:
        """ Creates a specific configuration setting a specific eta group """
        return OneShotEntangledConfiguration({
            'state_probability': cast(OneShotEntangledConfiguration, configuration).state_probability,
            'angle_rx0': cast(OneShotEntangledConfiguration, configuration).angle_rx0,
            'angle_ry0': cast(OneShotEntangledConfiguration, configuration).angle_ry0,
            'angle_rx1': cast(OneShotEntangledConfiguration, configuration).angle_rx1,
            'angle_ry1': cast(OneShotEntangledConfiguration, configuration).angle_ry1,
            'eta_group': eta_group
        })


from qcd.typings.configurations import Fidelities, ValidatedConfiguration
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

    def _get_max_counts_distribution_for_all_channels(
            self,
            all_channel_counts: List[dict],
            counts_distribution: List[float]) -> List[float]:
        """ returns the max counts between the max counts up to that moment and the circuit counts """
        if not all_channel_counts:
            return counts_distribution

        one_channel_counts = all_channel_counts.pop()
        max_counts = [0.0, 0.0]
        if '0' in one_channel_counts:
            max_counts[0] = max([counts_distribution[0], one_channel_counts['0']])
        if '1' in one_channel_counts:
            max_counts[1] = max([counts_distribution[1], one_channel_counts['1']])

        return self._get_max_counts_distribution_for_all_channels(all_channel_counts, max_counts)

    def _guess_probability_from_counts(self,
                                       eta_counts: List[dict],
                                       plays: int,
                                       eta_group_length: int) -> float:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        counts_distribution = self._get_max_counts_distribution_for_all_channels(eta_counts, [0.0, 0.0])
        return (counts_distribution[0] +
                counts_distribution[1]) / (plays * eta_group_length)

    def _create_one_circuit_without_measurement(self,
                                                configuration: ChannelConfiguration,
                                                eta: float) -> Tuple[ClassicalRegister, QuantumCircuit]:
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
        return creg_c, circuit

    def _add_measurement_to_one_circuit(self,
                                        creg_c: ClassicalRegister,
                                        circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.measure([0], creg_c[0])
        return circuit

    def _create_one_configuration(self,
                                  configuration: ChannelConfiguration,
                                  eta_group: List[float]) -> OneShotConfiguration:
        """ Creates a specific configuration setting a specific eta group """
        return OneShotConfiguration({
            'state_probability': cast(OneShotConfiguration, configuration).state_probability,
            'angle_rx': cast(OneShotConfiguration, configuration).angle_rx,
            'angle_ry': cast(OneShotConfiguration, configuration).angle_ry,
            'eta_group': eta_group
        })

    def _compute_upper_and_lower_fidelity_bounds(self, state_vectors: List[np.ndarray]) -> Fidelities:
        """ Computes upper and lower fidelity bounds from the given state vectors """
        raise NotImplementedError('Method not implemented yet')

    def _get_probabilities_and_etas_assigned_from_counts(self, counts: List[dict], plays: int,
                                                         eta_group_length: int) -> ValidatedConfiguration:
        """ Computes the validated probability from the 'counts' measured
            based on the guess strategy that is required to use and returns the
            etas assigned for each measured state
        """
        raise NotImplementedError('Method not implemented yet')

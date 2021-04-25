from qcd.typings.configurations import (Fidelities, MeasuredStatesEtaAssignment,
                                        MeasuredStatesCounts, ValidatedConfiguration)
from qcd.configurations.configuration import ChannelConfiguration
from qcd.configurations import OneShotConfiguration, OneShotEntangledConfiguration
from qiskit.quantum_info.states.utils import partial_trace
from . import OneShotCircuit
from .aux import upper_bound_fidelity, lower_bound_fidelity
from typing import List, Tuple, cast
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import itertools
import random
from qutip import fidelity, Qobj


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

    def _get_max_counts_and_etas_for_all_channels(
            self,
            all_channel_counts: List[dict],
            eta_assignments: MeasuredStatesEtaAssignment,
            measured_states_counts: MeasuredStatesCounts) -> Tuple[
                List[int],
                MeasuredStatesEtaAssignment,
                MeasuredStatesCounts]:
        """ returns the max counts between the max counts up to that moment and the circuit counts
            assigning the winner eta channel
        """
        max_counts = [0, 0, 0, 0]
        for current_eta, one_channel_counts in enumerate(all_channel_counts):
            if '00' in one_channel_counts:
                max_counts[0], eta_assignments['state_00'] = self._update_max_counts_and_eta_assignment(
                    eta_assignments['state_00'],
                    max_counts[0],
                    one_channel_counts['00'],
                    current_eta)
                measured_states_counts['state_00'].append(one_channel_counts['00'])
            if '01' in one_channel_counts:
                max_counts[1], eta_assignments['state_01'] = self._update_max_counts_and_eta_assignment(
                    eta_assignments['state_01'],
                    max_counts[1],
                    one_channel_counts['01'],
                    current_eta)
                measured_states_counts['state_01'].append(one_channel_counts['01'])
            if '10' in one_channel_counts:
                max_counts[2], eta_assignments['state_10'] = self._update_max_counts_and_eta_assignment(
                    eta_assignments['state_10'],
                    max_counts[2],
                    one_channel_counts['10'],
                    current_eta)
                measured_states_counts['state_10'].append(one_channel_counts['10'])
            if '11' in one_channel_counts:
                max_counts[3], eta_assignments['state_11'] = self._update_max_counts_and_eta_assignment(
                    eta_assignments['state_11'],
                    max_counts[3],
                    one_channel_counts['11'],
                    current_eta)
                measured_states_counts['state_11'].append(one_channel_counts['11'])
            measured_states_counts = self._update_null_measured_counts(one_channel_counts, measured_states_counts)
        return (max_counts, eta_assignments, measured_states_counts)

    def _update_max_counts_and_eta_assignment(self,
                                              max_eta: int,
                                              max_counts: int,
                                              new_value: int,
                                              new_eta: int) -> Tuple[int, int]:
        """ return the max value between the new value or the current and
            increase the eta index when new value is greater or
            when is the same value, increase it on 50% of the times
        """
        if new_value > max_counts:
            return (new_value, new_eta)
        if new_value == max_counts:
            random_assignment = random.choice([max_eta, new_eta])
            return (new_value, random_assignment)
        return (max_counts, max_eta)

    def _update_null_measured_counts(self,
                                     one_channel_counts: dict,
                                     measured_states_counts: MeasuredStatesCounts) -> MeasuredStatesCounts:
        if '00' not in one_channel_counts:
            measured_states_counts['state_00'].append(0)
        if '01' not in one_channel_counts:
            measured_states_counts['state_01'].append(0)
        if '10' not in one_channel_counts:
            measured_states_counts['state_10'].append(0)
        if '11' not in one_channel_counts:
            measured_states_counts['state_11'].append(0)
        return measured_states_counts

    def _get_probabilities_and_etas_assigned_from_counts(self,
                                                         eta_counts: List[dict],
                                                         plays: int,
                                                         eta_group_length: int) -> ValidatedConfiguration:
        """ Computes the validated probability from the 'counts' measured
            based on the guess strategy that is required to use and returns the
            etas assigned for each measured state
        """
        eta_assignments = MeasuredStatesEtaAssignment(state_00=-1,
                                                      state_01=-1,
                                                      state_10=-1,
                                                      state_11=-1)
        mesaured_states_counts = MeasuredStatesCounts(state_00=[],
                                                      state_01=[],
                                                      state_10=[],
                                                      state_11=[],
                                                      total_counts=(plays * eta_group_length))

        max_counts, eta_assignments, measured_states_counts = self._get_max_counts_and_etas_for_all_channels(
            eta_counts,
            eta_assignments,
            measured_states_counts=mesaured_states_counts)
        self._check_etas_assignments_and_etas_counts(eta_assignments, measured_states_counts, max_counts, eta_counts)
        global_average_success_probability, etas_probability = self._get_global_and_etas_probabilities(
            max_counts, eta_assignments, plays, eta_group_length)

        return ValidatedConfiguration(validated_probability=global_average_success_probability,
                                      etas_probability=etas_probability,
                                      measured_states_eta_assignment=eta_assignments,
                                      measured_states_counts=measured_states_counts)

    def _check_etas_assignments_and_etas_counts(self, etas_assignments: MeasuredStatesEtaAssignment,
                                                measured_states_counts: MeasuredStatesCounts,
                                                initial_max_counts, initial_eta_counts) -> None:
        """ Checks that all assigments are correct """

        for state in etas_assignments:
            assigned_eta = etas_assignments[state]  # type: ignore
            if assigned_eta == -1:
                continue
            max_eta = -1
            max_counts = -1
            max_eta_equal = -1
            max_eta_equal2 = -1
            for idx_eta, eta_counts in enumerate(measured_states_counts[state]):  # type: ignore
                if eta_counts > max_counts:
                    max_counts = eta_counts
                    max_eta = idx_eta
                    continue
                if eta_counts == max_counts and max_eta_equal == -1:
                    max_eta_equal = idx_eta
                    continue
                if eta_counts == max_counts and max_eta_equal != -1:
                    max_eta_equal2 = idx_eta
            if (assigned_eta != max_eta and
                assigned_eta != max_eta_equal and
                    assigned_eta != max_eta_equal2):
                assigned_max_counts = measured_states_counts[state][assigned_eta]  # type: ignore
                if max_eta_equal == -1:
                    raise ValueError('invalid eta assignment. eta assigned:' +
                                     f'eta_{assigned_eta} ' +
                                     f'with counts: {assigned_max_counts}, and ' +
                                     f'eta max is eta_{max_eta}$ with max counts: {max_counts}')
                if max_eta_equal2 == -1:
                    raise ValueError('invalid eta assignment. eta assigned:' +
                                     f'eta_{assigned_eta} ' +
                                     f'with counts: {assigned_max_counts}, and ' +
                                     f'eta max is eta_{max_eta} or eta_{max_eta_equal} ' +
                                     f'with max counts: {max_counts}')
                raise ValueError('invalid eta assignment. eta assigned:' +
                                 f'eta_{assigned_eta} ' +
                                 f'with counts: {assigned_max_counts}, and ' +
                                 f'eta max is eta_{max_eta} or eta_{max_eta_equal} or ' +
                                 f'eta_{max_eta_equal2} with max counts: {max_counts}')

    def _get_global_and_etas_probabilities(self,
                                           max_counts: List[int],
                                           etas_assignments: MeasuredStatesEtaAssignment,
                                           plays: int,
                                           eta_group_length: int) -> Tuple[float, List[float]]:
        """ return the global average success probability and the one from each eta channel """
        global_average_success_probability = (max_counts[0] +
                                              max_counts[1] +
                                              max_counts[2] +
                                              max_counts[3]) / (plays * eta_group_length)

        eta_counts = [0] * eta_group_length
        for idx, state in enumerate(etas_assignments):
            if state != 'state_00' and state != 'state_01' and state != 'state_10' and state != 'state_11':
                raise ValueError(f'invalid state: {state}')
            eta_assigned = etas_assignments[state]  # type: ignore
            if eta_assigned < -1 or eta_assigned > 2:
                raise ValueError(f'invalid eta: {eta_assigned}')
            if eta_assigned != -1:
                eta_counts[eta_assigned] += max_counts[idx]

        eta_probabilities = [eta_count / (plays * eta_group_length) for eta_count in eta_counts]
        if np.round(sum(eta_probabilities), 3) != np.round(global_average_success_probability, 3):
            raise ValueError(f'invalid probabilities! Globa avg: {global_average_success_probability} and ' +
                             f'sum probs: {sum(eta_probabilities)}')
        return (global_average_success_probability, eta_probabilities)

    def _create_one_circuit_without_measurement(self,
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
        return circuit

    def _add_measurement_to_one_circuit(self,
                                        creg_c: ClassicalRegister,
                                        circuit: QuantumCircuit) -> QuantumCircuit:
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

    def _compute_upper_and_lower_fidelity_bounds(self, state_vectors: List[np.ndarray]) -> Fidelities:
        """ Computes upper and lower fidelity bounds from the given state vectors """
        density_matrices = [partial_trace(np.outer(state_vector, state_vector.conj()), [2])
                            for state_vector in state_vectors]
        index_pairs = list(itertools.combinations(range(len(density_matrices)), 2))
        channel_fidelities = [fidelity(Qobj(density_matrices[first_index_pair].data), Qobj(
            density_matrices[second_index_pair].data)) for first_index_pair, second_index_pair in index_pairs]
        return Fidelities(upper_bound_fidelity=upper_bound_fidelity(channel_fidelities),
                          lower_bound_fidelity=lower_bound_fidelity(channel_fidelities))

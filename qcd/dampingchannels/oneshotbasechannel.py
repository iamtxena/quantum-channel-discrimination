from . import DampingChannel
from typing import Optional, List, Union, cast, Tuple
from ..backends import DeviceBackend
from ..configurations import OneShotSetupConfiguration
from ..executions import Execution, OneShotExecution
from ..typings import CloneSetup, OneShotResults
from ..optimizations import OneShotOptimization
from ..typings import (ResultStates,
                       ResultState,
                       ResultStatesReshaped,
                       ResultProbabilities,
                       ResultProbabilitiesOneChannel,
                       OptimizationSetup)
from ..typings.configurations import OptimalConfigurations
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.job import JobV1 as Job
from qiskit.result import Result
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info.states.utils import partial_trace
import numpy as np
import math
import matplotlib.pyplot as plt
from qcd import save_object_to_disk


class OneShotDampingChannel(DampingChannel):
    """ Representation of the One Shot Quantum Damping Channel """

    def __init__(self,
                 channel_setup_configuration: Optional[OneShotSetupConfiguration] = None) -> None:
        super().__init__(channel_setup_configuration)
        self.__channel_setup_configuration = channel_setup_configuration
        if channel_setup_configuration is not None:
            self._circuits, self._initial_states = self._create_all_circuits(channel_setup_configuration)

    def run(self, backend: Union[DeviceBackend, List[DeviceBackend]],
            iterations: Optional[int] = 1024, timeout: Optional[float] = None) -> Execution:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        if isinstance(backend, list):
            return OneShotExecution(list(map(lambda one_backend:
                                             self._execute_all_circuits_one_backend(one_backend,
                                                                                    iterations=iterations,
                                                                                    timeout=timeout),
                                             backend)))

        return OneShotExecution(self._execute_all_circuits_one_backend(backend, iterations, timeout))

    @staticmethod
    def find_optimal_configurations(optimization_setup: OptimizationSetup,
                                    clone_setup: Optional[CloneSetup]) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """

        optimal_configurations = OneShotOptimization(optimization_setup).find_optimal_configurations(clone_setup)
        if clone_setup is not None and clone_setup['file_name'] is not None:
            save_object_to_disk(optimal_configurations,
                                f"{clone_setup['file_name']}_{clone_setup['id_clone']}", clone_setup['path'])
        return optimal_configurations

    def plot_first_channel(self):
        return self._circuits[0][0].draw('mpl')

    def plot_fidelity(self) -> None:
        """ Displays the channel fidelity for 11 discrete attenuation levels ranging from
            0 (minimal attenuation) to 1 (maximal attenuation) """
        if self.__channel_setup_configuration is None:
            raise ValueError('SetupConfiguration must be specified')
        # Representation of fidelity
        initial_states = self._prepare_initial_states_fixed_phase(self.__channel_setup_configuration.angles_theta)

        fig = plt.figure(figsize=(25, 10))
        fig.suptitle('Fidelity Analysis', fontsize=20)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Output vs. Input. ', fontsize=14)
        ax1.set_xlabel('Input State ||' + '$\\alpha||^2 \\vert0\\rangle$', fontsize=14)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('Output versus $\\vert0\\rangle$ state', fontsize=14)
        ax2.set_xlabel('Input State ||' + '$\\alpha||^2 \\vert0\\rangle$', fontsize=14)

        angles_eta = self.__channel_setup_configuration.angles_eta
        index_channel = 0
        modulus_number = np.round(len(angles_eta) / 10)
        index_to_print = 0

        for idx, eta in enumerate(angles_eta):
            if ((index_to_print == 0 or len(angles_eta) <= modulus_number) or
                    (index_to_print != 0 and index_channel % modulus_number == 0 and index_to_print < 10) or
                    (idx == len(angles_eta) - 1)):
                X = []
                Y = []
                Z = []

                fidelity = self._calculate_fidelity(initial_states, eta)
                for i in range(len(initial_states)):
                    X.append((initial_states[i][0] * np.conj(initial_states[i][0])).real)
                    Y.append(fidelity[i][0])
                    Z.append(fidelity[i][1])
                label = f'$\lambda = {np.round(math.sin(eta)**2, 3)}$'
                ax1.plot(X, Y, label=label)
                ax2.plot(X, Z, label=label)
                index_to_print += 1
            index_channel += 1

        ax1.legend()
        ax2.legend()
        plt.show()

    def _calculate_fidelity(self, initial_states: List[Tuple[float, complex]], eta: float) -> List[Tuple[float, float]]:
        qreg_q = QuantumRegister(2, 'q')
        creg_c = ClassicalRegister(1, 'c')
        circ = QuantumCircuit(qreg_q, creg_c)
        backend_sim = Aer.get_backend('statevector_simulator')

        stavec = []
        fidelity = []
        for i in range(len(initial_states)):
            circ.initialize([initial_states[i][0], initial_states[i][1]], qreg_q[0])
            circ.reset(qreg_q[1])
            circ.cry(2 * eta, qreg_q[0], qreg_q[1])
            circ.cx(qreg_q[1], qreg_q[0])
            res = execute(circ, backend_sim).result()
            sv = res.get_statevector(circ)
            stavec.append([sv[0], sv[1]])
            fidelity.append([state_fidelity([initial_states[i][0], initial_states[i][1]],
                                            partial_trace(np.outer(sv, sv), [1]), validate=True),
                             state_fidelity([1, 0], partial_trace(np.outer(sv, sv), [1]), validate=True)])
        return np.array(fidelity)

    def _create_all_circuits(self,
                             channel_setup_configuration: OneShotSetupConfiguration) -> Tuple[List[QuantumCircuit],
                                                                                              ResultStates]:
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
        return circuits, initial_states

    def _prepare_initial_states(self, angles_theta: List[float], angles_phase: List[float]) -> ResultStates:
        """ Prepare initial states to pass through the circuit """
        # As we have to provide the state values to initialize the qreg[0] we have to do a conversion
        # from angles in the sphere to statevector amplitudes. These statevectors will be the combination of
        # Zero_Amplitude*|0> plus One_Amplitude*|1>
        initial_states_zero_amplitude = []
        initial_states_one_amplitude = []

        for theta in angles_theta:
            for phase in angles_phase:
                initial_states_zero_amplitude.append(math.cos(theta))
                initial_states_one_amplitude.append((math.sin(theta) * math.e**(1j * phase)))
        return ResultStates(zero_amplitude=initial_states_zero_amplitude,
                            one_amplitude=initial_states_one_amplitude)

    def _prepare_initial_states_fixed_phase(self, angles_theta: List[float]) -> List[Tuple[float, complex]]:
        """ Prepare initial states to pass through the circuit """
        """ with a fixed Phase angle. Only moving across Theta """

        result_initial_states = self._prepare_initial_states(angles_theta, [0])
        initial_states = []

        for idx, _ in enumerate(result_initial_states['zero_amplitude']):
            initial_states.append([
                result_initial_states['zero_amplitude'][idx],
                result_initial_states['one_amplitude'][idx]
            ])

        return np.array(initial_states)

    def _execute_all_circuits_one_backend(self, backend: DeviceBackend, iterations: Optional[int] = 1024,
                                          timeout: Optional[float] = None) -> OneShotResults:
        if self.__channel_setup_configuration is None:
            raise ValueError('SetupConfiguration must be specified')

        final_states_list = []
        final_states_reshaped_list = []
        total_lambdas_per_state = []
        total_x_input_0 = []
        total_x_input_1 = []
        total_z_output_0 = []
        total_z_output_1 = []
        attenuation_factors = self.__channel_setup_configuration.attenuation_factors

        for idx, circuits_one_lambda in enumerate(self._circuits):
            job = cast(Job, execute(circuits_one_lambda, backend=backend.backend, shots=iterations))
            if idx % 10 == 0:
                print(
                    'Execution using channel with ' + u"\u03BB" '=' +
                    f'{np.round(self.__channel_setup_configuration.attenuation_factors[idx], 1)} ' +
                    f'launched to {job.backend()} with id={job.job_id()}')
            job.wait_for_final_state(timeout=timeout)
            final_states, probabilities_one_channel = self._process_job_one_lambda(job, self._initial_states)
            final_states_reshaped = self._compute_finale_state_vector_coords_reshaped(
                final_states['zero_amplitude'],
                self.__channel_setup_configuration.angles_phase,
                self.__channel_setup_configuration.points_theta,
                self.__channel_setup_configuration.points_phase)
            final_states_list.append(final_states)
            total_x_input_0.append(probabilities_one_channel['x_input_0'])
            total_x_input_1.append(probabilities_one_channel['x_input_1'])
            total_z_output_0.append(probabilities_one_channel['z_output_0'])
            total_z_output_1.append(probabilities_one_channel['z_output_1'])
            final_states_reshaped_list.append(final_states_reshaped)
            total_lambdas_per_state.append(
                [self.__channel_setup_configuration.attenuation_factors[idx]] * len(final_states['zero_amplitude']))

        return OneShotResults(final_states=final_states_list,
                              final_states_reshaped=final_states_reshaped_list,
                              probabilities=ResultProbabilities(x_input_0=np.array(total_x_input_0),
                                                                x_input_1=np.array(total_x_input_1),
                                                                z_output_0=np.array(total_z_output_0),
                                                                z_output_1=np.array(total_z_output_1)),
                              attenuation_factors=attenuation_factors,
                              attenuation_factor_per_state=np.array(total_lambdas_per_state),
                              backend_name=backend.backend.name())

    def _process_job_one_lambda(self,
                                job: Job,
                                initial_states: ResultStates) -> Tuple[ResultStates,
                                                                       ResultProbabilitiesOneChannel]:
        results = cast(Result, job.result())
        final_states = ResultStates(zero_amplitude=[], one_amplitude=[])
        x_input_0 = []
        x_input_1 = []
        z_output_0 = []
        z_output_1 = []

        for initial_states_idx, _ in enumerate(initial_states["zero_amplitude"]):
            counts = results.get_counts(initial_states_idx)
            final_state = self._convert_counts_to_final_state(counts)
            final_states["zero_amplitude"].append(final_state["zero_amplitude"])
            final_states["one_amplitude"].append(final_state["one_amplitude"])
            z_output_0.append(final_state["zero_amplitude"]**2)
            z_output_1.append(final_state["one_amplitude"]**2)
            Prob_0 = initial_states["zero_amplitude"][initial_states_idx] * \
                np.conj(initial_states["zero_amplitude"][initial_states_idx])
            Prob_1 = initial_states["one_amplitude"][initial_states_idx] * \
                np.conj(initial_states["one_amplitude"][initial_states_idx])
            x_input_0.append(Prob_0.real)
            x_input_1.append(Prob_1.real)

        return (final_states,
                ResultProbabilitiesOneChannel(x_input_0=x_input_0,
                                              x_input_1=x_input_1,
                                              z_output_0=z_output_0,
                                              z_output_1=z_output_1))

    def _compute_finale_state_vector_coords_reshaped(self,
                                                     amplitudes_vector: List[float],
                                                     angles_phase: List[float],
                                                     n_points_theta: int,
                                                     n_points_phase: int) -> ResultStatesReshaped:
        """ Compute the final state reshaped coordinates from a given amplitudes """
        final_state_vector_coords_x = []
        final_state_vector_coords_y = []
        final_state_vector_coords_z = []
        for amplitude in amplitudes_vector:
            theta_i = np.arccos(amplitude)
            theta_bloch = 2 * theta_i
            final_state_vector_coords_z.append(np.cos(theta_bloch))

        # Reshaping matrices X, Y and Z in right dimensions to be represented
        top_z = max(final_state_vector_coords_z)
        min_z = min(final_state_vector_coords_z)
        radius = (top_z - min_z) / 2
        center_z = 1 - radius

        for idx, _ in enumerate(final_state_vector_coords_z):
            new_theta_i = np.arccos((final_state_vector_coords_z[idx] - center_z) / radius)
            new_phase_i = angles_phase[idx % n_points_phase]
            final_state_vector_coords_x.append(np.sqrt(radius) * np.sin(new_theta_i) * np.cos(new_phase_i))
            final_state_vector_coords_y.append(np.sqrt(radius) * np.sin(new_theta_i) * np.sin(new_phase_i))

        return ResultStatesReshaped(reshaped_coords_x=np.reshape(final_state_vector_coords_x, (n_points_theta,
                                                                                               n_points_phase)),
                                    reshaped_coords_y=np.reshape(final_state_vector_coords_y,
                                                                 (n_points_theta, n_points_phase)),
                                    reshaped_coords_z=np.reshape(final_state_vector_coords_z,
                                                                 (n_points_theta, n_points_phase)),
                                    center=center_z)

    def _convert_counts_to_final_state(self, counts: dict) -> ResultState:
        """ Convert the results of a simulation to a final state: aplha*|0> + beta*|1> """
        counts_zero = 0
        counts_one = 0

        if "0" in counts:
            counts_zero = counts["0"]
        if "1" in counts:
            counts_one = counts["1"]

        total_cycles = counts_zero + counts_one

        return ResultState(zero_amplitude=np.sqrt(counts_zero / total_cycles),
                           one_amplitude=np.sqrt(counts_one / total_cycles))

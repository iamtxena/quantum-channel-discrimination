from . import DampingChannel
from typing import Optional, List, Union, cast, Tuple
from ..backends import DeviceBackend
from ..configurations import OneShotSetupConfiguration
from ..executions import Execution, OneShotExecution
from ..results import OneShotResults
from ..optimizations import OptimizationSetup, OptimalConfigurations
from ..typings import ResultStates, ResultState, ResultStatesReshaped
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.job import JobV1 as Job
from qiskit.result import Result
import numpy as np
import math


class OneShotDampingChannel(DampingChannel):
    """ Representation of the One Shot Quantum Damping Channel """

    def __init__(self,
                 channel_setup_configuration: OneShotSetupConfiguration,
                 optimization_setup: Optional[OptimizationSetup] = None) -> None:
        super().__init__(channel_setup_configuration, optimization_setup)
        self.__channel_setup_configuration = channel_setup_configuration
        self.__optimization_setup = optimization_setup

        self._circuits, self._initial_states = self._create_all_circuits(channel_setup_configuration)

    def run(self, backend: Union[DeviceBackend, List[DeviceBackend]],
            iterations: Optional[int] = 1024, timeout: Optional[float] = None) -> Union[Execution, List[Execution]]:
        """ Runs all the experiments using the configured circuits launched to the provided backend """
        if isinstance(backend, list):
            return list(map(lambda one_backend:
                            OneShotExecution(self._execute_all_circuits_one_backend(one_backend,
                                                                                    iterations=iterations,
                                                                                    timeout=timeout)),
                            backend))

        return OneShotExecution(self._execute_all_circuits_one_backend(backend, iterations, timeout))

    def find_optimal_configurations(self) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        raise NotImplementedError('Method not implemented')

    def plot_first_channel(self):
        return self._circuits[0][0].draw('mpl')

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

    def _execute_all_circuits_one_backend(self, backend: DeviceBackend, iterations: Optional[int] = 1024,
                                          timeout: Optional[float] = None) -> OneShotResults:
        results = []
        for circuits_one_lambda in self._circuits:
            job = cast(Job, execute(circuits_one_lambda, backend=backend.backend, shots=iterations))
            print(f'Job launched to {job.backend()} with id = {job.job_id()}')
            job.wait_for_final_state(timeout=timeout)
            final_state = self._process_information_from_job_one_lambda(job, self._initial_states)
            final_state_reshaped = self._compute_finale_state_vector_coords_reshaped(
                final_state['zero_amplitude'],
                self.__channel_setup_configuration.angles_phase,
                self.__channel_setup_configuration.points_theta,
                self.__channel_setup_configuration.points_phase)
            results.append(final_state_reshaped)
        return OneShotResults(results, backend.backend.name())

    def _process_information_from_job_one_lambda(self, job: Job, initial_states: ResultStates) -> ResultStates:
        results = cast(Result, job.result())
        final_states = ResultStates(zero_amplitude=[], one_amplitude=[])

        for initial_states_idx, _ in enumerate(initial_states["zero_amplitude"]):
            counts = results.get_counts(initial_states_idx)
            final_state = self._convert_counts_to_final_state(counts)
            final_states["zero_amplitude"].append(final_state["zero_amplitude"])
            final_states["one_amplitude"].append(final_state["one_amplitude"])

        return final_states

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

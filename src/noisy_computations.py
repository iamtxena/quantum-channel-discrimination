from qiskit.providers.ibmq import least_busy
from qiskit import IBMQ, Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.test.mock import FakeAthens
from qiskit.providers.aer import QasmSimulator
from typing import Optional, List, Union
from .quantum_channel_discrimination import (
    prepareInitialStates, computeFinaleStateVectorCoordsReshaped, convertCountsToFinalState, draw_cube)
import numpy as np
import matplotlib.pyplot as plt
import enum


class Device(enum.Enum):
    simulator = 1
    fake_noise_simulator = 2
    ibmq = 3


def get_least_busy_device():
    provider = IBMQ.load_account()
    # Load IBM Q account and get the least busy backend device
    least_busy_device = least_busy(provider.backends(
        filters=lambda x: x.configuration().n_qubits >= 2, simulator=False))
    print("Running on current least busy device: ", least_busy_device)
    return least_busy_device


def get_backend(device=Device.simulator):
    backend = Aer.get_backend('qasm_simulator')

    if device == Device.fake_noise_simulator:
        athens_noisy_backend = FakeAthens()
        backend = QasmSimulator.from_backend(athens_noisy_backend)
    if device == Device.ibmq:
        backend = get_least_busy_device()

    return backend


def create_all_circuits(lambdas, points_theta, points_phase, out_rx_angle=0, out_ry_angle=0):
    # Create 2 qbits circuit and 1 output classical bit
    qreg_q = QuantumRegister(2, 'q')
    creg_c = ClassicalRegister(1, 'c')
    # First we generate the angles which will help the draw the sphere
    angles_theta = np.mgrid[0:np.pi / 2:points_theta * 1j]
    angles_phase = np.mgrid[0:2 * np.pi:points_phase * 1j]

    circuits = []
    # Initialize circuit with desired initial_state
    initial_states = prepareInitialStates(angles_theta, angles_phase)

    for one_lambda in lambdas:
        for index_initial_state, _ in enumerate(initial_states["zeroAmplitude"]):
            circuit = QuantumCircuit(qreg_q, creg_c)
            circuit.initialize([initial_states["zeroAmplitude"][index_initial_state],
                                initial_states["oneAmplitude"][index_initial_state]], qreg_q[0])
            circuit.reset(qreg_q[1])
            circuit.cry(2 * np.arcsin(np.sqrt(one_lambda)), qreg_q[0], qreg_q[1])
            circuit.cx(qreg_q[1], qreg_q[0])
            circuit.rx(out_rx_angle, qreg_q[0])
            circuit.ry(out_ry_angle, qreg_q[0])
            circuit.measure(qreg_q[0], creg_c[0])
            circuits.append(circuit)
    return circuits, initial_states, angles_phase


def process_information_from_job_one_lambda(job, initial_states):
    results = job.result()
    final_states = {
        "zeroAmplitude": [],
        "oneAmplitude": [],
    }

    for initial_states_idx, _ in enumerate(initial_states["zeroAmplitude"]):
        counts = results.get_counts(initial_states_idx)
        final_state = convertCountsToFinalState(counts)
        final_states["zeroAmplitude"].append(final_state["zeroAmplitude"])
        final_states["oneAmplitude"].append(final_state["oneAmplitude"])

    return final_states


def run_all_circuits_one_lambda_one_device(one_lambda, points_theta, points_phase,
                                           iterations, out_rx_angle, out_ry_angle,
                                           timeout: Optional[float] = None,
                                           device: Optional[Device] = Device.simulator):
    backend = get_backend(device)
    circuits, initial_states, angles_phase = create_all_circuits(
        [one_lambda], points_theta, points_phase, out_rx_angle, out_ry_angle)
    job = execute(circuits, backend=backend, shots=iterations)
    print(f'Job launched to {job.backend()} with id = {job.job_id()}')
    job.wait_for_final_state(timeout=timeout)
    final_state = process_information_from_job_one_lambda(job, initial_states)
    final_state_reshaped = computeFinaleStateVectorCoordsReshaped(final_state.get('zeroAmplitude'),
                                                                  angles_phase,
                                                                  points_theta,
                                                                  points_phase)
    final_state_reshaped['device'] = device
    return final_state_reshaped


def run_all_circuits_one_lambda(one_lambda, points_theta, points_phase, iterations=1024,
                                out_rx_angle=0, out_ry_angle=0, timeout: Optional[float] = None,
                                device: Optional[Union[Device, List[Device]]] = Device.simulator):

    if isinstance(device, list):
        return list(run_all_circuits_one_lambda_one_device(one_lambda, points_theta,
                                                           points_phase, iterations,
                                                           out_rx_angle, out_ry_angle,
                                                           timeout, one_device) for one_device in device)

    return list(run_all_circuits_one_lambda_one_device(one_lambda, points_theta,
                                                       points_phase, iterations,
                                                       out_rx_angle, out_ry_angle,
                                                       timeout, device))


def plot_all_wireframe_bloch_list_one_lambda(all_channels_final_states_reshaped, one_lambda, rows=3, cols=3):
    fig = plt.figure(figsize=(20, 25))
    for idx, finalStatesReshaped in enumerate(all_channels_final_states_reshaped):
        # set up the axes for the second plot
        ax = fig.add_subplot(rows, cols, 1 + idx, projection='3d')
        draw_cube(ax)
        # draw final states
        ax.plot_wireframe(finalStatesReshaped['reshapedCoordsX'],
                          finalStatesReshaped['reshapedCoordsY'],
                          finalStatesReshaped['reshapedCoordsZ'], color="r")
        name_device = 'a Simulator'
        if finalStatesReshaped.get('device') == Device.fake_noise_simulator:
            name_device = 'a Fake Noise Simulator'
        if finalStatesReshaped.get('device') == Device.ibmq:
            name_device = 'IBM Q'
        title = f"Output States executed on {name_device}\n Channel $\lambda= {one_lambda}$"
        ax.set_title(title)
        # draw center
        ax.scatter([0], [0], finalStatesReshaped["center"], color="g", s=50)

    plt.show()

from itertools import product, combinations
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.quantum_info import state_fidelity
from numpy import pi
from math import e
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import math


def prepareInitialStates(inAnglesTheta, inAnglesPhase):
    """ Prepare initial states to pass through the circuit """
    # As we have to provide the state values to initialize the qreg[0] we have to do a conversion
    # from angles in the sphere to statevector amplitudes. These statevectors will be the combination of
    # Zero_Amplitude*|0> plus One_Amplitude*|1>
    initialStatesZeroAmplitude = []
    initialStatesOneAmplitude = []

    for theta in inAnglesTheta:
        for phase in inAnglesPhase:
            initialStatesZeroAmplitude.append(math.cos(theta / 2) * (1 + 0j))
            initialStatesOneAmplitude.append((math.sin(theta / 2) * e**(1j * phase) + 0 + 0j))
    return {
        "zeroAmplitude": initialStatesZeroAmplitude,
        "oneAmplitude": initialStatesOneAmplitude,
    }


def convertCountsToFinalState(inCounts):
    """ Convert the results of a simulation to a final state: aplha*|0> + beta*|1> """
    countsZero = 0
    countsOne = 0

    if "0" in inCounts:
        countsZero = inCounts["0"]
    if "1" in inCounts:
        countsOne = inCounts["1"]

    totalCycles = countsZero + countsOne

    return {
        "zeroAmplitude": np.sqrt(countsZero / totalCycles),
        "oneAmplitude": np.sqrt(countsOne / totalCycles),
    }


def computeStateVectorCoordsReshaped(amplitudesVector, inputAnglesPhase, nPointsTheta, nPointsPhase):
    """ Compute the reshaped coordinates from a given amplitudes """
    initialStateVectorCoordsX = []
    initialStateVectorCoordsY = []
    initialStateVectorCoordsZ = []
    for indexAmplitudes in range(len(amplitudesVector)):
        Theta_i = 2 * np.arccos(amplitudesVector[indexAmplitudes])
        Phase_i = inputAnglesPhase[indexAmplitudes % nPointsPhase]
        initialStateVectorCoordsX.append(np.sin(Theta_i) * np.cos(Phase_i))
        initialStateVectorCoordsY.append(np.sin(Theta_i) * np.sin(Phase_i))
        initialStateVectorCoordsZ.append(np.cos(Theta_i))
    # Reshaping matrices X, Y and Z in right dimensions to be represented
    return {
        'reshapedCoordsX': np.reshape(initialStateVectorCoordsX, (nPointsTheta, nPointsPhase)),
        'reshapedCoordsY': np.reshape(initialStateVectorCoordsY, (nPointsTheta, nPointsPhase)),
        'reshapedCoordsZ': np.reshape(initialStateVectorCoordsZ, (nPointsTheta, nPointsPhase)),
    }


def computeFinaleStateVectorCoordsReshaped(amplitudesVector, inputAnglesPhase, nPointsTheta, nPointsPhase):
    """ Compute the final state reshaped coordinates from a given amplitudes """
    finalStateVectorCoordsX = []
    finalStateVectorCoordsY = []
    finalStateVectorCoordsZ = []
    for indexAmplitudes in range(len(amplitudesVector)):
        Theta_i = 2 * np.arccos(amplitudesVector[indexAmplitudes])
        finalStateVectorCoordsZ.append(np.cos(Theta_i))
    # Reshaping matrices X, Y and Z in right dimensions to be represented
    top_z = max(finalStateVectorCoordsZ)
    min_z = min(finalStateVectorCoordsZ)
    radius = (top_z - min_z) / 2
    center_z = 1 - radius

    for newIndexAmplitudes in range(len(amplitudesVector)):
        newTheta_i = np.arccos((finalStateVectorCoordsZ[newIndexAmplitudes] - center_z) / radius)
        newPhase_i = inputAnglesPhase[newIndexAmplitudes % nPointsPhase]
        finalStateVectorCoordsX.append(np.sqrt(radius) * np.sin(newTheta_i) * np.cos(newPhase_i))
        finalStateVectorCoordsY.append(np.sqrt(radius) * np.sin(newTheta_i) * np.sin(newPhase_i))
    return {
        'reshapedCoordsX': np.reshape(finalStateVectorCoordsX, (nPointsTheta, nPointsPhase)),
        'reshapedCoordsY': np.reshape(finalStateVectorCoordsY, (nPointsTheta, nPointsPhase)),
        'reshapedCoordsZ': np.reshape(finalStateVectorCoordsZ, (nPointsTheta, nPointsPhase)),
        'center': center_z
    }


def draw_cube(axes):
    """ Draw a cube passing axes as a parameter """
    r = [-1, 1]
    for s, l in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - l)) == r[1] - r[0]:
            axes.plot3D(*zip(s, l), color="w")


def runDampingChannelSimulation(anglesEta, pointsTheta, pointsPhase,
                                iterations, backend, state_backend_sim, out_rz_angle=0, out_ry_angle=0):
    # Create 2 qbits circuit and 1 output classical bit
    qreg_q = QuantumRegister(2, 'q')
    creg_c = ClassicalRegister(1, 'c')
    # First we generate the angles which will help the draw the sphere
    anglesTheta = np.mgrid[0:pi:pointsTheta * 1j]
    anglesPhase = np.mgrid[0:2 * pi:pointsPhase * 1j]

    totalResults = []
    totalCounts = []
    totalCircuits = []
    totalFinalStates = []
    total_state_vectors = []

    Z_Output0 = []
    Z_Output1 = []
    X_Input0 = []
    X_Input1 = []
    Y_Eta = []

    initialStates = prepareInitialStates(anglesTheta, anglesPhase)

    # Initialize circuit with desired initial_state
    for eta in anglesEta:
        print("Simulating channel with " + u"\u03B7" + " = " + str(int(math.degrees(eta))) + u"\u00B0")
        circuitResultsSpecificChannel = []
        countsSpecificChannel = []
        circuitSpecificChannel = []
        finalStates = {
            "zeroAmplitude": [],
            "oneAmplitude": [],
        }

        nested_Z_Output0 = []
        nested_Z_Output1 = []
        nested_X_Input0 = []
        nested_X_Input1 = []
        nested_Y_Eta = []
        Z_Output0.append(nested_Z_Output0)
        Z_Output1.append(nested_Z_Output1)
        X_Input0.append(nested_X_Input0)
        X_Input1.append(nested_X_Input1)
        Y_Eta.append(nested_Y_Eta)
        loop_counter = 0
        channel_state_vectors = []

        for indexInitialState in range(len(initialStates["zeroAmplitude"])):
            phase_cycle = indexInitialState // (len(anglesPhase) * len(anglesTheta))
            if loop_counter < phase_cycle:
                loop_counter = phase_cycle
                nested_Z_Output0 = []
                nested_Z_Output1 = []
                nested_X_Input0 = []
                nested_X_Input1 = []
                nested_Y_Eta = []

            circuit = QuantumCircuit(qreg_q, creg_c)
            circuit.initialize([initialStates["zeroAmplitude"][indexInitialState],
                                initialStates["oneAmplitude"][indexInitialState]], qreg_q[0])
            circuit.reset(qreg_q[1])
            # measuring the Initial StateVector
            input_state_vector = execute(circuit, state_backend_sim).result().get_statevector(circuit)
            circuit.cry(eta, qreg_q[0], qreg_q[1])
            circuit.cx(qreg_q[1], qreg_q[0])
            circuit.rz(out_rz_angle, qreg_q[0])
            circuit.ry(out_ry_angle, qreg_q[0])
            circuit.measure(qreg_q[0], creg_c[0])
            circuitSpecificChannel.append(circuit)
            # execute circuit on backends
            job_sim = execute(circuit, backend, shots=iterations)
            output_state_vector = execute(circuit, state_backend_sim,
                                          shots=iterations).result().get_statevector(circuit)
            # get results
            results_sim = job_sim.result()
            circuitResultsSpecificChannel.append(results_sim)
            counts = results_sim.get_counts(circuit)
            countsSpecificChannel.append(counts)
            finalState = convertCountsToFinalState(counts)
            finalStates["zeroAmplitude"].append(finalState["zeroAmplitude"])
            finalStates["oneAmplitude"].append(finalState["oneAmplitude"])
            nested_Z_Output0.append(finalState["zeroAmplitude"]**2)
            nested_Z_Output1.append(finalState["oneAmplitude"]**2)
            Prob_0 = initialStates["zeroAmplitude"][indexInitialState] * \
                np.conj(initialStates["zeroAmplitude"][indexInitialState])
            Prob_1 = initialStates["oneAmplitude"][indexInitialState] * \
                np.conj(initialStates["oneAmplitude"][indexInitialState])
            nested_X_Input0.append(Prob_0.real)
            nested_X_Input1.append(Prob_1.real)
            nested_Y_Eta.append(eta)
            channel_state_vectors.append({
                "input_state_vector": input_state_vector,
                "output_state_vector": output_state_vector
            })

            # append the results for a specific channel
        totalCircuits.append(circuitSpecificChannel)
        totalFinalStates.append(finalStates)
        totalCounts.append(countsSpecificChannel)
        totalResults.append(circuitResultsSpecificChannel)
        total_state_vectors.append(channel_state_vectors)

    return {
        "initialStates": initialStates,
        "totalCircuits": totalCircuits,
        "totalFinalStates": totalFinalStates,
        "totalCounts": totalCounts,
        "totalResults": totalResults,
        "anglesPhase": anglesPhase,
        "Z_Output0": np.array(Z_Output0),
        "Z_Output1": np.array(Z_Output1),
        "X_Input0": np.array(X_Input0),
        "X_Input1": np.array(X_Input1),
        "Y_Eta": np.array(Y_Eta),
        "total_state_vectors": total_state_vectors
    }


def probability_from_amplitude(amplitude) -> int:
    return np.round(np.linalg.norm(amplitude) ** 2, 2)


def extract_amplitudes_states(initial_states, final_states, number_execution, output_probabilities):
    total_amplitudes_to_extract = 20
    initial_state_zero_amplitude = "input_0"
    initial_state_one_amplitude = "input_1"
    final_state_zero_amplitude = "output_0"
    final_state_one_amplitude = "output_1"
    zero_amplitude = "zeroAmplitude"
    one_amplitude = "oneAmplitude"

    output_probabilities.append({
        initial_state_zero_amplitude: probability_from_amplitude(initial_states[zero_amplitude][0]),
        final_state_zero_amplitude: probability_from_amplitude(final_states[number_execution][zero_amplitude][0]),
        initial_state_one_amplitude: probability_from_amplitude(initial_states[one_amplitude][0]),
        final_state_one_amplitude: probability_from_amplitude(final_states[number_execution][one_amplitude][0])
    })

    for index in range(2, total_amplitudes_to_extract):
        index_state = int((len(final_states[number_execution][zero_amplitude]) / total_amplitudes_to_extract) * index)
        output_probabilities.append({
            initial_state_zero_amplitude: probability_from_amplitude(initial_states[zero_amplitude][index_state]),
            final_state_zero_amplitude:
            probability_from_amplitude(final_states[number_execution][zero_amplitude][index_state]),
            initial_state_one_amplitude: probability_from_amplitude(initial_states[one_amplitude][index_state]),
            final_state_one_amplitude: probability_from_amplitude(
                final_states[number_execution][one_amplitude][index_state])
        })

    final_index_state = int((len(final_states[number_execution][zero_amplitude]) /
                             total_amplitudes_to_extract) * total_amplitudes_to_extract) - 1
    output_probabilities.append({
        initial_state_zero_amplitude: probability_from_amplitude(initial_states[zero_amplitude][final_index_state]),
        final_state_zero_amplitude:
        probability_from_amplitude(final_states[number_execution][zero_amplitude][final_index_state]),
        initial_state_one_amplitude: probability_from_amplitude(initial_states[one_amplitude][final_index_state]),
        final_state_one_amplitude: probability_from_amplitude(
            final_states[number_execution][one_amplitude][final_index_state])
    })
    return output_probabilities


def plot_probabilities(input_0, input_1, output_0, output_1, index_angle, anglesEta):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True)

    axs[0][0].bar(range(len(input_0)), input_0, color="orange")
    # axs[0][0].set_title("Input " + "$\\alpha|0\\rangle$")
    axs[0][0].set_xlabel("Input " + "$\\alpha|0\\rangle$")
    axs[0][0].set_ylabel("Probabilities")
    axs[0][1].bar(range(len(input_1)), input_1, color="orange")
    # axs[0][1].set_title("Input " + "$\\beta|1\\rangle$")
    axs[0][1].set_xlabel("Input " + "$\\beta|1\\rangle$")
    axs[1][0].bar(range(len(output_0)), output_0)
    # axs[1][0].set_title("Output " + "$\\alpha|0\\rangle$")
    axs[1][0].set_xlabel("Output " + "$\\alpha|0\\rangle$")
    axs[1][0].set_ylabel("Probabilities")
    axs[1][1].bar(range(len(output_1)), output_1)
    # axs[1][1].set_title("Output " + "$\\beta|1\\rangle$")
    axs[1][1].set_xlabel("Output " + "$\\beta|1\\rangle$")

    for row_i in range(len(axs)):
        for col_j in range(len(axs[row_i])):
            for label in axs[row_i][col_j].get_xaxis().get_ticklabels():
                label.set_visible(False)

    fig.suptitle(
        f"Probabilities Input vs Output Channel $\eta={str(int(math.degrees(anglesEta[index_angle])))} \degree$")
    fig.set_size_inches(7, 5)

    plt.show()


def prepare_and_plot_probabilities(anglesEta, initialStates, totalFinalStates):
    # extract just a sample of all the computed probabilities
    total_output_probabilities = []
    for indexCountsToPrint in range(len(anglesEta)):
        output_probabilities = []
        output_probabilities = extract_amplitudes_states(
            initialStates, totalFinalStates, indexCountsToPrint, output_probabilities)
        total_output_probabilities.append(output_probabilities)
    final_probabilities = []
    # prepare data to be plotted as bars
    for index in range(len(total_output_probabilities)):
        input_0 = []
        input_1 = []
        output_0 = []
        output_1 = []
        for prob_i in total_output_probabilities[index]:
            input_0.append(prob_i["input_0"])
            input_1.append(prob_i["input_1"])
            output_0.append(prob_i["output_0"])
            output_1.append(prob_i["output_1"])
        final_probabilities.append({
            "input_0": input_0,
            "input_1": input_1,
            "output_0": output_0,
            "output_1": output_1,
        })
    for index in range(len(final_probabilities)):
        plot_probabilities(final_probabilities[index]["input_0"], final_probabilities[index]["input_1"],
                           final_probabilities[index]["output_0"], final_probabilities[index]["output_1"],
                           index, anglesEta)


def plot_surface_probabilities(X_Input0, X_Input1, Z_Output0, Z_Output1, eta_degrees):
    # Representation of output probabilities for all circuit in a 3d plot
    fig = plt.figure(figsize=(25, 35))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X_Input0, eta_degrees, Z_Output0, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    ax.set_title("Output Probabilities for $\\vert0\\rangle$", fontsize=30)
    plt.ylabel("$\eta$ (degrees)")
    plt.xlabel("Input State ||" + "$\\alpha||^2 |0\\rangle$")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(X_Input1, eta_degrees, Z_Output1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    ax.set_title("Output Probabilities for $\\vert1\\rangle$", fontsize=30)
    plt.ylabel("$\eta$ (degrees)")
    plt.xlabel("Input State ||" + "$\\beta||^2 |1\\rangle$")

    plt.show()


def plot_probabilities2(data_0, data_1, color, angle):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)
    axs[0].set_ylabel("Probabilities")
    axs[0].grid(zorder=0.1)
    axs[0].bar(range(len(data_0)), data_0, color=color)
    axs[1].grid(zorder=0.1)
    axs[1].bar(range(len(data_1)), data_1, color=color)
    if angle == "None":
        fig.suptitle("Input States Probabilities for any " + "$\eta$" + " value", fontsize=50)
        axs[0].set_xlabel("Input " + "$\\alpha|0\\rangle$")
        axs[1].set_xlabel("Input " + "$\\beta|1\\rangle$")
    else:
        fig.suptitle("Output States Probabilities for " + "$\eta=" +
                     str(int(math.degrees(angle))) + "\degree$", fontsize=50)
        axs[0].set_xlabel("Output " + "$\\alpha|0\\rangle$")
        axs[1].set_xlabel("Output " + "$\\beta|1\\rangle$")

    fig.set_size_inches(15, 5)
    plt.show()


def plot_wireframe_blochs(initialStatesReshaped, allChannelsFinalStatesReshaped, anglesEta, rows=3, cols=3):
    fig = plt.figure(figsize=(20, 25))
    # ===============
    #  First subplot
    # ===============
    # set up the axes for the first plot
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    draw_cube(ax)

    # draw initial states
    ax.plot_wireframe(initialStatesReshaped['reshapedCoordsX'],
                      initialStatesReshaped['reshapedCoordsY'],
                      initialStatesReshaped['reshapedCoordsZ'], color="c")
    ax.set_title("Input States")
    # draw center
    ax.scatter([0], [0], [0], color="g", s=50)

    # ===============
    # Next subplots
    # ===============

    indexFinalStateReshaped = 0
    modulus_number = np.round(len(allChannelsFinalStatesReshaped) / (rows * cols - 1))
    index_to_print = 0
    for finalStatesReshaped in allChannelsFinalStatesReshaped:
        if ((index_to_print == 0 or len(allChannelsFinalStatesReshaped) < modulus_number) or
                (index_to_print != 0 and indexFinalStateReshaped % modulus_number == 0 and
                 index_to_print < (rows * cols - 1))):
            # set up the axes for the second plot
            ax = fig.add_subplot(rows, cols, 2 + index_to_print, projection='3d')
            draw_cube(ax)
            # draw final states
            ax.plot_wireframe(finalStatesReshaped['reshapedCoordsX'],
                              finalStatesReshaped['reshapedCoordsY'],
                              finalStatesReshaped['reshapedCoordsZ'], color="r")
            title = "Output States\n Channel " + "$\eta=" + \
                str(int(math.degrees(anglesEta[indexFinalStateReshaped]))) + "\degree$"
            ax.set_title(title)
            # draw center
            ax.scatter([0], [0], finalStatesReshaped["center"], color="g", s=50)
            index_to_print += 1
        indexFinalStateReshaped += 1

    plt.show()


def plot_surface_blochs(initialStatesReshaped, allChannelsFinalStatesReshaped, anglesEta, rows=3, cols=3):
    fig = plt.figure(figsize=(20, 25))
    # ===============
    #  First subplot
    # ===============
    # set up the axes for the first plot
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    draw_cube(ax)

    # draw initial states
    surf = ax.plot_surface(initialStatesReshaped['reshapedCoordsX'],
                           initialStatesReshaped['reshapedCoordsY'],
                           initialStatesReshaped['reshapedCoordsZ'], linewidth=0, antialiased=True)
    ax.set_title("Input States")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # ===============
    # Next subplots
    # ===============

    indexFinalStateReshaped = 0
    modulus_number = np.round(len(allChannelsFinalStatesReshaped) / (rows * cols - 1))
    index_to_print = 0
    for finalStatesReshaped in allChannelsFinalStatesReshaped:
        if ((index_to_print == 0 or len(allChannelsFinalStatesReshaped) < modulus_number) or
                (index_to_print != 0 and indexFinalStateReshaped % modulus_number == 0 and
                 index_to_print < (rows * cols - 1))):
            # set up the axes for the second plot
            ax = fig.add_subplot(rows, cols, 2 + index_to_print, projection='3d')
            draw_cube(ax)
            # draw final states
            surf = ax.plot_surface(finalStatesReshaped['reshapedCoordsX'],
                                   finalStatesReshaped['reshapedCoordsY'],
                                   finalStatesReshaped['reshapedCoordsZ'], linewidth=0, antialiased=True)
            title = "Output States\n Channel " + "$\eta=" + \
                str(int(math.degrees(anglesEta[indexFinalStateReshaped]))) + "\degree$"
            ax.set_title(title)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            index_to_print += 1
        indexFinalStateReshaped += 1

    plt.show()


def run_base_circuit(angles_eta, points_theta, points_phase, iterations=1024, out_rz_angle=0, out_ry_angle=0):
    # Use Aer's qasm_simulator
    backend_sim = Aer.get_backend('qasm_simulator')
    state_backend_sim = Aer.get_backend('statevector_simulator')
    simulatedResult = runDampingChannelSimulation(
        anglesEta=angles_eta, pointsTheta=points_theta, pointsPhase=points_phase,
        iterations=iterations, backend=backend_sim, state_backend_sim=state_backend_sim,
        out_rz_angle=out_rz_angle, out_ry_angle=out_ry_angle)
    initialStates = simulatedResult["initialStates"]
    totalResults = simulatedResult["totalResults"]
    totalCounts = simulatedResult["totalCounts"]
    totalCircuits = simulatedResult["totalCircuits"]
    totalFinalStates = simulatedResult["totalFinalStates"]
    anglesPhase = simulatedResult["anglesPhase"]
    Z_Output0 = simulatedResult["Z_Output0"]
    Z_Output1 = simulatedResult["Z_Output1"]
    X_Input0 = simulatedResult["X_Input0"]
    X_Input1 = simulatedResult["X_Input1"]
    Y_Eta = simulatedResult["Y_Eta"]
    total_state_vectors = simulatedResult["total_state_vectors"]

    # Set the Initial States
    initialStatesReshaped = computeStateVectorCoordsReshaped(
        initialStates["zeroAmplitude"], anglesPhase, points_theta, points_phase)
    # Set the Final States
    allChannelsFinalStatesReshaped = []
    for indexFinalState in range(len(totalFinalStates)):
        singleChannelFinalStatesReshaped = computeFinaleStateVectorCoordsReshaped(
            totalFinalStates[indexFinalState]["zeroAmplitude"], anglesPhase, points_theta, points_phase)
        allChannelsFinalStatesReshaped.append(singleChannelFinalStatesReshaped)

    eta_degrees = list(
        map(lambda etaArray:
            list(map(lambda radian: int(math.degrees(radian)), etaArray)),
            Y_Eta)
    )

    return (initialStates, totalResults, totalCounts, totalCircuits, totalFinalStates,
            anglesPhase, Z_Output0, Z_Output1, X_Input0, X_Input1, Y_Eta, initialStatesReshaped,
            allChannelsFinalStatesReshaped, eta_degrees, total_state_vectors)


def compute_fidelity(total_state_vectors):
    return list(map(lambda state_vectors_one_channel:
                    list(map(lambda state_vector_execution:
                             state_fidelity(state_vector_execution['input_state_vector'],
                                            state_vector_execution['output_state_vector']),
                             state_vectors_one_channel)), total_state_vectors)
                )


def plot_fidelity(input_1_probability, total_state_vectors, angles_eta, rows=3, cols=3, row_size=3 * 5, col_size=3 * 10):
    fidelity_all_channels = compute_fidelity(total_state_vectors)
    fig = plt.figure(figsize=(row_size, col_size))
    index_channel = 0
    modulus_number = np.round(len(fidelity_all_channels) / (rows * cols))
    index_to_print = 0
    for fidelity_one_channel in fidelity_all_channels:
        if ((index_to_print == 0 or len(fidelity_all_channels) < modulus_number) or
                (index_to_print != 0 and index_channel % modulus_number == 0 and
                 index_to_print < (rows * cols))):
            # set up the axes for the second plot
            ax = fig.add_subplot(rows, cols, 1 + index_to_print)
            # draw final states
            ax.plot(input_1_probability[index_channel], fidelity_one_channel, 'o',
                    label="$\eta=" + str(int(math.degrees(angles_eta[index_channel]))) + "\degree$")
            ax.set_title(f"Fidelity Channel $\eta={str(int(math.degrees(angles_eta[index_channel])))} \degree$")
            ax.set_xlabel("Input State " + "$\\vert\\vert\\beta\\vert\\vert ^2  \\vert1\\rangle$")
            ax.set_ylabel('% Fidelity')
            index_to_print += 1
        index_channel += 1
    plt.show()

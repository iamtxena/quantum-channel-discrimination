from itertools import product, combinations
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info.states.utils import partial_trace
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
            initialStatesZeroAmplitude.append(math.cos(theta) * (1 + 0j))
            initialStatesOneAmplitude.append((math.sin(theta) * e**(1j * phase) + 0 + 0j))
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
        Theta_i = np.arccos(amplitudesVector[indexAmplitudes])
        Phase_i = inputAnglesPhase[indexAmplitudes % nPointsPhase]
        Theta_Bloch = 2 * Theta_i
        initialStateVectorCoordsX.append(np.sin(Theta_Bloch) * np.cos(Phase_i))
        initialStateVectorCoordsY.append(np.sin(Theta_Bloch) * np.sin(Phase_i))
        initialStateVectorCoordsZ.append(np.cos(Theta_Bloch))
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
        Theta_i = np.arccos(amplitudesVector[indexAmplitudes])
        Theta_Bloch = 2 * Theta_i
        finalStateVectorCoordsZ.append(np.cos(Theta_Bloch))
#        finalStateVectorCoordsZ.append(np.cos(Theta_i))
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
                                iterations, backend, out_rx_angle=0, out_ry_angle=0):
    # Create 2 qbits circuit and 1 output classical bit
    qreg_q = QuantumRegister(2, 'q')
    creg_c = ClassicalRegister(1, 'c')
    # First we generate the angles which will help the draw the sphere
    anglesTheta = np.mgrid[0:pi / 2:pointsTheta * 1j]
    anglesPhase = np.mgrid[0:2 * pi:pointsPhase * 1j]

    totalResults = []
    totalCounts = []
    totalCircuits = []
    totalFinalStates = []

    Z_Output0 = []
    Z_Output1 = []
    X_Input0 = []
    X_Input1 = []
    Y_Eta = []

    initialStates = prepareInitialStates(anglesTheta, anglesPhase)
    index = -1
    # Initialize circuit with desired initial_state
    for eta in anglesEta:
        index += 1
        if index % 10 == 0:
            print("Simulating channel with " + u"\u03BB" + " = " + str(format(math.sin(eta) * math.sin(eta), '.2f')))
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
            circuit.cry(2 * eta, qreg_q[0], qreg_q[1])
            circuit.cx(qreg_q[1], qreg_q[0])
            circuit.rx(out_rx_angle, qreg_q[0])
            circuit.ry(out_ry_angle, qreg_q[0])
            circuit.measure(qreg_q[0], creg_c[0])
            circuitSpecificChannel.append(circuit)
            # execute circuit on backends
            job_sim = execute(circuit, backend, shots=iterations)
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

            # append the results for a specific channel
        totalCircuits.append(circuitSpecificChannel)
        totalFinalStates.append(finalStates)
        totalCounts.append(countsSpecificChannel)
        totalResults.append(circuitResultsSpecificChannel)

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


def plot_surface_probabilities(X_Input0, X_Input1, Z_Output0, Z_Output1, lambdas_per_state):
    # Representation of output probabilities for all circuit in a 3d plot
    fig = plt.figure(figsize=(25, 35))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X_Input0, lambdas_per_state, Z_Output0, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    ax.set_title("Output Probabilities for $\\vert0\\rangle$", fontsize=30)
    plt.ylabel("Attenuation factor $\lambda$")
    plt.xlabel("Input State ||" + "$\\alpha||^2 |0\\rangle$")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(X_Input1, lambdas_per_state, Z_Output1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    ax.set_title("Output Probabilities for $\\vert1\\rangle$", fontsize=30)
    plt.ylabel("Attenuation factor $\lambda$")
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


def plot_wireframe_blochs(allChannelsFinalStatesReshaped, lambdas, rows=3, cols=3):
    fig = plt.figure(figsize=(20, 25))
    # ===============
    #  First subplot
    # ===============
    # set up the axes for the first plot
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    draw_cube(ax)

    # ===============
    # Next subplots
    # ===============

    indexFinalStateReshaped = 0
    modulus_number = np.round(len(allChannelsFinalStatesReshaped) / (rows * cols - 1))
    index_to_print = 0
    for idx, finalStatesReshaped in enumerate(allChannelsFinalStatesReshaped):
        if ((index_to_print == 0 or len(allChannelsFinalStatesReshaped) < modulus_number) or
                (index_to_print != 0 and indexFinalStateReshaped % modulus_number == 0 and
                 index_to_print < (rows * cols - 1)) or
                (idx == len(allChannelsFinalStatesReshaped) - 1)):
            # set up the axes for the second plot
            ax = fig.add_subplot(rows, cols, 1 + index_to_print, projection='3d')
            draw_cube(ax)
            # draw final states
            ax.plot_wireframe(finalStatesReshaped['reshapedCoordsX'],
                              finalStatesReshaped['reshapedCoordsY'],
                              finalStatesReshaped['reshapedCoordsZ'], color="r")
            title = f"Output States\n Channel $\lambda= {lambdas[indexFinalStateReshaped]}$"
            ax.set_title(title)
            # draw center
            ax.scatter([0], [0], finalStatesReshaped["center"], color="g", s=50)
            index_to_print += 1
        indexFinalStateReshaped += 1

    plt.show()


def plot_surface_blochs(initialStatesReshaped, allChannelsFinalStatesReshaped, lambdas, rows=3, cols=3):
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
    # draw center
    ax.scatter([0], [0], [0], color="g", s=50)
    # Draw state
    ax.scatter(initialStatesReshaped['reshapedCoordsX'][6][0], initialStatesReshaped['reshapedCoordsY']
               [6][6], initialStatesReshaped['reshapedCoordsZ'][6][9], color="b", s=150)

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

            title = f"Final States\n Channel $\lambda= {lambdas[indexFinalStateReshaped]}$"

            ax.set_title(title)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.scatter([0], [0], finalStatesReshaped["center"], color="g", s=50)
            ax.scatter(finalStatesReshaped['reshapedCoordsX'][6][0], finalStatesReshaped['reshapedCoordsY'][6][6],
                       finalStatesReshaped['reshapedCoordsZ'][6][9], color="b", s=(150 - 25 * indexFinalStateReshaped))
            index_to_print += 1
        indexFinalStateReshaped += 1

    plt.show()


def run_base_circuit(angles_eta, points_theta, points_phase, iterations=1024,
                     out_rx_angle=0, out_ry_angle=0, backend=Aer.get_backend('qasm_simulator')):

    simulatedResult = runDampingChannelSimulation(
        anglesEta=angles_eta, pointsTheta=points_theta, pointsPhase=points_phase,
        iterations=iterations, backend=backend,
        out_rx_angle=out_rx_angle, out_ry_angle=out_ry_angle)
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
    lambdas_per_state = list(
        map(lambda etaArray:
            list(map(lambda eta: np.round(math.sin(eta)**2, 3), etaArray)),
            Y_Eta)
    )

    return (initialStates, totalResults, totalCounts, totalCircuits, totalFinalStates,
            anglesPhase, Z_Output0, Z_Output1, X_Input0, X_Input1, Y_Eta, initialStatesReshaped,
            allChannelsFinalStatesReshaped, eta_degrees, lambdas_per_state)


def prepareInitialStatesFixedPhase(pointsTheta, Phase=0):
    """ Prepare initial states to pass through the circuit """
    """ with a fixed Phase angle. Only moving across Theta """

    # As we have to provide the state values to initialize the qreg[0] we have to do a conversion
    # from angles in the sphere to statevector amplitudes. These statevectors will be the combination of
    # Zero_Amplitude*|0> plus One_Amplitude*|1>
    initialStates = []

    for Theta_Bloch in np.mgrid[0:pi:pointsTheta * 1j]:
        a = math.cos(Theta_Bloch / 2)
        b = math.sin(Theta_Bloch / 2) * e**(1j * Phase)
        norm = np.sqrt(a * np.conj(a) + b * np.conj(b))
        if norm > 1:
            initialStates.append([a / norm, b / norm])
        else:
            initialStates.append([a, b])

    return np.array(initialStates)


def calculate_fidelity(initialStates, eta, rx_angle=0, ry_angle=0):
    qreg_q = QuantumRegister(2, 'q')
    creg_c = ClassicalRegister(1, 'c')
    circ = QuantumCircuit(qreg_q, creg_c)
    backend_sim = Aer.get_backend('statevector_simulator')

    stavec = []
    fidelity = []
    for i in range(len(initialStates)):
        circ.initialize([initialStates[i][0], initialStates[i][1]], qreg_q[0])
        circ.reset(qreg_q[1])
        circ.cry(2 * eta, qreg_q[0], qreg_q[1])
        circ.cx(qreg_q[1], qreg_q[0])
        circ.rx(rx_angle, qreg_q[0])
        circ.ry(ry_angle, qreg_q[0])
        res = execute(circ, backend_sim).result()
        sv = res.get_statevector(circ)
        stavec.append([sv[0], sv[1]])
        fidelity.append([state_fidelity([initialStates[i][0], initialStates[i][1]],
                                        partial_trace(np.outer(sv, sv), [1]), validate=True),
                         state_fidelity([1, 0], partial_trace(np.outer(sv, sv), [1]), validate=True)])
    return np.array(fidelity)


def plot_fidelity(anglesEta, pointsTheta, rx_angle=0, ry_angle=0):
    # Representation of fidelity
    initialStates = prepareInitialStatesFixedPhase(pointsTheta, 0)

    fig = plt.figure(figsize=(25, 10))
    fig.suptitle('Fidelity Analysis', fontsize=20)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Output vs. Input. ' + ' $Rx = ' + str(int(math.degrees(rx_angle))) + '\degree$' +
                  ' $Ry = ' + str(int(math.degrees(ry_angle))) + '\degree$', fontsize=14)
    ax1.set_xlabel('Input State ||' + '$\\alpha||^2 \\vert0\\rangle$', fontsize=14)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Output versus $\\vert0\\rangle$ state ' + '$Rx = ' + str(int(math.degrees(rx_angle))) +
                  '\degree$' + ' $Ry = ' + str(int(math.degrees(ry_angle))) + '\degree$', fontsize=14)
    ax2.set_xlabel('Input State ||' + '$\\alpha||^2 \\vert0\\rangle$', fontsize=14)

    index_channel = 0
    modulus_number = np.round(len(anglesEta) / 10)
    index_to_print = 0

    for idx, eta in enumerate(anglesEta):
        if ((index_to_print == 0 or len(anglesEta) <= modulus_number) or
                (index_to_print != 0 and index_channel % modulus_number == 0 and index_to_print < 10) or
                (idx == len(anglesEta) - 1)):
            X = []
            Y = []
            Z = []

            fidelity = calculate_fidelity(initialStates, eta, rx_angle, ry_angle)
            for i in range(len(initialStates)):
                X.append((initialStates[i][0] * np.conj(initialStates[i][0])).real)
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

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
import math
import random
import numpy as np
from numpy import pi
import time
import itertools


def play_and_guess(list_theta, list_phase, list_eta, list_phi_rx, list_phi_ry, backend, plays=10, simulation_iterations=1024):
    """ Play the play_and_guess game for the number of plays for each combination of input parameters.
        Returns an array of execution results. 

        Each item of the result list consist of:
        -----------------------------------------------------------------
        | theta | phase | phi_rx | phi_ry | eta0 | eta1 | succ_avg_prob |
        -----------------------------------------------------------------
        succ_avg_prob: is the average probability of guessing correctly for this specific combination 
        We want to maximize these value.
    """
    results = []

    # when there is only one element, we add the same element
    if len(list_eta) == 1:
        list_eta.append(list_eta[0])
    # get combinations of two etas without repeats
    two_pair_etas = itertools.combinations(list_eta, 2)

    program_start_time = time.time()
    print("Starting the execution")
    for theta in list_theta:
        print(f"execution with theta {theta}")
        start_time = time.time()
        for phase in list_phase:
            print(f"execution with phase {phase}")
            for phi_rx in list_phi_rx:
                for phi_ry in list_phi_ry:
                    for eta_pair in two_pair_etas:
                        i = 0
                        configuration = {
                            "theta": theta,
                            "phase": phase,
                            "phi_rx": phi_rx,
                            "phi_ry": phi_ry,
                            "eta0": eta_pair[0],
                            "eta1": eta_pair[1]
                        }
                        success_counts = 0
                        for play in range(plays):
                            success_counts += play_and_guess_one_case(configuration, backend, simulation_iterations)
                        succ_avg_prob = success_counts / plays
                        print(f'appending result number {i}')
                        results.append({
                            "theta": theta,
                            "phase": phase,
                            "phi_rx": phi_rx,
                            "phi_ry": phi_ry,
                            "eta0": eta_pair[0],
                            "eta1": eta_pair[1],
                            "succ_avg_prob": succ_avg_prob
                        })
                        i += 1
        end_time = time.time()
        print("total minutes taken this theta: ", int(np.round((end_time - start_time) / 60)))
        print("total minutes taken so far: ", int(np.round((end_time - program_start_time) / 60)))
    end_time = time.time()
    print("total minutes of execution time: ", int(np.round((end_time - program_start_time) / 60)))
    print("All guesses have been calculated")
    return results

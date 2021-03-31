""" Auxiliary static methods """
import numpy as np
import matplotlib.pyplot as plt
from typing import List, cast


def build_probabilities_matrix(result):
    X1 = []
    for eta_pair in result['eta_pairs']:
        X1.append(int(eta_pair[1]))
        X1.append(int(eta_pair[0]))
    X1 = sorted(list(dict.fromkeys(X1)))
    lenx1 = len(X1)
    probs1 = np.zeros((lenx1, lenx1))
    values1 = list(result.values())
    for ind_prob in range(len(values1[2])):
        ind_0 = X1.index(int(result['eta_pairs'][ind_prob][0]))
        ind_1 = X1.index(int(result['eta_pairs'][ind_prob][1]))
        probs1[ind_1, ind_0] = values1[2][ind_prob]
    for i in range(len(X1)):
        probs1[i, i] = 0.5
    return probs1


def build_amplitudes_matrix(result):
    X1 = []
    for eta_pair in result['eta_pairs']:
        X1.append(int(eta_pair[1]))
        X1.append(int(eta_pair[0]))
    X1 = sorted(list(set(X1)))
    lenx1 = len(X1)
    amp1 = np.zeros((lenx1, lenx1))
    values1 = list(result.values())
    for ind_prob in range(len(values1[3])):
        ind_0 = X1.index(int(result['eta_pairs'][ind_prob][0]))
        ind_1 = X1.index(int(result['eta_pairs'][ind_prob][1]))
        amp1[ind_1, ind_0] = np.sin(values1[3][ind_prob][0])
    for i in range(len(X1)):
        amp1[i, i] = 0
    return amp1


def plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax):
    fig = plt.figure(title)
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(delta_probs, cmap='RdBu', extent=(0, 90, 90, 0), vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=bar_label)
    ax1.set_xlabel('Channel 0 (angle $\eta$)')
    ax1.set_ylabel('Channel 1 (angle $\eta$)')
    ax1.set_title(title)
    plt.show()


def compute_percentage_delta_values(delta_values: List[List[float]],
                                    theoretical_results: List[List[float]]) -> List[List[float]]:
    in_delta_values = cast(np.ndarray, delta_values)
    in_theoretical_results = cast(np.ndarray, theoretical_results)
    total_col, total_row = in_delta_values.shape
    col = total_col
    delta_pc_prob1 = np.zeros((total_col, total_row))
    while col > 0:
        row = cast(int, total_row)
        while row > 0:
            if in_theoretical_results[row - 1, col - 1] == 0:
                if in_delta_values[row - 1, col - 1] == 0:
                    delta_pc_prob1[row - 1, col - 1] = 0
                else:
                    delta_pc_prob1[row - 1, col - 1] = 10000
            else:
                delta_pc_prob1[row - 1, col - 1] = 100 * \
                    in_delta_values[row -
                                    1, col - 1] / in_theoretical_results[row - 1, col - 1]
            row = row - 1
        col = col - 1
    return delta_pc_prob1

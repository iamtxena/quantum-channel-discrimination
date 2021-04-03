""" Auxiliary static methods """
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, cast, Tuple, Union, Dict, Optional
from ..configurations import OneShotConfiguration
from ..typings.configurations import (OptimalConfigurations,
                                      TheoreticalOneShotOptimalConfigurations)


def build_probabilities_matrix(
        result: Union[OptimalConfigurations, TheoreticalOneShotOptimalConfigurations]) -> List[List[float]]:
    if isinstance(result['eta_pairs'][0][0], float):
        return _build_probabilities_matrix(cast(OptimalConfigurations, result))
    if isinstance(result['eta_pairs'][0][0], str):
        return cast(List[List[float]], _build_probabilities_matrix_legacy(cast(Dict, result)))
    raise ValueError('Bad input results')


def build_amplitudes_matrix(
        result: Union[OptimalConfigurations, TheoreticalOneShotOptimalConfigurations]) -> List[List[float]]:
    if isinstance(result['eta_pairs'][0][0], float) and _isTheoreticalOneShotOptimalConfigurations(result):
        return _build_theoretical_amplitudes_matrix(cast(TheoreticalOneShotOptimalConfigurations, result))
    if isinstance(result['eta_pairs'][0][0], float) and _isOptimalConfigurations(result):
        return _build_amplitudes_matrix(cast(OptimalConfigurations, result))
    if isinstance(result['eta_pairs'][0][0], str):
        return cast(List[List[float]], _build_amplitudes_matrix_legacy(cast(Dict, result)))
    raise ValueError('Bad input results')


def _build_probabilities_matrix(result: OptimalConfigurations) -> List[List[float]]:
    sorted_etas, matrix = _init_matrix(result)
    _assign_probabilities(result, sorted_etas, matrix)
    _reset_diagonal_matrix(sorted_etas, matrix, value=0.5)
    return matrix


def _build_probabilities_matrix_legacy(result: Dict) -> List[List[int]]:
    sorted_etas, matrix = _init_matrix_legacy(result)
    _assign_probabilities_legacy(result, sorted_etas, matrix)
    _reset_diagonal_matrix(sorted_etas, matrix, value=0.5)
    return matrix


def _build_amplitudes_matrix(result: OptimalConfigurations) -> List[List[float]]:
    sorted_etas, matrix = _init_matrix(result)
    _assign_amplitudes(result, sorted_etas, matrix)
    _reset_diagonal_matrix(sorted_etas, matrix, value=0)
    return matrix


def _build_theoretical_amplitudes_matrix(result: TheoreticalOneShotOptimalConfigurations) -> List[List[float]]:
    sorted_etas, matrix = _init_matrix(result)
    _assign_theoretical_amplitudes(result, sorted_etas, matrix)
    _reset_diagonal_matrix(sorted_etas, matrix, value=0)
    return matrix


def _build_amplitudes_matrix_legacy(result: Dict) -> List[List[int]]:
    sorted_etas, matrix = _init_matrix_legacy(result)
    _assign_amplitudes_legacy(result, sorted_etas, matrix)
    _reset_diagonal_matrix(sorted_etas, matrix, value=0)
    return matrix


def _assign_probabilities(result: OptimalConfigurations, sorted_etas: List[float], matrix: np.array):
    for idx, probability in enumerate(result['probabilities']):
        ind_0 = sorted_etas.index(result['eta_pairs'][idx][0])
        ind_1 = sorted_etas.index(result['eta_pairs'][idx][1])
        matrix[ind_0, ind_1] = probability


def _assign_probabilities_legacy(result: Dict, sorted_etas: List[int], matrix: np.array):
    for idx, probability in enumerate(result['probabilities']):
        ind_0 = sorted_etas.index(int(result['eta_pairs'][idx][0]))
        ind_1 = sorted_etas.index(int(result['eta_pairs'][idx][1]))
        matrix[ind_1, ind_0] = probability


def _assign_amplitudes(result: OptimalConfigurations, sorted_etas: List[float], matrix: np.array):
    for idx, configuration in enumerate(result['configurations']):
        ind_0 = sorted_etas.index(result['eta_pairs'][idx][0])
        ind_1 = sorted_etas.index(result['eta_pairs'][idx][1])
        matrix[ind_0, ind_1] = np.sin(cast(OneShotConfiguration, configuration).theta)


def _assign_theoretical_amplitudes(result: TheoreticalOneShotOptimalConfigurations,
                                   sorted_etas: List[float],
                                   matrix: np.array):
    for idx, best_theoretical_amplitude in enumerate(result['list_theoretical_amplitude']):
        ind_0 = sorted_etas.index(result['eta_pairs'][idx][0])
        ind_1 = sorted_etas.index(result['eta_pairs'][idx][1])
        matrix[ind_0, ind_1] = best_theoretical_amplitude


def _assign_amplitudes_legacy(result: Dict, sorted_etas: List[int], matrix: np.array):
    for idx, configuration in enumerate(result['configurations']):
        ind_0 = sorted_etas.index(int(result['eta_pairs'][idx][0]))
        ind_1 = sorted_etas.index(int(result['eta_pairs'][idx][1]))
        matrix[ind_1, ind_0] = np.sin(configuration[0])


def _reset_diagonal_matrix(values: Union[List[float], List[int]], matrix: np.array, value: float = 0) -> None:
    for idx, _ in enumerate(values):
        matrix[idx, idx] = value


def _get_sorted_etas_in_degrees(eta_pairs: List[Tuple[float, float]]) -> List[float]:
    X1 = []
    for eta_pair in eta_pairs:
        X1.append(eta_pair[1])
        X1.append(eta_pair[0])
    return sorted(list(set(X1)))


def _get_sorted_etas_in_degrees_legacy(eta_pairs: List[Tuple[float, float]]) -> List[int]:
    X1 = []
    for eta_pair in eta_pairs:
        X1.append(int(eta_pair[1]))
        X1.append(int(eta_pair[0]))
    return sorted(list(set(X1)))


def _init_matrix(result) -> Tuple[List[float], np.array]:
    sorted_etas = _get_sorted_etas_in_degrees(result['eta_pairs'])
    lenx1 = len(sorted_etas)
    amp1 = np.zeros((lenx1, lenx1))
    return sorted_etas, amp1


def _init_matrix_legacy(result) -> Tuple[List[int], np.array]:
    sorted_etas = _get_sorted_etas_in_degrees_legacy(result['eta_pairs'])
    lenx1 = len(sorted_etas)
    amp1 = np.zeros((lenx1, lenx1))
    return sorted_etas, amp1


def plot_one_result(result, title, bar_label, vmin, vmax, cmap='viridis'):
    fig = plt.figure(title)
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(result,
                    cmap, extent=(0, 90, 90, 0), vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=bar_label)
    ax1.set_xlabel('Channel 0 (angle $\eta$)')
    ax1.set_ylabel('Channel 1 (angle $\eta$)')
    ax1.set_title(title, fontsize=14)
    plt.show()


def plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax):
    fig = plt.figure(title)
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(delta_probs, cmap='RdBu', extent=(0, 90, 90, 0), vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=bar_label)
    ax1.set_xlabel('Channel 0 (angle $\eta$)')
    ax1.set_ylabel('Channel 1 (angle $\eta$)')
    ax1.set_title(title, fontsize=14)
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


def _isOptimalConfigurations(input_dict: Union[OptimalConfigurations,
                                               TheoreticalOneShotOptimalConfigurations]) -> bool:
    """ check if input dictionary is an OptimalConfigurations one """
    tmp_dict = cast(OptimalConfigurations, input_dict)
    if (tmp_dict.get('eta_pairs') and
        tmp_dict.get('best_algorithm') and
        tmp_dict.get('probabilities') and
        tmp_dict.get('configurations') and
            tmp_dict.get('number_calls_made')):
        return True
    return False


def _isTheoreticalOneShotOptimalConfigurations(input_dict: Union[OptimalConfigurations,
                                                                 TheoreticalOneShotOptimalConfigurations]) -> bool:
    """ check if input dictionary is an TheoreticalOneShotOptimalConfigurations one """
    tmp_dict = cast(TheoreticalOneShotOptimalConfigurations, input_dict)
    if (tmp_dict.get('eta_pairs') and
        tmp_dict.get('probabilities') and
            tmp_dict.get('list_theoretical_amplitude')):
        return True
    return False


def load_result_from_file(name: str, path: Optional[str] = "") -> OptimalConfigurations:
    """ load result from a file """
    with open(f'./{path}{name}.pkl', 'rb') as file:
        return pickle.load(file)


def save_result_to_disk(optimal_configurations: OptimalConfigurations, name: str, path: Optional[str] = "") -> None:
    """ save result to a file """
    with open(f'./{path}{name}.pkl', 'wb') as file:
        pickle.dump(optimal_configurations, file, pickle.HIGHEST_PROTOCOL)

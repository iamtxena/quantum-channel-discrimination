""" Auxiliary static methods """
import pickle
from qcd.typings import TheoreticalOptimizationSetup
from qcd.optimizations.aux import get_combinations_n_etas_without_repeats
import numpy as np
import matplotlib.pyplot as plt
from typing import List, cast, Tuple, Union, Dict, Optional
from ..configurations import OneShotConfiguration
from ..typings.configurations import (OptimalConfigurations, TheoreticalOneShotEntangledOptimalConfigurations,
                                      TheoreticalOneShotOptimalConfigurations)


def build_probabilities_matrix(
        result: Union[OptimalConfigurations, TheoreticalOneShotOptimalConfigurations]) -> List[List[float]]:
    if (isinstance(result['eta_groups'][0][0], float) and
            (_isTheoreticalOneShotOptimalConfigurations(result) or _isOptimalConfigurations(result))):
        return _build_probabilities_matrix(cast(OptimalConfigurations, result))
    if isinstance(result['eta_groups'][0][0], str):
        return cast(List[List[float]], _build_probabilities_matrix_legacy(cast(Dict, result)))
    if isinstance(result['eta_groups'][0][0], float):
        return cast(List[List[float]], _build_probabilities_matrix_new_legacy(cast(Dict, result)))
    raise ValueError('Bad input results')


def build_amplitudes_matrix(
        result: Union[OptimalConfigurations, TheoreticalOneShotOptimalConfigurations]) -> List[List[float]]:
    if isinstance(result['eta_groups'][0][0], float) and _isTheoreticalOneShotOptimalConfigurations(result):
        return _build_theoretical_amplitudes_matrix(cast(TheoreticalOneShotOptimalConfigurations, result))
    if isinstance(result['eta_groups'][0][0], float) and _isOptimalConfigurations(result):
        return _build_amplitudes_matrix(cast(OptimalConfigurations, result))
    if isinstance(result['eta_groups'][0][0], float):
        return cast(List[List[float]], _build_amplitudes_matrix_new_legacy(cast(Dict, result)))
    if isinstance(result['eta_groups'][0][0], str):
        return cast(List[List[float]], _build_amplitudes_matrix_legacy(cast(Dict, result)))
    raise ValueError('Bad input results')


def _build_probabilities_matrix(result: OptimalConfigurations) -> List[List[float]]:
    sorted_etas, matrix = _init_matrix(result)
    # if 'legacy' in result and result['legacy'] is True:
    return _assign_probabilities_new_legacy(result, sorted_etas, matrix)  # type: ignore
    # return _assign_probabilities(result, sorted_etas, matrix)


def _build_probabilities_matrix_new_legacy(result: Dict) -> List[List[int]]:
    sorted_etas, matrix = _init_matrix(result)
    return _assign_probabilities_new_legacy(result, sorted_etas, matrix)


def _build_probabilities_matrix_legacy(result: Dict) -> List[List[int]]:
    sorted_etas, matrix = _init_matrix_legacy(result)
    return _assign_probabilities_legacy(result, sorted_etas, matrix)


def _build_amplitudes_matrix(result: OptimalConfigurations) -> List[List[float]]:
    sorted_etas, matrix = _init_matrix(result)
    # if 'legacy' in result and result['legacy'] is True:
    return _assign_amplitudes_different_order(result, sorted_etas, matrix)
    if hasattr(cast(OneShotConfiguration, result['configurations'][0]), 'state_probability'):
        return _assign_amplitudes(result, sorted_etas, matrix)
    if hasattr(cast(OneShotConfiguration, result['configurations'][0]), 'theta'):
        # support for legacy results
        return _assign_amplitudes_with_thetas(result, sorted_etas, matrix)
    if 'state_probability' in cast(Dict, result)['configurations'][0]:
        return _assign_amplitudes_dict(result, sorted_etas, matrix)
    raise ValueError('Optimal Configurations require either state_probability or theta properties')


def _build_theoretical_amplitudes_matrix(result: TheoreticalOneShotOptimalConfigurations) -> List[List[float]]:
    sorted_etas, matrix = _init_matrix(result)
    return _assign_theoretical_amplitudes(result, sorted_etas, matrix)


def _build_amplitudes_matrix_new_legacy(result: Dict) -> List[List[int]]:
    sorted_etas, matrix = _init_matrix(result)
    return _assign_amplitudes_new_legacy(result, sorted_etas, matrix)


def _build_amplitudes_matrix_legacy(result: Dict) -> List[List[int]]:
    sorted_etas, matrix = _init_matrix_legacy(result)
    return _assign_amplitudes_legacy(result, sorted_etas, matrix)


def _assign_probabilities(result: OptimalConfigurations, sorted_etas: List[float], matrix: np.array):
    for idx, probability in enumerate(result['probabilities']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = probability
    return matrix


def _assign_probabilities_new_legacy(result: Dict, sorted_etas: List[float], matrix: np.array):
    for idx, probability in enumerate(result['probabilities']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group_different_order(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = probability
    return matrix


def _assign_probabilities_legacy(result: Dict, sorted_etas: List[int], matrix: np.array):
    for idx, probability in enumerate(result['probabilities']):
        ind_0 = (len(sorted_etas) - 1) - sorted_etas.index(int(result['eta_groups'][idx][0]))
        ind_1 = sorted_etas.index(int(result['eta_groups'][idx][1]))
        matrix[ind_1, ind_0] = probability
    return matrix


def _assign_amplitudes(result: OptimalConfigurations,
                       sorted_etas: List[float],
                       matrix: np.array) -> List[List[float]]:
    for idx, configuration in enumerate(result['configurations']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = cast(OneShotConfiguration, configuration).state_probability
    return matrix


def _assign_amplitudes_dict(result: OptimalConfigurations,
                            sorted_etas: List[float],
                            matrix: np.array) -> List[List[float]]:
    for idx, configuration in enumerate(result['configurations']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = cast(Dict, configuration)['state_probability']
    return matrix


def _assign_amplitudes_different_order(result: OptimalConfigurations,
                                       sorted_etas: List[float],
                                       matrix: np.array) -> List[List[float]]:
    for idx, configuration in enumerate(result['configurations']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group_different_order(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = cast(OneShotConfiguration, configuration).state_probability
    return matrix


def _assign_amplitudes_with_thetas(result: OptimalConfigurations,
                                   sorted_etas: List[float],
                                   matrix: np.array) -> List[List[float]]:
    for idx, configuration in enumerate(result['configurations']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = np.sin(cast(OneShotConfiguration, configuration).theta)
    return matrix


def _assign_theoretical_amplitudes(result: TheoreticalOneShotOptimalConfigurations,
                                   sorted_etas: List[float],
                                   matrix: np.array):
    for idx, best_theoretical_amplitude in enumerate(result['list_theoretical_amplitude']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = best_theoretical_amplitude
    return matrix


def _assign_amplitudes_legacy(result: Dict, sorted_etas: List[int], matrix: np.array):
    for idx, configuration in enumerate(result['configurations']):
        ind_0 = (len(sorted_etas) - 1) - sorted_etas.index(int(result['eta_groups'][idx][0]))
        ind_1 = sorted_etas.index(int(result['eta_groups'][idx][1]))
        matrix[ind_1, ind_0] = np.sin(configuration[0])
    return matrix


def _assign_amplitudes_new_legacy(result: Dict, sorted_etas: List[float], matrix: np.array):
    for idx, configuration in enumerate(result['configurations']):
        ind_0, ind_1 = _get_matrix_index_from_eta_group_different_order(result, sorted_etas, idx)
        matrix[ind_1, ind_0] = configuration[0]
    return matrix


def _get_matrix_index_from_eta_group(result: OptimalConfigurations,
                                     sorted_etas: List[float], idx: int) -> Tuple[int, int]:
    ind_0 = sorted_etas.index(result['eta_groups'][idx][0])
    ind_1 = (len(sorted_etas) - 1) - sorted_etas.index(result['eta_groups'][idx][1])
    return ind_0, ind_1


def _get_matrix_index_from_eta_group_different_order(result, sorted_etas, idx):
    ind_0 = (len(sorted_etas) - 1) - sorted_etas.index(result['eta_groups'][idx][0])
    ind_1 = sorted_etas.index(result['eta_groups'][idx][1])
    return ind_0, ind_1


def _reset_diagonal_matrix(values: Union[List[float], List[int]], matrix: np.array, value: float = 0) -> None:
    for idx, _ in enumerate(values):
        matrix[idx, idx] = value


def _get_sorted_etas_in_degrees(eta_groups: List[Tuple[float, float]]) -> List[float]:
    X1 = []
    for eta_group in eta_groups:
        X1.append(eta_group[0])
        X1.append(eta_group[1])
    return sorted(list(set(X1)))


def _get_sorted_etas_in_degrees_legacy(eta_groups: List[Tuple[float, float]]) -> List[int]:
    X1 = []
    for eta_group in eta_groups:
        X1.append(int(eta_group[0]))
        X1.append(int(eta_group[1]))
    return sorted(list(set(X1)))


def _init_matrix(result) -> Tuple[List[float], np.array]:
    sorted_etas = _get_sorted_etas_in_degrees(result['eta_groups'])
    lenx1 = len(sorted_etas)
    amp1 = np.zeros((lenx1, lenx1))
    return sorted_etas, amp1


def _init_matrix_legacy(result) -> Tuple[List[int], np.array]:
    sorted_etas = _get_sorted_etas_in_degrees_legacy(result['eta_groups'])
    lenx1 = len(sorted_etas)
    amp1 = np.zeros((lenx1, lenx1))
    return sorted_etas, amp1


def plot_one_result(result, title, bar_label, vmin, vmax, cmap='viridis'):
    fig = plt.figure(title)
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(result,
                    cmap, extent=(0, 90, 0, 90), vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=bar_label)
    ax1.set_xlabel('Channel 0 (angle $\eta$)')
    ax1.set_ylabel('Channel 1 (angle $\eta$)')
    ax1.set_title(title, fontsize=14)
    plt.show()


def plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax, cmap='RdBu'):
    fig = plt.figure(title)
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(delta_probs, cmap=cmap, extent=(0, 90, 0, 90), vmin=vmin, vmax=vmax)
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
    if (tmp_dict.get('eta_groups') and
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
    if (tmp_dict.get('eta_groups') and
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


def get_theoretical_optimization_setup_from_number_of_etas(
        number_channels_to_discriminate: int = 2,
        eta_partitions: int = 20,
        number_third_channels: int = 5) -> TheoreticalOptimizationSetup:
    """ compute the eta groups given the number of etas to generate from 0 to pi/2 (including pi/2) """
    eta_groups = get_combinations_n_etas_without_repeats(number_channels_to_discriminate,
                                                         eta_partitions,
                                                         number_third_channels)
    return {'eta_groups': eta_groups,
            'number_channels_to_discriminate': number_channels_to_discriminate}


def build_improvement_matrix(
        optimal_configurations: TheoreticalOneShotEntangledOptimalConfigurations) -> List[List[float]]:
    """ Returns the correspondent matrix of the improvement values """
    sorted_etas, matrix = _init_matrix(optimal_configurations)
    _assign_improvement(optimal_configurations, sorted_etas, matrix)
    return matrix


def _assign_improvement(result: TheoreticalOneShotEntangledOptimalConfigurations,
                        sorted_etas: List[float],
                        matrix: np.array):
    for idx, improvement in enumerate(result['improvements']):
        ind_0 = sorted_etas.index(result['eta_groups'][idx][0])
        ind_1 = (len(sorted_etas) - 1) - sorted_etas.index(result['eta_groups'][idx][1])
        matrix[ind_1, ind_0] = improvement

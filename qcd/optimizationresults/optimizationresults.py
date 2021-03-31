from abc import ABC
from typing import Optional, List, Union, cast
from ..typings.configurations import OptimalConfigurations
import pickle
import numpy as np
import matplotlib.pyplot as plt


class OptimizationResults(ABC):
    """ Generic class acting as an interface for any Optimization Results """

    def __init__(self, optimal_configurations: Optional[OptimalConfigurations] = None) -> None:
        self._results: List[OptimalConfigurations] = []
        if optimal_configurations is not None:
            self._results = [optimal_configurations]
        self._probabilities_matrices: List[List[List[float]]] = []
        self._amplitudes_matrices: List[List[List[float]]] = []
        self._theoretical_probabilities_matrix: List[List[float]] = []
        self._theoretical_amplitudes_matrix: List[List[float]] = []

    """ save and load results to and from a file """

    def save_results_to_disk(self, name: str) -> None:
        for idx, result in enumerate(self._results):
            with open(f'results/{name}_{idx}.pkl', 'wb') as file:
                pickle.dump(result, file, pickle.HIGHEST_PROTOCOL)

    def load_results_from_file(self, name: str) -> None:
        with open('results/' + name + '.pkl', 'rb') as file:
            self._results.append(pickle.load(file))

    def load_theoretical_results_from_file(self, name: str) -> None:
        theoretical_results = []
        with open('results/' + name + '.pkl', 'rb') as file:
            theoretical_results.append(pickle.load(file))
        self._theoretical_probabilities_matrix = self._build_probabilities_matrix(theoretical_results)
        self._theoretical_amplitudes_matrix = self._build_amplitudes_matrix(theoretical_results)

    def load_results(self, file_names: Union[str, List[str]]) -> None:
        """
          1. Load results from file for all file names
          2. build probability matrices for each loaded result
          3. build amplitude matrix for each loaded result
        """
        names = file_names
        if not isinstance(names, List):
            names = [cast(str, file_names)]

        for file_name in names:
            self.load_results_from_file(file_name)

        self.build_probabilities_matrix()
        self.build_amplitudes_matrix()

    def build_probabilities_matrix(self) -> None:
        """ Build probabilities matrix for all loaded results """
        for result in self._results:
            probs1 = self._build_probabilities_matrix(result)
            self._probabilities_matrices.append(probs1)

    def _build_probabilities_matrix(self, result):
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

    def build_amplitudes_matrix(self) -> None:
        """ Build amplitudes matrix for all loaded results """
        for result in self._results:
            amp1 = self._build_amplitudes_matrix(result)
            self._amplitudes_matrices.append(amp1)

    def _build_amplitudes_matrix(self, result):
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

    def plot_probabilities(self,
                           results_index: int,
                           title: str = 'Probabilities from simulation',
                           bar_label: str = 'Probabilities value',
                           vmin: float = 0.0,
                           vmax: float = 1.0) -> None:
        """ Plot probabilities analysis """
        self._plot_one_result(results_index, title, bar_label, vmin, vmax)

    def _plot_one_result(self, results_index, title, bar_label, vmin, vmax):
        fig = plt.figure(title)
        ax1 = fig.add_subplot(111)
        im = ax1.imshow(self._probabilities_matrices[results_index],
                        cmap='viridis', extent=(0, 90, 90, 0), vmin=vmin, vmax=vmax)
        plt.colorbar(im, label=bar_label)
        ax1.set_xlabel('Channel 0 (angle $\eta$)')
        ax1.set_ylabel('Channel 1 (angle $\eta$)')
        ax1.set_title(title)
        plt.show()

    def plot_probabilities_comparison(self,
                                      results_index1: int,
                                      results_index2: int,
                                      title: str = 'Difference in Probabilities from simulation',
                                      bar_label: str = 'Probabilities value',
                                      vmin: float = -0.1,
                                      vmax: float = 0.1) -> None:
        """ Plot probabilities comparing two results """
        delta_probs = cast(np.ndarray, self._probabilities_matrices[results_index1]) - \
            cast(np.ndarray, self._probabilities_matrices[results_index2])
        self._plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_probabilities_comparison_with_theorical_results(self,
                                                             results_index: int,
                                                             title: str = 'Difference in Probabilities' +
                                                                          '(theory vs. simulation)',
                                                             bar_label: str = 'Probabilities Delta value',
                                                             vmin: float = -0.1,
                                                             vmax: float = 0.1) -> None:
        """ Plot probabilities comparing two results """
        delta_probs = cast(np.ndarray, self._theoretical_probabilities_matrix) - \
            cast(np.ndarray, self._probabilities_matrices[results_index])
        self._plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def _plot_comparison_between_two_results(self, delta_probs, title, bar_label, vmin, vmax):
        fig = plt.figure(title)
        ax1 = fig.add_subplot(111)
        im = ax1.imshow(delta_probs, cmap='RdBu', extent=(0, 90, 90, 0), vmin=vmin, vmax=vmax)
        plt.colorbar(im, label=bar_label)
        ax1.set_xlabel('Channel 0 (angle $\eta$)')
        ax1.set_ylabel('Channel 1 (angle $\eta$)')
        ax1.set_title(title)
        plt.show()

    def plot_amplitudes(self,
                        results_index: int,
                        title: str = 'Input state amplitude |1> obtained from simulation',
                        bar_label: str = 'Amplitude value',
                        vmin: float = 0.0,
                        vmax: float = 1.0) -> None:
        """ Plot amplitudes analysis """
        self._plot_one_result(results_index, title, bar_label, vmin, vmax)

    def plot_amplitudes_comparison(self,
                                   results_index1: int,
                                   results_index2: int,
                                   title: str = 'Difference in Amplitudes (between simulations)',
                                   bar_label: str = 'Amplitude value',
                                   vmin: float = -1.0,
                                   vmax: float = 1.0) -> None:
        """ Plot amplitudes comparing two results """
        delta_probs = cast(np.ndarray, self._amplitudes_matrices[results_index1]) - \
            cast(np.ndarray, self._amplitudes_matrices[results_index2])
        self._plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_amplitudes_comparison_with_theorical_results(self,
                                                          results_index: int,
                                                          title: str = 'Difference in Amplitudes' +
                                                          '(theory vs. simulation)',
                                                          bar_label: str = 'Amplitude Delta value',
                                                          vmin: float = -0.1,
                                                          vmax: float = 0.1) -> None:
        """ Plot amplitudes comparing two results """
        delta_probs = cast(np.ndarray, self._theoretical_amplitudes_matrix) - \
            cast(np.ndarray, self._amplitudes_matrices[results_index])
        self._plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def _compute_percentage_delta_values(self,
                                         delta_values: List[List[float]],
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

    def plot_probabilities_comparison_percentage(self,
                                                 results_index: int,
                                                 title: str = 'Deviation in % from theoric ' +
                                                 'probability (simulation vs. theory',
                                                 bar_label: str = 'Probabilities Delta (%)',
                                                 vmin: float = -40.,
                                                 vmax: float = 0.0) -> None:
        """ Plot probabilities comparing theoretical results displaying relative differences """
        delta_probs = cast(np.ndarray, self._theoretical_probabilities_matrix) - \
            cast(np.ndarray, self._probabilities_matrices[results_index])
        percentage_delta_probs = self._compute_percentage_delta_values(
            delta_probs, self._theoretical_probabilities_matrix)
        self._plot_comparison_between_two_results(percentage_delta_probs, title, bar_label, vmin, vmax)

    def plot_amplitudes_comparison_percentage(self,
                                              results_index: int,
                                              title: str = 'Deviation in % from theoric ' +
                                              'amplitude (simulation vs. theory',
                                              bar_label: str = 'Amplitude Delta (%)',
                                              vmin: float = -40.,
                                              vmax: float = 0.0) -> None:
        """ Plot amplitudes comparing theoretical results displaying relative differences """
        delta_amplitudes = cast(np.ndarray, self._theoretical_amplitudes_matrix) - \
            cast(np.ndarray, self._amplitudes_matrices[results_index])
        percentage_delta_amplitudes = self._compute_percentage_delta_values(
            delta_amplitudes, self._theoretical_probabilities_matrix)
        self._plot_comparison_between_two_results(percentage_delta_amplitudes, title, bar_label, vmin, vmax)

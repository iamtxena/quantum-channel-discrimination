from abc import ABC
from typing import Optional, List, Union, cast
from ..typings.configurations import OptimalConfigurations
from .aux import (build_probabilities_matrix, build_amplitudes_matrix,
                  plot_comparison_between_two_results, compute_percentage_delta_values)
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

    def save_results_to_disk(self, name: str, path: Optional[str] = "") -> None:
        for idx, result in enumerate(self._results):
            with open(f'./{path}{name}_{idx}.pkl', 'wb') as file:
                pickle.dump(result, file, pickle.HIGHEST_PROTOCOL)

    def load_results_from_file(self, name: str, path: Optional[str] = "") -> None:
        with open(f'./{path}{name}.pkl', 'rb') as file:
            self._results.append(pickle.load(file))

    def load_theoretical_results_from_file(self, name: str) -> None:
        theoretical_results = []
        with open('results/' + name + '.pkl', 'rb') as file:
            theoretical_results.append(pickle.load(file))
        self._theoretical_probabilities_matrix = build_probabilities_matrix(theoretical_results)
        self._theoretical_amplitudes_matrix = build_amplitudes_matrix(theoretical_results)

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
            probs1 = build_probabilities_matrix(result)
            self._probabilities_matrices.append(probs1)

    def build_amplitudes_matrix(self) -> None:
        """ Build amplitudes matrix for all loaded results """
        for result in self._results:
            amp1 = build_amplitudes_matrix(result)
            self._amplitudes_matrices.append(amp1)

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
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

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
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

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
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

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
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

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
        percentage_delta_probs = compute_percentage_delta_values(
            delta_probs, self._theoretical_probabilities_matrix)
        plot_comparison_between_two_results(percentage_delta_probs, title, bar_label, vmin, vmax)

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
        percentage_delta_amplitudes = compute_percentage_delta_values(
            delta_amplitudes, self._theoretical_probabilities_matrix)
        plot_comparison_between_two_results(percentage_delta_amplitudes, title, bar_label, vmin, vmax)

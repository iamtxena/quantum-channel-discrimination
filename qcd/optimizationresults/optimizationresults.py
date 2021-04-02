from abc import ABC, abstractmethod
from typing import Optional, List, Union, cast
from ..typings import TheoreticalOptimizationSetup
from ..typings.configurations import OptimalConfigurations, TheoreticalOptimalConfigurations
from .aux import (build_probabilities_matrix, build_amplitudes_matrix,
                  plot_comparison_between_two_results, compute_percentage_delta_values, plot_one_result)
import pickle
import numpy as np


class OptimizationResults(ABC):
    """ Generic class acting as an interface for any Optimization Results """

    def __init__(self, optimal_configurations: Optional[OptimalConfigurations] = None) -> None:
        self._results: List[OptimalConfigurations] = []
        self._probabilities_matrices: List[List[List[float]]] = []
        self._amplitudes_matrices: List[List[List[float]]] = []
        if optimal_configurations is not None:
            self._results = [optimal_configurations]
            self._build_theoretical_matrices_result()

    """ save and load results to and from a file """

    def save_results_to_disk(self, name: str, path: Optional[str] = "") -> None:
        for idx, result in enumerate(self._results):
            with open(f'./{path}{name}_{idx}.pkl', 'wb') as file:
                pickle.dump(result, file, pickle.HIGHEST_PROTOCOL)

    def load_results_from_file(self, name: str, path: Optional[str] = "") -> None:
        with open(f'./{path}{name}.pkl', 'rb') as file:
            self._results.append(pickle.load(file))

    @abstractmethod
    def _compute_theoretical_optimal_results(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOptimalConfigurations:
        pass

    def _build_theoretical_matrices_result(self) -> None:

        theoretical_results = self._compute_theoretical_optimal_results(
            {'eta_pairs': self._results[0]['eta_pairs']})
        self._theoretical_probabilities_matrix = build_probabilities_matrix(theoretical_results)
        self._theoretical_amplitudes_matrix = build_amplitudes_matrix(theoretical_results)

    def load_results(self, file_names: Union[str, List[str]], path: Optional[str] = "") -> None:
        """
          1. Load results from file for all file names
          2. build probability matrices for each loaded result
          3. build amplitude matrix for each loaded result
        """
        names = file_names
        if not isinstance(names, List):
            names = [cast(str, file_names)]

        for file_name in names:
            self.load_results_from_file(file_name, path)

        self._build_probabilities_matrix()
        self._build_amplitudes_matrix()
        self._build_theoretical_matrices_result()

    def _build_probabilities_matrix(self) -> None:
        """ Build probabilities matrix for all loaded results """
        self._probabilities_matrices = []
        for result in self._results:
            probs1 = build_probabilities_matrix(result)
            self._probabilities_matrices.append(probs1)

    def _build_amplitudes_matrix(self) -> None:
        """ Build amplitudes matrix for all loaded results """
        self._amplitudes_matrices = []
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
        plot_one_result(self._probabilities_matrices[results_index], title, bar_label, vmin, vmax)

    def plot_theoretical_probabilities(self,
                                       title: str = 'Probabilities from theory',
                                       bar_label: str = 'Probabilities value',
                                       vmin: float = 0.0,
                                       vmax: float = 1.0) -> None:
        """ Plot theoretical probabilities analysis """
        plot_one_result(self._theoretical_probabilities_matrix, title, bar_label, vmin, vmax)

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
        plot_one_result(self._amplitudes_matrices[results_index], title, bar_label, vmin, vmax)

    def plot_theoretical_amplitudes(self,
                                    title: str = 'Input state amplitude |1> obtained from theory',
                                    bar_label: str = 'Amplitude value',
                                    vmin: float = 0.0,
                                    vmax: float = 1.0) -> None:
        """ Plot theoretical amplitudes analysis """
        plot_one_result(self._theoretical_amplitudes_matrix, title, bar_label, vmin, vmax)

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
                                                          vmin: float = -1,
                                                          vmax: float = 1) -> None:
        """ Plot amplitudes comparing two results """
        delta_probs = cast(np.ndarray, self._theoretical_amplitudes_matrix) - \
            cast(np.ndarray, self._amplitudes_matrices[results_index])
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_probabilities_comparison_percentage(self,
                                                 results_index: int,
                                                 title: str = 'Deviation in % from theoric ' +
                                                 'probability (thoery vs. simulation)',
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
                                              'amplitude (theory vs. simulation)',
                                              bar_label: str = 'Amplitude Delta (%)',
                                              vmin: float = -40.,
                                              vmax: float = 0.0) -> None:
        """ Plot amplitudes comparing theoretical results displaying relative differences """
        delta_amplitudes = cast(np.ndarray, self._theoretical_amplitudes_matrix) - \
            cast(np.ndarray, self._amplitudes_matrices[results_index])
        percentage_delta_amplitudes = compute_percentage_delta_values(
            delta_amplitudes, self._theoretical_probabilities_matrix)
        plot_comparison_between_two_results(percentage_delta_amplitudes, title, bar_label, vmin, vmax)

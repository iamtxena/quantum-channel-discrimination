from abc import ABC, abstractmethod
from typing import cast
from ..typings import TheoreticalOptimizationSetup
from ..typings.configurations import OptimalConfigurations, TheoreticalOptimalConfigurations
from .aux import (build_probabilities_matrix, build_amplitudes_matrix,
                  plot_comparison_between_two_results, compute_percentage_delta_values, plot_one_result)
import numpy as np


class OptimizationResult(ABC):
    """ Generic class acting as an interface for any Optimization Result """

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        theoretical_result = self._prepare_etas_and_compute_theoretical_result(optimal_configurations)

        self._probabilities_matrix = build_probabilities_matrix(optimal_configurations)
        self._amplitudes_matrix = build_amplitudes_matrix(optimal_configurations)
        self._theoretical_probabilities_matrix = build_probabilities_matrix(theoretical_result)
        self._theoretical_amplitudes_matrix = build_amplitudes_matrix(theoretical_result)

    def _prepare_etas_and_compute_theoretical_result(
            self,
            optimal_configurations: OptimalConfigurations) -> TheoreticalOptimalConfigurations:
        eta_pairs = optimal_configurations.get('eta_pairs')
        if eta_pairs is None:
            raise ValueError('eta_pairs is mandatory')
        theoretical_result = self._compute_theoretical_optimal_result(
            {'eta_pairs': eta_pairs})
        return theoretical_result

    @abstractmethod
    def _compute_theoretical_optimal_result(
            self,
            optimization_setup: TheoreticalOptimizationSetup) -> TheoreticalOptimalConfigurations:
        pass

    def plot_probabilities(self,
                           results_index: int,
                           title: str = 'Probabilities from simulation',
                           bar_label: str = 'Probabilities value',
                           vmin: float = 0.0,
                           vmax: float = 1.0) -> None:
        """ Plot probabilities analysis """
        plot_one_result(self._probabilities_matrix[results_index], title, bar_label, vmin, vmax)

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
        delta_probs = cast(np.ndarray, self._probabilities_matrix[results_index1]) - \
            cast(np.ndarray, self._probabilities_matrix[results_index2])
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_probabilities_comparison_with_theoretical_result(self,
                                                              results_index: int,
                                                              title: str = 'Difference in Probabilities' +
                                                              '(theory vs. simulation)',
                                                              bar_label: str = 'Probabilities Delta value',
                                                              vmin: float = -0.1,
                                                              vmax: float = 0.1) -> None:
        """ Plot probabilities comparing theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_probabilities_matrix) - \
            cast(np.ndarray, self._probabilities_matrix[results_index])
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_amplitudes(self,
                        results_index: int,
                        title: str = 'Input state amplitude |1> obtained from simulation',
                        bar_label: str = 'Amplitude value',
                        vmin: float = 0.0,
                        vmax: float = 1.0) -> None:
        """ Plot amplitudes analysis """
        plot_one_result(self._amplitudes_matrix[results_index], title, bar_label, vmin, vmax)

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
        delta_probs = cast(np.ndarray, self._amplitudes_matrix[results_index1]) - \
            cast(np.ndarray, self._amplitudes_matrix[results_index2])
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_amplitudes_comparison_with_theoretical_result(self,
                                                           results_index: int,
                                                           title: str = 'Difference in Amplitudes' +
                                                           '(theory vs. simulation)',
                                                           bar_label: str = 'Amplitude Delta value',
                                                           vmin: float = -1,
                                                           vmax: float = 1) -> None:
        """ Plot amplitudes comparing theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_amplitudes_matrix) - \
            cast(np.ndarray, self._amplitudes_matrix[results_index])
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
            cast(np.ndarray, self._probabilities_matrix[results_index])
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
            cast(np.ndarray, self._amplitudes_matrix[results_index])
        percentage_delta_amplitudes = compute_percentage_delta_values(
            delta_amplitudes, self._theoretical_probabilities_matrix)
        plot_comparison_between_two_results(percentage_delta_amplitudes, title, bar_label, vmin, vmax)

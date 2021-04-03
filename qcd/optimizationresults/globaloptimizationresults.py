from abc import ABC
from typing import Optional, List, Union, cast
from ..typings.configurations import OptimalConfigurations
from .aux import (load_result_from_file, plot_one_result,
                  plot_comparison_between_two_results, compute_percentage_delta_values)
from .global_aux import build_optimization_result
from .theoreticaloptimizationresults import (
    TheoreticalOneShotOptimizationResult, TheoreticalOneShotEntangledOptimizationResult)
from ..typings.theoreticalresult import TheoreticalResult, STRATEGY
import numpy as np


class GlobalOptimizationResults(ABC):
    """ Global class to load, process and plot any Optimization Results """

    @staticmethod
    def load_results(file_names: Union[str, List[str]], path: Optional[str] = ""):
        """
          1. Load results from file for all file names
          2. build probability matrices for each loaded result
          3. build amplitude matrix for each loaded result
        """
        names = file_names
        if not isinstance(names, List):
            names = [cast(str, file_names)]

        results = [load_result_from_file(file_name, path) for file_name in names]
        return GlobalOptimizationResults(results)

    def __init__(self, optimal_configurations: Union[OptimalConfigurations, List[OptimalConfigurations]]) -> None:
        optimal_results = optimal_configurations
        if not isinstance(optimal_configurations, List):
            optimal_results = [optimal_configurations]
        optimal_results = cast(List[OptimalConfigurations], optimal_results)
        self._optimization_results = [build_optimization_result(optimal_result) for optimal_result in optimal_results]
        self._build_all_theoretical_optimizations_results(optimal_results[0])

    def _build_all_theoretical_optimizations_results(self, optimal_configurations: OptimalConfigurations) -> None:
        """ Build the theoretical optimization result for each damping channel supported """
        self._theoretical_results: TheoreticalResult = {
            'one_shot': TheoreticalOneShotOptimizationResult(optimal_configurations),
            'one_shot_side_entanglement':
            TheoreticalOneShotEntangledOptimizationResult(optimal_configurations),
        }

    def plot_probabilities(self,
                           results_index: int,
                           title: str = 'Probabilities from simulation',
                           bar_label: str = 'Probabilities value',
                           vmin: float = 0.0,
                           vmax: float = 1.0) -> None:
        """ Plot probabilities analysis """
        plot_one_result(self._optimization_results[results_index].probabilities_matrix, title, bar_label, vmin, vmax)

    def plot_theoretical_probabilities(self,
                                       strategy: STRATEGY = 'one_shot',
                                       title: str = 'Probabilities from theory',
                                       bar_label: str = 'Probabilities value',
                                       vmin: float = 0.0,
                                       vmax: float = 1.0) -> None:
        """ Plot theoretical probabilities analysis """
        plot_one_result(
            self._theoretical_results[strategy].probabilities_matrix, title, bar_label, vmin, vmax)

    def plot_amplitudes(self,
                        results_index: int,
                        title: str = 'Input state amplitude |1> obtained from simulation',
                        bar_label: str = 'Amplitude value',
                        vmin: float = 0.0,
                        vmax: float = 1.0) -> None:
        """ Plot amplitudes analysis """
        plot_one_result(self._optimization_results[results_index].amplitudes_matrix, title, bar_label, vmin, vmax)

    def plot_theoretical_amplitudes(self,
                                    strategy: STRATEGY = 'one_shot',
                                    title: str = 'Input state amplitude |1> obtained from theory',
                                    bar_label: str = 'Amplitude value',
                                    vmin: float = 0.0,
                                    vmax: float = 1.0) -> None:
        """ Plot theoretical amplitudes analysis """
        plot_one_result(self._theoretical_results[strategy].amplitudes_matrix, title, bar_label, vmin, vmax)

    def plot_probabilities_comparison(self,
                                      results_index1: int,
                                      results_index2: int,
                                      title: str = 'Difference in Probabilities from simulation',
                                      bar_label: str = 'Probabilities value',
                                      vmin: float = -0.1,
                                      vmax: float = 0.1) -> None:
        """ Plot probabilities comparing two results """
        delta_probs = cast(np.ndarray, self._optimization_results[results_index1].probabilities_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index2].probabilities_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_theoretical_probabilities_comparison(
            self,
            first_strategy: STRATEGY = 'one_shot',
            second_strategy: STRATEGY = 'one_shot_side_entanglement',
            title: str = 'Difference in Probabilities from theoretical strategies',
            bar_label: str = 'Probabilities value',
            vmin: float = -0.1,
            vmax: float = 0.1) -> None:
        """ Plot probabilities comparing two results """
        delta_probs = cast(np.ndarray, self._theoretical_results[first_strategy].probabilities_matrix) - \
            cast(np.ndarray, self._theoretical_results[second_strategy].probabilities_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_probabilities_comparison_with_theoretical_result(
            self,
            results_index: int,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Difference in Probabilities' +
            '(theory vs. simulation)',
            bar_label: str = 'Probabilities Delta value',
            vmin: float = -0.1,
            vmax: float = 0.1) -> None:
        """ Plot probabilities comparing theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_results[strategy].probabilities_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].probabilities_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_amplitudes_comparison(self,
                                   results_index1: int,
                                   results_index2: int,
                                   title: str = 'Difference in Amplitudes (between simulations)',
                                   bar_label: str = 'Amplitude value',
                                   vmin: float = -1.0,
                                   vmax: float = 1.0) -> None:
        """ Plot amplitudes comparing two results """
        delta_probs = cast(np.ndarray, self._optimization_results[results_index1].amplitudes_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index2].amplitudes_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_theoretical_amplitudes_comparison(
            self,
            first_strategy: STRATEGY = 'one_shot',
            second_strategy: STRATEGY = 'one_shot_side_entanglement',
            title: str = 'Difference in Amplitudes from theoretical strategies',
            bar_label: str = 'Amplitude value',
            vmin: float = -1.0,
            vmax: float = 1.0) -> None:
        """ Plot amplitudes comparing two theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_results[first_strategy].amplitudes_matrix) - \
            cast(np.ndarray, self._theoretical_results[second_strategy].amplitudes_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_amplitudes_comparison_with_theoretical_result(
            self,
            results_index: int,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Difference in Amplitudes' +
            '(theory vs. simulation)',
            bar_label: str = 'Amplitude Delta value',
            vmin: float = -1,
            vmax: float = 1) -> None:
        """ Plot amplitudes comparing theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_results[strategy].amplitudes_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].amplitudes_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax)

    def plot_probabilities_comparison_percentage(
            self,
            results_index: int,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Deviation in % from theoric ' +
            'probability (thoery vs. simulation)',
            bar_label: str = 'Probabilities Delta (%)',
            vmin: float = -40.,
            vmax: float = 0.0) -> None:
        """ Plot probabilities comparing theoretical results displaying relative differences """
        delta_probs = cast(np.ndarray, self._theoretical_results[strategy].probabilities_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].probabilities_matrix)
        percentage_delta_probs = compute_percentage_delta_values(
            delta_probs, self._theoretical_results[strategy].probabilities_matrix)
        plot_comparison_between_two_results(percentage_delta_probs, title, bar_label, vmin, vmax)

    def plot_amplitudes_comparison_percentage(
            self,
            results_index: int,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Deviation in % from theoric ' +
            'amplitude (theory vs. simulation)',
            bar_label: str = 'Amplitude Delta (%)',
            vmin: float = -40.,
            vmax: float = 0.0) -> None:
        """ Plot amplitudes comparing theoretical results displaying relative differences """
        delta_amplitudes = cast(np.ndarray, self._theoretical_results[strategy].amplitudes_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].amplitudes_matrix)
        percentage_delta_amplitudes = compute_percentage_delta_values(
            delta_amplitudes, self._theoretical_results[strategy].amplitudes_matrix)
        plot_comparison_between_two_results(percentage_delta_amplitudes, title, bar_label, vmin, vmax)
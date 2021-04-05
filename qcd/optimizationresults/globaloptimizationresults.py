from abc import ABC
from qcd.circuits.aux import set_only_eta_pairs
from typing import Optional, List, Union, cast, Dict
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
        self._optimal_configurations = optimal_configurations if isinstance(
            optimal_configurations, List) else [optimal_configurations]
        self._optimal_configurations = set_only_eta_pairs(cast(List[Dict], self._optimal_configurations))
        self._optimization_results = [build_optimization_result(
            optimal_result) for optimal_result in self._optimal_configurations]
        self._build_all_theoretical_optimizations_results(len(self._optimization_results[0].probabilities_matrix))

    def _build_all_theoretical_optimizations_results(self, number_etas: int) -> None:
        """ Build the theoretical optimization result for each damping channel supported """
        self._theoretical_results: TheoreticalResult = {
            'one_shot': TheoreticalOneShotOptimizationResult(number_etas),
            'one_shot_side_entanglement':
            TheoreticalOneShotEntangledOptimizationResult(number_etas),
        }

    def add_results(self, file_names: Union[str, List[str]], path: Optional[str] = "") -> None:
        """ Load more results from given files and add them to the existing results """
        names = file_names
        if not isinstance(names, List):
            names = [cast(str, file_names)]

        new_results = [load_result_from_file(file_name, path) for file_name in names]
        self._optimal_configurations.extend(new_results)
        new_optimization_results = [build_optimization_result(optimal_result) for optimal_result in new_results]
        self._optimization_results.extend(new_optimization_results)

    def plot_probabilities(self,
                           results_index: int = 0,
                           title: str = 'Probabilities from simulation',
                           bar_label: str = 'Probabilities value',
                           vmin: float = 0.0,
                           vmax: float = 1.0,
                           cmap='viridis') -> None:
        """ Plot probabilities analysis """
        plot_one_result(
            self._optimization_results[results_index].probabilities_matrix, title, bar_label, vmin, vmax, cmap)

    def plot_theoretical_probabilities(self,
                                       strategy: STRATEGY = 'one_shot',
                                       title: str = 'Probabilities from theory',
                                       bar_label: str = 'Probabilities value',
                                       vmin: float = 0.0,
                                       vmax: float = 1.0,
                                       cmap='viridis') -> None:
        """ Plot theoretical probabilities analysis """
        plot_one_result(
            self._theoretical_results[strategy].probabilities_matrix, title, bar_label, vmin, vmax, cmap)

    def plot_amplitudes(self,
                        results_index: int = 0,
                        title: str = 'Input state amplitude |1> obtained from simulation',
                        bar_label: str = 'Amplitude value',
                        vmin: float = 0.0,
                        vmax: float = 1.0,
                        cmap='viridis') -> None:
        """ Plot amplitudes analysis """
        plot_one_result(self._optimization_results[results_index].amplitudes_matrix, title, bar_label, vmin, vmax, cmap)

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
                                      vmax: float = 0.1,
                                      cmap='RdBu') -> None:
        """ Plot probabilities comparing two results """
        delta_probs = cast(np.ndarray, self._optimization_results[results_index1].probabilities_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index2].probabilities_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax, cmap)

    def plot_theoretical_probabilities_comparison(
            self,
            first_strategy: STRATEGY = 'one_shot_side_entanglement',
            second_strategy: STRATEGY = 'one_shot',
            title: str = 'Difference in Probabilities from theoretical strategies',
            bar_label: str = 'Probabilities value',
            vmin: float = 0,
            vmax: float = 0.05,
            cmap='viridis') -> None:
        """ Plot probabilities comparing two results """
        delta_probs = cast(np.ndarray, self._theoretical_results[first_strategy].probabilities_matrix) - \
            cast(np.ndarray, self._theoretical_results[second_strategy].probabilities_matrix)
        vmin = np.min(delta_probs)
        vmax = np.max(delta_probs)
        print(f'min: {vmin}, max {vmax}')
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax, cmap)

    def plot_probabilities_comparison_with_theoretical_result(
            self,
            results_index: int = 0,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Difference in Probabilities' +
            '(theory vs. simulation)',
            bar_label: str = 'Probabilities Delta value',
            vmin: float = -0.1,
            vmax: float = 0.1,
            cmap='RdBu') -> None:
        """ Plot probabilities comparing theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_results[strategy].probabilities_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].probabilities_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax, cmap)

    def plot_amplitudes_comparison(self,
                                   results_index1: int,
                                   results_index2: int,
                                   title: str = 'Difference in Amplitudes (between simulations)',
                                   bar_label: str = 'Amplitude value',
                                   vmin: float = -1.0,
                                   vmax: float = 1.0,
                                   cmap='RdBu') -> None:
        """ Plot amplitudes comparing two results """
        delta_probs = cast(np.ndarray, self._optimization_results[results_index1].amplitudes_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index2].amplitudes_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax, cmap)

    def plot_theoretical_amplitudes_comparison(
            self,
            first_strategy: STRATEGY = 'one_shot',
            second_strategy: STRATEGY = 'one_shot_side_entanglement',
            title: str = 'Difference in Amplitudes from theoretical strategies',
            bar_label: str = 'Amplitude value',
            vmin: float = -1.0,
            vmax: float = 1.0,
            cmap='RdBu') -> None:
        """ Plot amplitudes comparing two theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_results[first_strategy].amplitudes_matrix) - \
            cast(np.ndarray, self._theoretical_results[second_strategy].amplitudes_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax, cmap)

    def plot_amplitudes_comparison_with_theoretical_result(
            self,
            results_index: int = 0,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Difference in Amplitudes' +
            '(theory vs. simulation)',
            bar_label: str = 'Amplitude Delta value',
            vmin: float = -1,
            vmax: float = 1,
            cmap='RdBu') -> None:
        """ Plot amplitudes comparing theoretical results """
        delta_probs = cast(np.ndarray, self._theoretical_results[strategy].amplitudes_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].amplitudes_matrix)
        plot_comparison_between_two_results(delta_probs, title, bar_label, vmin, vmax, cmap)

    def plot_probabilities_comparison_percentage(
            self,
            results_index: int = 0,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Deviation in % from theoric ' +
            'probability (theory vs. simulation)',
            bar_label: str = 'Probabilities Delta (%)',
            vmin: float = -40.,
            vmax: float = 0.0,
            cmap='RdBu') -> None:
        """ Plot probabilities comparing theoretical results displaying relative differences """
        delta_probs = cast(np.ndarray, self._theoretical_results[strategy].probabilities_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].probabilities_matrix)
        percentage_delta_probs = compute_percentage_delta_values(
            delta_probs, self._theoretical_results[strategy].probabilities_matrix)
        plot_comparison_between_two_results(percentage_delta_probs, title, bar_label, vmin, vmax, cmap)

    def plot_amplitudes_comparison_percentage(
            self,
            results_index: int = 0,
            strategy: STRATEGY = 'one_shot',
            title: str = 'Deviation in % from theoric ' +
            'amplitude (theory vs. simulation)',
            bar_label: str = 'Amplitude Delta (%)',
            vmin: float = -40.,
            vmax: float = 0.0,
            cmap='RdBu') -> None:
        """ Plot amplitudes comparing theoretical results displaying relative differences """
        delta_amplitudes = cast(np.ndarray, self._theoretical_results[strategy].amplitudes_matrix) - \
            cast(np.ndarray, self._optimization_results[results_index].amplitudes_matrix)
        percentage_delta_amplitudes = compute_percentage_delta_values(
            delta_amplitudes, self._theoretical_results[strategy].amplitudes_matrix)
        plot_comparison_between_two_results(percentage_delta_amplitudes, title, bar_label, vmin, vmax, cmap)

    def plot_theoretical_improvement(self,
                                     title: str = 'Improvement on Side Entanglement Theory',
                                     bar_label: str = 'Improvement value',
                                     vmin: float = 0.0,
                                     vmax: float = 0.05,
                                     cmap='viridis') -> None:
        """ Plot theoretical improvement analysis """
        vmin = np.min(self._theoretical_results['one_shot_side_entanglement']._improvement_matrix)
        vmax = np.max(self._theoretical_results['one_shot_side_entanglement']._improvement_matrix)
        print(f'min: {vmin}, max {vmax}')
        plot_one_result(cast(TheoreticalOneShotEntangledOptimizationResult,
                             self._theoretical_results['one_shot_side_entanglement'])._improvement_matrix,
                        title, bar_label, vmin, vmax, cmap)

from abc import ABC
from typing import Optional, List
from ..typings.configurations import OptimalConfigurations
import pickle


class OptimizationResults(ABC):
    """ Generic class acting as an interface for any Optimization Results """

    def __init__(self, optimal_configurations: Optional[OptimalConfigurations] = None) -> None:
        self._results = optimal_configurations

    """ save and load results to and from a file """

    def save_results_to_disk(self, name: str) -> None:
        with open('results/' + name + '.pkl', 'wb') as f:
            pickle.dump(self._results, f, pickle.HIGHEST_PROTOCOL)

    def load_results_from_file(self, name: str) -> None:
        with open('results/' + name + '.pkl', 'rb') as f:
            self._results = pickle.load(f)

    def load_results(self, file_names: List[str]) -> None:
        """
          1. Load results from file for all file names
          2. build probability matrices for each loaded result
          3. build theoretical probabilities matrix
          4. build amplitude matrix for each loaded result
          5. build theoretical amplitude matrix
        """

    def build_probabilities_matrix(self) -> None:
        """ Plot probabilities analysis """

    def build_amplitudes_matrix(self) -> None:
        """ Plot probabilities analysis """

    def compute_theoretical_results(self) -> None:
        """ Plot probabilities analysis """

    def plot_analysis(self) -> None:
        """ Plot all probabilities and amplitudes analysis """

    def plot_probabilities(self) -> None:
        """ Plot probabilities analysis """

    def plot_amplitudes(self) -> None:
        """ Plot probabilities analysis """

    def plot_probabilities_comparison(self) -> None:
        """ Plot probabilities analysis """

    def plot_amplitudes_comparison(self) -> None:
        """ Plot probabilities analysis """

    def plot_probabilities_comparison_percentage(self) -> None:
        """ Plot probabilities analysis """

    def plot_amplitudes_comparison_percentage(self) -> None:
        """ Plot probabilities analysis """

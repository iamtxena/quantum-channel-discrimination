from abc import ABC
from typing import Optional
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

from abc import ABC, abstractmethod
from ..typings import TheoreticalOptimizationSetup
from ..typings.configurations import OptimalConfigurations
from ..typings.configurations import OptimalConfiguration


class TheoreticalOptimization(ABC):
    """ Generic class acting as an interface for any Theoretical Channel Optimization """

    def __init__(self, optimization_setup: TheoreticalOptimizationSetup):
        self._eta_pairs = optimization_setup['eta_pairs']
        self._global_eta_pair = (0.0, 0.0)

    @abstractmethod
    def _compute_theoretical_best_configuration(self) -> OptimalConfiguration:
        """ Find out the theoretical best configuration with a global pair of etas (channels) """

    def compute_theoretical_optimal_results(self) -> OptimalConfigurations:
        """ Finds out the theoretical optimal configuration for each pair of attenuation levels """
        probabilities = []
        configurations = []
        best_algorithm = []
        number_calls_made = []

        print("Starting the theoretical computation")

        for eta_pair in self._eta_pairs:
            self._global_eta_pair = eta_pair
            result = self._compute_theoretical_best_configuration()
            probabilities.append(result['best_probability'])
            configurations.append(result['best_configuration'])
            best_algorithm.append(result['best_algorithm'])
            number_calls_made.append(result['number_calls_made'])

        print(f'Total pair of etas tested: {len(self._eta_pairs)}')

        return {
            'eta_pairs': self._eta_pairs,
            'best_algorithm': best_algorithm,
            'probabilities': probabilities,
            'configurations': configurations,
            'number_calls_made': number_calls_made}

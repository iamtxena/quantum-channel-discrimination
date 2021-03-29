from typing import List, Tuple
from . import Optimization
from ..optimizations import OptimalConfigurations, OneShotOptimalConfigurations
from ..typings import OptimizationSetup
import itertools


class OneShotOptimization(Optimization):
    """ Representation of the One Shot Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        self._setup = optimization_setup
        self._attenuation_pairs = self._get_combinations_two_lambdas_without_repeats()

    def find_optimal_configurations(self,
                                    optimization_setup: OptimizationSetup) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """

        return OneShotOptimalConfigurations()

    def _get_combinations_two_lambdas_without_repeats(self) -> List[Tuple[float, float]]:
        """ from a given list of attenuations (lambdas) create a
            list of all combinatorial pairs of possible lambdas
            without repeats (order does not matter).
            For us it is the same testing first lambda 0.1 and second lambda 0.2
            than first lambda 0.2 and second lambda 0.1
        """
        list_lambda = self._setup['attenuation_factors']
        # when there is only one element, we add the same element
        if len(list_lambda) == 1:
            list_lambda.append(list_lambda[0])
        # get combinations of two lambdas without repeats
        return list(itertools.combinations(list_lambda, 2))

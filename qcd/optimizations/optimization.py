from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, cast
from ..typings import OptimizationSetup, CloneSetup
from ..typings.configurations import OptimalConfiguration, OptimalConfigurations
from ..configurations import ChannelConfiguration
from .aux import get_combinations_n_etas_without_repeats, get_combinations_two_etas_without_repeats_from_etas
import numpy as np
from qiskit.aqua.components.optimizers import SLSQP, L_BFGS_B, ADAM, CRS, DIRECT_L, DIRECT_L_RAND, ESCH, ISRES


class Optimization(ABC):
    """ Generic class acting as an interface for any Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        self._setup = optimization_setup
        self._add_initial_parameters_and_variable_bounds_to_optimization_setup()
        if self._setup['number_third_channels'] <= 1:
            self._eta_groups = get_combinations_two_etas_without_repeats_from_etas(self._setup['eta_partitions'])
            self._global_eta_group = [0.0]
        else:
            self._eta_groups = get_combinations_n_etas_without_repeats(self._setup['number_channels_to_discriminate'],
                                                                       self._setup['eta_partitions'],
                                                                       self._setup['number_third_channels'])
            self._global_eta_group = [0.0] * optimization_setup['number_channels_to_discriminate']

    @abstractmethod
    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_group: List[float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a channel configuration """
        pass

    @abstractmethod
    def _cost_function(self, params: List[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
              defined in the optimization setup.
              Cost is computed as 1 (perfect probability) - average success probability for
              all the plays with the given configuration
              Returns the Cost (error probability).
          """
        pass

    @abstractmethod
    def _best_configuration_to_print(self, best_configuration: List[float]) -> str:
        pass

    def find_optimal_configurations(self, clone_setup: Optional[CloneSetup]) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        pass

    def _compute_best_configuration(self) -> OptimalConfiguration:
        """ Find out the best configuration with a global pair of etas (channels) trying out
            a list of specified optimization algorithm """
        optimizer_algorithms = self._setup['optimizer_algorithms']
        optimizer_iterations = self._setup['optimizer_iterations']
        best_probability = 0
        best_configuration = []
        best_optimizer_algorithm = ""
        number_calls_made = 0

        for optimizer_algorithm, max_evals in zip(optimizer_algorithms, optimizer_iterations):
            print("Analyzing Optimizer Algorithm: ", optimizer_algorithm)
            if optimizer_algorithm == 'ADAM':
                optimizer = ADAM(maxiter=max_evals)
            if optimizer_algorithm == 'SLSQP':
                optimizer = SLSQP(maxiter=max_evals)
            if optimizer_algorithm == 'L_BFGS_B':
                optimizer = L_BFGS_B(maxfun=max_evals, maxiter=max_evals)
            if optimizer_algorithm == 'CRS':
                optimizer = CRS(max_evals=max_evals)
            if optimizer_algorithm == 'DIRECT_L':
                optimizer = DIRECT_L(max_evals=max_evals)
            if optimizer_algorithm == 'DIRECT_L_RAND':
                optimizer = DIRECT_L_RAND(max_evals=max_evals)
            if optimizer_algorithm == 'ESCH':
                optimizer = ESCH(max_evals=max_evals)
            if optimizer_algorithm == 'ISRES':
                optimizer = ISRES(max_evals=max_evals)

            ret = optimizer.optimize(num_vars=len(self._setup['initial_parameters']),
                                     objective_function=self._cost_function,
                                     variable_bounds=self._setup['variable_bounds'],
                                     initial_point=self._setup['initial_parameters'])
        print("Best Average Probability:", -ret[1])
        if (-ret[1]) > best_probability:
            best_configuration = ret[0]
            best_probability = -ret[1]
            number_calls_made = ret[2]
            best_optimizer_algorithm = optimizer_algorithm

        # Print results
        print("Final Best Optimizer Algorithm: ", best_optimizer_algorithm)
        print("Final Best Average Probability:", best_probability)
        print("Number of cost function calls made:", number_calls_made)
        print(self._best_configuration_to_print(best_configuration))

        return {'best_algorithm': best_optimizer_algorithm,
                'best_probability': best_probability,
                'best_configuration': self._convert_optimizer_results_to_channel_configuration(best_configuration,
                                                                                               self._global_eta_group),
                'number_calls_made': number_calls_made}

    def _add_initial_parameters_and_variable_bounds_to_optimization_setup(self) -> None:
        """ Update the optimization setup with intial parameters and variable bounds """
        self._setup['initial_parameters'] = [0] * 5
        variable_bounds = [(0, 1)]  # amplitude_probability
        variable_bounds += [(0, 2 * np.pi)
                            for i in range(4)]
        self._setup['variable_bounds'] = cast(List[Tuple[float, float]], variable_bounds)

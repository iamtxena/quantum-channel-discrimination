from abc import ABC, abstractmethod
from typing import List, Tuple
from ..optimizations import OptimizationResults
from ..typings import OptimizationSetup, OptimalConfiguration, OptimalConfigurations
from ..configurations import ChannelConfiguration
import itertools
import time
import numpy as np
import math
import random
from qiskit.aqua.components.optimizers import CRS, DIRECT_L, DIRECT_L_RAND, ESCH, ISRES


class Optimization(ABC):
    """ Generic class acting as an interface for any Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        self._setup = optimization_setup
        self._attenuation_pairs = self._get_combinations_two_lambdas_without_repeats()
        self._global_attenuation_pair = (0.0, 0.0)

    def find_optimal_configurations(self) -> OptimizationResults:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        probabilities = []
        configurations = []
        best_algorithm = []

        program_start_time = time.time()
        print("Starting the execution")

        for lambda_pair in self._attenuation_pairs:
            start_time = time.time()
            self._global_attenuation_pair = lambda_pair
            result = self._compute_best_configuration()
            probabilities.append(result['best_probability'])
            configurations.append(result['best_configuration'])
            best_algorithm.append(result['best_algorithm'])
            end_time = time.time()
            print("total minutes taken this pair of lambdas: ", int(np.round((end_time - start_time) / 60)))
            print("total minutes taken so far: ", int(np.round((end_time - program_start_time) / 60)))

        end_time = time.time()
        print("total minutes of execution time: ", int(np.round((end_time - program_start_time) / 60)))
        print("All guesses have been calculated")
        print(f'Total pair of lambdas tested: {len(self._attenuation_pairs)}')

        return OptimizationResults(OptimalConfigurations(
            attenuation_pairs=self._attenuation_pairs,
            best_algorithm=best_algorithm,
            probabilities=probabilities,
            configurations=configurations))

    @abstractmethod
    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: np.ndarray[float],
                                                            attenuation_pair: Tuple[float, float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a channel configuration """
        pass

    @abstractmethod
    def _prepare_initial_state(self, theta: float, phase: float) -> Tuple[complex, complex]:
        """ Prepare initial state """
        pass

    @abstractmethod
    def _convert_counts_to_final_result(self, counts: str) -> int:
        """ Convert the execution result to the final measured value: 0 or 1 """
        pass

    def _set_random_lambda(self, attenuation_pair: Tuple[float, float]) -> int:
        """ return a random choice from attenuation pair with the correspondent index value """
        lambda_value = random.choice(attenuation_pair)
        if lambda_value == attenuation_pair[0]:
            return 0
        return 1

    @abstractmethod
    def _compute_damping_channel(self, channel_configuration: ChannelConfiguration, attenuation_index: int) -> int:
        pass

    @abstractmethod
    def _guess_lambda_used(self, real_measured_result: int) -> int:
        """ Decides which lambda was used on the real execution.
            It is a silly guess.
            It returns the same lambda used as the measured result
        """
        pass

    @abstractmethod
    def _cost_function(self, params: np.ndarray[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
              defined in the optimization setup.
              Cost is computed as 1 (perfect probability) - average success probability for
              all the plays with the given configuration
              Returns the Cost (error probability).
          """
        pass

    def _play_and_guess_one_case(self, channel_configuration: ChannelConfiguration) -> int:
        """ Execute a real execution with a random lambda from the two passed,
            guess which one was used on the exection and
            check the result.
            Returns 1 on success (it was a correct guess) or 0 on fail (it was an incorrect guess)
        """
        attenuation_index = self._set_random_lambda(channel_configuration.attenuation_pair)
        result = self._compute_damping_channel(channel_configuration, attenuation_index)
        guess_index_attenuation_factor = self._guess_lambda_used(result)

        return self._check_value(attenuation_index, guess_index_attenuation_factor)

    def _check_value(self, real_index_attenuation_factor: int,
                     guess_index_attenuation_factor: int):
        if real_index_attenuation_factor == guess_index_attenuation_factor:
            return 1
        return 0

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

    def _compute_best_configuration(self) -> OptimalConfiguration:
        """ Find out the best configuration with a global pair of lambdas (channels) trying out
            a list of specified optimization algorithm """
        optimizer_algorithms = self._setup['optimizer_algorithms']
        optimizer_iterations = self._setup['optimizer_iterations']
        best_probability = 0
        best_configuration = []
        best_optimizer_algorithm = ""

        for optimizer_algorithm, max_evals in zip(optimizer_algorithms, optimizer_iterations):
            print("Analyzing Optimizer Algorithm: ", optimizer_algorithm)
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
            print("Best Average Probability:", 1 - ret[1])
            if (1 - ret[1]) > best_probability:
                best_probability = 1 - ret[1]
                best_configuration = ret[0]
                best_optimizer_algorithm = optimizer_algorithm

        # Print results
        print("Final Best Optimizer Algorithm: ", best_optimizer_algorithm)
        print("Final Best Average Probability:", best_probability)
        print("Parameters Found: " + u"\u03B8" + " = " + str(int(math.degrees(best_configuration[0]))) + u"\u00B0" +
              ", Phase = " + str(int(math.degrees(best_configuration[1]))) + u"\u00B0" +
              ", " + u"\u03D5" + "rx = " + str(int(math.degrees(best_configuration[2]))) + u"\u00B0" +
              ", " + u"\u03D5" + "ry = " + str(int(math.degrees(best_configuration[3]))) + u"\u00B0" +
              ", " + u"\u03BB" + u"\u2080" + " = " + str(self._global_attenuation_pair[0]) +
              ", " + u"\u03BB" + u"\u2081" + " = " + str(self._global_attenuation_pair[1]))
        return OptimalConfiguration(
            best_algorithm=best_optimizer_algorithm,
            best_probability=best_probability,
            best_configuration=self._convert_optimizer_results_to_channel_configuration(best_configuration,
                                                                                        self._global_attenuation_pair)
        )

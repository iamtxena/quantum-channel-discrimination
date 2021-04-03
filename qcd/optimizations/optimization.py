from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ..typings import OptimizationSetup, GuessStrategy, CloneSetup
from ..typings.configurations import OptimalConfiguration, OptimalConfigurations
from ..configurations import ChannelConfiguration
from .aux import set_random_eta, check_value, get_combinations_two_etas_without_repeats_from_etas
import math
from qiskit.aqua.components.optimizers import L_BFGS_B, ADAM, CRS, DIRECT_L, DIRECT_L_RAND, ESCH, ISRES


class Optimization(ABC):
    """ Generic class acting as an interface for any Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        self._setup = optimization_setup
        self._eta_pairs = get_combinations_two_etas_without_repeats_from_etas(self._setup['attenuation_angles'])
        self._global_eta_pair = (0.0, 0.0)

    @abstractmethod
    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_pair: Tuple[float, float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a channel configuration """
        pass

    @abstractmethod
    def _prepare_initial_state(self, theta: float) -> Tuple[complex, complex]:
        """ Prepare initial state """
        pass

    @abstractmethod
    def _compute_damping_channel(self, channel_configuration: ChannelConfiguration, attenuation_index: int) -> int:
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
    def _convert_counts_to_eta_used(self,
                                    counts_dict: dict,
                                    guess_strategy: GuessStrategy) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        pass

    def find_optimal_configurations(self, clone_setup: Optional[CloneSetup]) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        pass

    def _play_and_guess_one_case(self, channel_configuration: ChannelConfiguration) -> int:
        """ Execute a real execution with a random eta from the two passed,
            guess which one was used on the execution and
            check the result.
            Returns 1 on success (it was a correct guess) or 0 on fail (it was an incorrect guess)
        """
        eta_index = set_random_eta(channel_configuration.eta_pair)
        guess_index_eta = self._compute_damping_channel(channel_configuration, eta_index)

        return check_value(eta_index, guess_index_eta)

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
            print("Best Average Probability:", 1 - ret[1])
            if (1 - ret[1]) > best_probability:
                best_configuration = ret[0]
                best_probability = 1 - ret[1]
                number_calls_made = ret[2]
                best_optimizer_algorithm = optimizer_algorithm

        # Print results
        print("Final Best Optimizer Algorithm: ", best_optimizer_algorithm)
        print("Final Best Average Probability:", best_probability)
        print("Number of cost function calls made:", number_calls_made)
        print("Parameters Found: " + u"\u03B8" + " = " + str(int(math.degrees(best_configuration[0]))) + u"\u00B0" +
              ", " + u"\u03D5" + "rx = " + str(int(math.degrees(best_configuration[1]))) + u"\u00B0" +
              ", " + u"\u03D5" + "ry = " + str(int(math.degrees(best_configuration[2]))) + u"\u00B0" +
              ", " + u"\u03B7" + u"\u2080" + " = " + str(int(math.degrees(self._global_eta_pair[0]))) + u"\u00B0" +
              ", " + u"\u03B7" + u"\u2081" + " = " + str(int(math.degrees(self._global_eta_pair[1]))) + u"\u00B0")

        return {'best_algorithm': best_optimizer_algorithm,
                'best_probability': best_probability,
                'best_configuration': self._convert_optimizer_results_to_channel_configuration(best_configuration,
                                                                                               self._global_eta_pair),
                'number_calls_made': number_calls_made}

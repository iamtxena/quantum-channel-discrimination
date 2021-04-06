from abc import ABC, abstractmethod
from qcd.configurations.oneshotbaseconfiguration import OneShotConfiguration
from qcd.circuits.aux import check_value, set_random_eta
from qcd.configurations.configuration import ChannelConfiguration
from qcd.backends import DeviceBackend, SimulatorBackend
from typing import Optional, Tuple, cast, List, Dict
from ..typings import GuessStrategy
from ..typings.configurations import OptimalConfigurations
from .aux import set_only_eta_pairs, fix_configurations
import numpy as np
import time
from qiskit.aqua.components.optimizers import SLSQP, L_BFGS_B, ADAM, CRS, DIRECT_L, DIRECT_L_RAND, ESCH, ISRES


class Circuit(ABC):
    """ Generic class acting as an interface for any Quantum Damping Channel Circuit"""

    def __init__(self,
                 optimal_configurations: Optional[OptimalConfigurations] = None,
                 backend: Optional[DeviceBackend] = SimulatorBackend()):

        if optimal_configurations is not None:
            """ !!! THIS IS ANOTHER FIX: to be able to read legacy format configuration """
            self._optimal_configurations = set_only_eta_pairs(cast(List[Dict], [optimal_configurations])).pop()
            """ !!! THIS IS A FIX: to be able to read legacy format configuration """
            self._optimal_configurations = fix_configurations(optimal_configurations)
            # print(self._optimal_configurations)
        self._backend = backend

    def one_shot_run(self, plays: Optional[int] = 100) -> OptimalConfigurations:
        """ Runs all the experiments using the optimal configurations and computing the success probability """
        if self._optimal_configurations is None:
            raise ValueError('Optimal Configurations must be provided on Circuit Constructor to run experiments')
        total_configurations = len(self._optimal_configurations['configurations'])
        optimal_results: OptimalConfigurations = {'eta_pairs': self._optimal_configurations['eta_pairs'],
                                                  'best_algorithm': self._optimal_configurations['best_algorithm'],
                                                  'probabilities': [0] * total_configurations,
                                                  'configurations': self._optimal_configurations['configurations'],
                                                  'number_calls_made': [1] * total_configurations,
                                                  'legacy': (True if
                                                             ('legacy' in self._optimal_configurations and
                                                              self._optimal_configurations['legacy'] is True)
                                                             else False)}
        program_start_time = time.time()
        print(f"Starting the computation for {total_configurations} configurations.")
        for idx, configuration in enumerate(self._optimal_configurations['configurations']):
            optimal_results['probabilities'][idx] = self.compute_average_success_probability(
                configuration, plays)
            # optimal_results['probabilities'][idx] = self._optimization_one_iteration(
            #     cast(OneShotConfiguration, configuration))
            end_time = time.time()
            if idx % 30 == 0 and (end_time - program_start_time <= 60):
                print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                      f'{np.round((end_time - program_start_time), 0)} seconds')
            if idx % 30 == 0 and (end_time - program_start_time > 60):
                print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                      f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
                      f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
            if idx % 30 == 0:
                print(configuration.to_dict())
                print(f"global eta_pair: {self._optimal_configurations['eta_pairs'][idx]}")
                print(f"Configuration index: {idx}, Probabilities ->  computed: " +
                      f"{optimal_results['probabilities'][idx]}, " +
                      f"optimized: {self._optimal_configurations['probabilities'][idx]} and " +
                      "Delta: ",
                      np.round(optimal_results['probabilities'][idx] -
                               self._optimal_configurations['probabilities'][idx], 2))

        end_time = time.time()
        print("total minutes of execution time: " +
              f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
              f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
        print(f"Probabilities from optimization: {self._optimal_configurations['probabilities']}")
        print(f"Probabilities computed: {optimal_results['probabilities']}")
        delta_probabilities = np.array(
            self._optimal_configurations['probabilities']) - np.array(optimal_results['probabilities'])
        print(f"Delta probabilities (Computed - Optimized):\n {delta_probabilities}")
        self._optimal_results = optimal_results
        return optimal_results

    def _optimization_one_iteration(self, configuration: OneShotConfiguration) -> float:
        optimizer_algorithm = 'SLSQP'
        max_evals = 1
        # print("Analyzing Optimizer Algorithm: ", optimizer_algorithm)
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

        self._global_eta_pair = configuration.eta_pair
        initial_parameters = [
            configuration.state_probability,
            configuration.angle_rx,
            configuration.angle_ry,
        ]
        variable_bounds = [
            (0, 1),
            (0, 2 * np.pi),
            (0, 2 * np.pi)
        ]

        ret = optimizer.optimize(num_vars=3,
                                 objective_function=self._cost_function,
                                 variable_bounds=variable_bounds,
                                 initial_point=initial_parameters)
        # print("Best Average Probability:", 1 - ret[1])
        return ret[1]

    @abstractmethod
    def _cost_function(self, params: List[float]) -> float:
        pass

    def compute_average_success_probability(self,
                                            configuration=ChannelConfiguration,
                                            plays: Optional[int] = 100) -> float:
        """ Computes the average success probability of running a specific configuration for the number of plays
            defined in the configuration.
        """

        if plays is None:
            plays = 100
        success_counts = 0
        for play in range(plays):
            success_counts += self._play_and_guess_one_case(configuration)

        return (success_counts / plays)

    def _play_and_guess_one_case(self, channel_configuration: ChannelConfiguration) -> int:
        """ Execute a real execution with a random eta from the two passed,
            guess which one was used on the execution and
            check the result.
            Returns 1 on success (it was a correct guess) or 0 on fail (it was an incorrect guess)
        """
        eta_pair_index_to_use = set_random_eta(channel_configuration.eta_pair)
        eta_pair_index_guessed = self._compute_damping_channel(channel_configuration, eta_pair_index_to_use)

        return check_value(eta_pair_index_to_use, eta_pair_index_guessed)

    @ abstractmethod
    def _prepare_initial_state(self, state_probability: float) -> Tuple[complex, complex]:
        """ Prepare initial state """
        pass

    @ abstractmethod
    def _compute_damping_channel(self,
                                 channel_configuration: ChannelConfiguration,
                                 eta_pair_index: int) -> int:
        pass

    @ abstractmethod
    def _convert_counts_to_eta_used(self,
                                    counts_dict: dict,
                                    guess_strategy: GuessStrategy) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        pass

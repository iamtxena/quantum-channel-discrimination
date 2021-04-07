from abc import ABC, abstractmethod
from qcd.optimizations.aux import reorder_pair
from qcd.circuits.aux import check_value, set_random_eta
from qcd.configurations.configuration import ChannelConfiguration
from qcd.backends import DeviceBackend, SimulatorBackend
from typing import Optional, Tuple, cast, List, Dict
from ..typings import GuessStrategy
from ..typings.configurations import OptimalConfigurations
from .aux import set_only_eta_pairs, fix_configurations
import numpy as np
import time


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
        self._backend = backend

    def one_shot_run(self, plays: Optional[int] = 100) -> OptimalConfigurations:
        """ Runs all the experiments using the optimal configurations and computing the success probability """
        if self._optimal_configurations is None:
            raise ValueError('Optimal Configurations must be provided on Circuit Constructor to run experiments')
        total_configurations = len(self._optimal_configurations['configurations'])
        optimal_results: OptimalConfigurations = {
            'eta_pairs': self._optimal_configurations['eta_pairs'],
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

            end_time = time.time()
            if idx % 30 == 0 and (end_time - program_start_time <= 60):
                print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                      f'{np.round((end_time - program_start_time), 0)} seconds')
            if idx % 30 == 0 and (end_time - program_start_time > 60):
                print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                      f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
                      f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
            if idx % 30 == 0:
                print('computed ',
                      (self._create_one_configuration(configuration, reorder_pair(configuration.eta_pair)).to_dict()))
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
        self._optimal_results = optimal_results
        return optimal_results

    def compute_average_success_probability(self,
                                            configuration: ChannelConfiguration,
                                            plays: Optional[int] = 100) -> float:
        """ Computes the average success probability of running a specific configuration for the number of plays
            defined in the configuration.
        """

        if plays is None:
            plays = 100

        reordered_configuration = self._create_one_configuration(
            configuration,
            reorder_pair(configuration.eta_pair))

        success_counts = 0
        for play in range(plays):
            success_counts += self._play_and_guess_one_case(reordered_configuration)

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

    @abstractmethod
    def _create_one_configuration(self, configuration: ChannelConfiguration,
                                  eta_pair: Tuple[float, float]) -> ChannelConfiguration:
        """ Creates a specific configuration setting a specific eta pair """
        pass

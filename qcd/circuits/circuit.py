from abc import ABC, abstractmethod
from qcd.circuits.aux import check_value, set_random_eta
from qcd.configurations.configuration import ChannelConfiguration
from qcd.backends import DeviceBackend, SimulatorBackend
from typing import Optional, Tuple
from ..typings import GuessStrategy
from ..typings.configurations import OptimalConfigurations
from ..configurations import OneShotConfiguration


class Circuit(ABC):
    """ Generic class acting as an interface for any Quantum Damping Channel Circuit"""

    def __init__(self,
                 optimal_configurations: Optional[OptimalConfigurations] = None,
                 backend: Optional[DeviceBackend] = SimulatorBackend()):
        self._optimal_configurations = optimal_configurations
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
                                                  'number_calls_made': [1] * total_configurations}
        for idx, configuration in enumerate(self._optimal_configurations['configurations']):
            self._optimal_configurations['probabilities'][idx] = self.compute_average_success_probability(
                configuration, plays)
        return optimal_results

    def compute_average_success_probability(self,
                                            configuration=OneShotConfiguration,
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

    @abstractmethod
    def _prepare_initial_state(self, state_probability: float) -> Tuple[complex, complex]:
        """ Prepare initial state """
        pass

    @abstractmethod
    def _compute_damping_channel(self,
                                 channel_configuration: ChannelConfiguration,
                                 eta_pair_index: int) -> int:
        pass

    @abstractmethod
    def _convert_counts_to_eta_used(self,
                                    counts_dict: dict,
                                    guess_strategy: GuessStrategy) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        pass

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.compiler import assemble, transpile
from qiskit.qobj import Qobj
from qcd.optimizations.aux import reorder_pair
from qcd.configurations.configuration import ChannelConfiguration
from qcd.backends import DeviceBackend, SimulatorBackend
from typing import Optional, Tuple, cast, List, Dict
from ..typings.configurations import OptimalConfigurations
from .aux import set_only_eta_pairs, fix_configurations
import numpy as np
import time
from bitarray.util import urandom, count_xor
from bitarray import frozenbitarray


class Circuit(ABC):
    """ Generic class acting as an interface for any Quantum Damping Channel Circuit"""

    def __init__(self,
                 optimal_configurations: Optional[OptimalConfigurations] = None,
                 backend: Optional[DeviceBackend] = SimulatorBackend()):
        self._backend = SimulatorBackend() if backend is None else backend

        if optimal_configurations is not None:
            """ !!! THIS IS ANOTHER FIX: to be able to read legacy format configuration """
            self._optimal_configurations = set_only_eta_pairs(cast(List[Dict], [optimal_configurations])).pop()
            """ !!! THIS IS A FIX: to be able to read legacy format configuration """
            self._optimal_configurations = fix_configurations(optimal_configurations)

    def one_shot_run(self, plays: Optional[int] = 100) -> OptimalConfigurations:
        """ Runs all the experiments using the optimal configurations and computing the success probability """
        if self._optimal_configurations is None:
            raise ValueError('Optimal Configurations must be provided on Circuit Constructor to run experiments')

        total_configurations = len(self._optimal_configurations['configurations'])
        optimal_results = self.init_optimal_results(total_configurations)
        program_start_time = time.time()

        print(f"Starting the computation for {total_configurations} configurations.")
        for idx, configuration in enumerate(self._optimal_configurations['configurations']):
            optimal_results['probabilities'][idx] = self.compute_average_success_probability(
                configuration=configuration,
                plays=plays)

            self.print_computation_time(total_configurations, optimal_results, program_start_time, idx, configuration)

        end_time = time.time()
        print("total minutes of execution time: " +
              f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
              f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
        self._optimal_results = optimal_results
        return optimal_results

    def init_optimal_results(self, total_configurations):
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
        return optimal_results

    def print_computation_time(self,
                               total_configurations,
                               optimal_results,
                               program_start_time,
                               idx,
                               configuration):
        end_time = time.time()
        if idx % 10 == 0 and (end_time - program_start_time <= 60):
            print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                  f'{np.round((end_time - program_start_time), 0)} seconds')
        if idx % 10 == 0 and (end_time - program_start_time > 60):
            print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                  f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
                  f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
        if idx % 10 == 0:
            print('computed ',
                  (self._create_one_configuration(configuration, reorder_pair(configuration.eta_pair)).to_dict()))
            print(f"Configuration index: {idx}, Probabilities ->  computed: " +
                  f"{optimal_results['probabilities'][idx]}, " +
                  f"optimized: {self._optimal_configurations['probabilities'][idx]} and " +
                  "Delta: ",
                  np.round(optimal_results['probabilities'][idx] -
                           self._optimal_configurations['probabilities'][idx], 2))

    def compute_average_success_probability(self,
                                            configuration: ChannelConfiguration,
                                            plays: Optional[int] = 100,) -> float:
        """ Computes the average success probability of running a specific configuration
            for the number of plays defined in the configuration.
        """
        if plays is None:
            plays = 100

        random_etas = frozenbitarray(urandom(plays))  # creates a bitstring of plays length
        circuit_pair = self._create_transpiled_circuit_pair(configuration)
        guesses = frozenbitarray([self._run_circuit_and_return_gues(circuit_pair[random_eta])
                                  for random_eta in random_etas])
        # count the number of total correct guesses and compute the average success probability
        return 1 - (count_xor(random_etas, guesses) / plays)

    def _run_circuit_and_return_gues(self, obj: Qobj) -> int:
        """ ONE SHOT run for all given circuits and for each resulting counts, guess the eta used """
        counts = self._backend.backend.run(obj).result().get_counts()
        return self._convert_counts_to_eta_used(counts)

    def _create_transpiled_circuit_pair(self,
                                        configuration: ChannelConfiguration) -> Tuple[Qobj,
                                                                                      Qobj]:
        """ Create a pair of Quantum Circuits, in its transpiled form, from a given configuration """
        reordered_configuration = self._reorder_configuration(configuration)
        self._backend = SimulatorBackend() if self._backend is None else self._backend
        experiments = [self._create_transpiled_circuit(reordered_configuration, self._backend, eta)
                       for eta in reordered_configuration.eta_pair]
        if len(experiments) > 2:
            raise ValueError('Transpiled circuits must have length 2')
        return (experiments[0], experiments[1])

    def _create_transpiled_circuit(self, reordered_configuration, backend, eta) -> Qobj:
        one_circuit = self._create_one_circuit(reordered_configuration, eta)
        transpiled_circuit = transpile(one_circuit, backend=backend.backend)
        return assemble(transpiled_circuit, backend=self._backend.backend, shots=1)

    def _reorder_configuration(self, configuration):
        return self._create_one_configuration(
            configuration,
            reorder_pair(configuration.eta_pair))

    @ abstractmethod
    def _prepare_initial_state(self, state_probability: float) -> Tuple[complex, complex]:
        """ Prepare initial state """
        pass

    @abstractmethod
    def _convert_counts_to_eta_used(self, counts_dict: dict) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        pass

    @ abstractmethod
    def _convert_all_counts_to_all_eta_used(self,
                                            counts_all_circuits: List[dict]) -> List[int]:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        pass

    @ abstractmethod
    def _create_one_configuration(self, configuration: ChannelConfiguration,
                                  eta_pair: Tuple[float, float]) -> ChannelConfiguration:
        """ Creates a specific configuration setting a specific eta pair """
        pass

    @ abstractmethod
    def _create_one_circuit(self,
                            configuration: ChannelConfiguration,
                            eta: float) -> QuantumCircuit:
        """ Creates one circuit from a given  configuration and eta """
        pass

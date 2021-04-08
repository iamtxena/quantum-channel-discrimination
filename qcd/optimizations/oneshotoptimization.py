from . import Optimization
from ..typings import CloneSetup, OptimizationSetup
from ..typings.configurations import OptimalConfigurations
from typing import Optional, Tuple, cast, List, Dict
from ..configurations import ChannelConfiguration, OneShotConfiguration
from ..circuits import OneShotCircuit
import time
import numpy as np
import math


class OneShotOptimization(Optimization):
    """ Representation of the One Shot Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)
        self._one_shot_circuit = OneShotCircuit()

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_pair: Tuple[float, float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotConfiguration({
            'state_probability': configuration[0],
            'angle_rx': configuration[1],
            'angle_ry': configuration[2],
            'eta_pair': eta_pair})

    def _cost_function(self, params: List[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
            defined in the optimization setup.
            Cost is computed as 1 (perfect probability) - average success probability for
            all the plays with the given configuration
            Returns the Cost (error probability).
        """
        configuration = OneShotConfiguration({
            'state_probability': params[0],
            'angle_rx': params[1],
            'angle_ry': params[2],
            'eta_pair': self._global_eta_pair})

        return 1 - self._one_shot_circuit.compute_new_average_success_probability(configuration=configuration,
                                                                                  plays=self._setup['plays'])

    def find_optimal_configurations(self, clone_setup: Optional[CloneSetup]) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        eta_pairs_idx_to_optimize, optimal_configurations = self._select_eta_pairs_to_optimize(clone_setup)

        print(f'number of eta_pairs_idx_to_optimize: {len(eta_pairs_idx_to_optimize)} -> {eta_pairs_idx_to_optimize}')

        program_start_time = time.time()
        print("Starting the execution")

        for eta_pair_idx in eta_pairs_idx_to_optimize:
            start_time = time.time()
            self._global_eta_pair = self._eta_pairs[eta_pair_idx]
            result = self._compute_best_configuration()
            optimal_configurations['probabilities'].append(result['best_probability'])
            optimal_configurations['configurations'].append(result['best_configuration'])
            optimal_configurations['best_algorithm'].append(result['best_algorithm'])
            optimal_configurations['number_calls_made'].append(result['number_calls_made'])
            end_time = time.time()
            print(f"Pair of etas # {eta_pair_idx} of {len(eta_pairs_idx_to_optimize)}, time taken this pair of etas: " +
                  f'{np.round((end_time - start_time)/60, 0)} minutes' +
                  f' and {np.round((end_time - start_time) % 60, 0)} seconds')
            print("total minutes taken this pair of etas: ", int(np.round((end_time - start_time) / 60)))
            print("total time taken so far: " +
                  f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
                  f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
            remaining_time = (len(eta_pairs_idx_to_optimize) * (end_time -
                                                                program_start_time)) / (eta_pair_idx + 1)
            remaining_days = np.round(remaining_time / 60 / 60 / 24, 0)
            remaining_hours = np.round((remaining_time % 60 / 60 / 24) * 24, 0)
            float_minutes = (math.modf((remaining_time % 60 / 60 / 24) * 24)[0]) * 60
            remaining_minutes = np.round(float_minutes, 0)
            float_seconds = (math.modf(float_minutes)[0]) * 60
            remaining_seconds = np.round(float_seconds, 0)
            print(f"estimated remaining time: {remaining_days} days, {remaining_hours} hours, " +
                  f" {remaining_minutes} minutes and {remaining_seconds}")

        end_time = time.time()
        print("total minutes of execution time: ", int(np.round((end_time - program_start_time) / 60)))
        print(f'Number eta pairs optimized: {len(eta_pairs_idx_to_optimize)}' +
              f'from the total eta pairs: {len(self._eta_pairs)} ')
        optimal_configurations['eta_pairs'] = self._eta_pairs

        return optimal_configurations

    def _select_eta_pairs_to_optimize(self, clone_setup: Optional[CloneSetup]) -> Tuple[List[int],
                                                                                        OptimalConfigurations]:
        """ from the given clone setup, select the eta pairs to be optimized,
            and set the non computed pairs configuration as default values   """
        eta_pair_idx_init, eta_pair_idx_end = self._set_eta_pair_index_bounds(clone_setup)
        index_dict = self._build_eta_pair_index_lists(eta_pair_idx_init, eta_pair_idx_end)
        default_optimal_configurations = self._set_default_optimal_configurations(index_dict['eta_pair_idx_to_skip'])

        return (index_dict['eta_pair_idx_to_compute'],
                default_optimal_configurations)

    def _set_default_optimal_configurations(self, eta_pair_idx_to_skip: List[int]) -> OptimalConfigurations:
        """ Return the optimal configurations set to default values for the indexes to be skipped """
        elements_to_skip = len(eta_pair_idx_to_skip)

        configurations: List[ChannelConfiguration] = []
        for eta_pair_idx in eta_pair_idx_to_skip:
            one_configuration = OneShotConfiguration({'state_probability': 0,
                                                      'angle_rx': 0,
                                                      'angle_ry': 0,
                                                      'eta_pair': self._eta_pairs[eta_pair_idx]})
            configurations.append(cast(ChannelConfiguration, one_configuration))

        return {'eta_pairs': [],
                'best_algorithm': ['NA'] * elements_to_skip,
                'probabilities': [0] * elements_to_skip,
                'configurations': configurations,
                'number_calls_made': [0] * elements_to_skip}

    def _set_eta_pair_index_bounds(self, clone_setup: Optional[CloneSetup]) -> Tuple[int, int]:
        """ set the first and last eta pair index from which to optimize the configuration """
        total_eta_pairs = len(self._eta_pairs)

        if clone_setup is None or clone_setup['total_clones'] <= 1:
            return (0, total_eta_pairs)

        eta_pair_idx_init = int(np.floor(clone_setup['id_clone'] * total_eta_pairs / clone_setup['total_clones']))
        eta_pair_idx_end = min(int((clone_setup['id_clone'] + 1) *
                                   total_eta_pairs / clone_setup['total_clones']), total_eta_pairs)
        return (eta_pair_idx_init, eta_pair_idx_end)

    def _build_eta_pair_index_lists(self, eta_pair_idx_init: int, eta_pair_idx_end: int) -> Dict:
        """ create two lists with the the eta pair index to be computed and the index to be skipped """
        first_part_to_skip = list(range(0, eta_pair_idx_init))
        last_part_to_skip = list(range(eta_pair_idx_end, len(self._eta_pairs)))

        return {
            'eta_pair_idx_to_compute': list(range(eta_pair_idx_init, eta_pair_idx_end)),
            'eta_pair_idx_to_skip': first_part_to_skip + last_part_to_skip
        }

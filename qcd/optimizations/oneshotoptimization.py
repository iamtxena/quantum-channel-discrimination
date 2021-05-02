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
        self._add_initial_parameters_and_variable_bounds_to_optimization_setup()
        self._one_shot_circuit = OneShotCircuit()

    def _add_initial_parameters_and_variable_bounds_to_optimization_setup(self) -> None:
        """ Update the optimization setup with intial parameters and variable bounds """
        self._setup['initial_parameters'] = [0] * 3
        variable_bounds = [(0, 1)]  # amplitude_probability
        variable_bounds += [(0, 2 * np.pi)
                            for i in range(2)]
        self._setup['variable_bounds'] = cast(List[Tuple[float, float]], variable_bounds)

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_group: List[float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotConfiguration({
            'state_probability': configuration[0],
            'angle_rx': configuration[1],
            'angle_ry': configuration[2],
            'eta_group': eta_group})

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
            'eta_group': self._global_eta_group})

        return - self._one_shot_circuit.compute_average_success_probability(configuration=configuration,
                                                                            plays=self._setup['plays'])

    def find_optimal_configurations(self, clone_setup: Optional[CloneSetup]) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
            using the configured optimization algorithm """
        eta_groups_idx_to_optimize, optimal_configurations = self._select_eta_groups_to_optimize(clone_setup)

        print(
            f'number of eta_groups_idx_to_optimize: {len(eta_groups_idx_to_optimize)} -> {eta_groups_idx_to_optimize}')

        program_start_time = time.time()
        print("Starting the execution")

        for eta_group_idx in eta_groups_idx_to_optimize:
            start_time = time.time()
            self._global_eta_group = self._eta_groups[eta_group_idx]
            result = self._compute_best_configuration()
            optimal_configurations['probabilities'].append(result['best_probability'])
            optimal_configurations['configurations'].append(result['best_configuration'])
            optimal_configurations['best_algorithm'].append(result['best_algorithm'])
            optimal_configurations['number_calls_made'].append(result['number_calls_made'])
            end_time = time.time()
            print(f"Group of etas # {eta_group_idx} of {len(eta_groups_idx_to_optimize)}, " +
                  "time taken this group of etas: " +
                  f'{np.round((end_time - start_time)/60, 0)} minutes' +
                  f' and {np.round((end_time - start_time) % 60, 0)} seconds')
            print(f"total minutes taken this group of etas: {np.round((end_time - start_time) / 60)}")
            print("total time taken so far: " +
                  f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
                  f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
            remaining_time = (len(eta_groups_idx_to_optimize) * (end_time -
                                                                 program_start_time)) / (eta_group_idx + 1)
            remaining_days = np.round(remaining_time / 60 / 60 / 24, 0)
            remaining_hours = np.round((remaining_time % 60 / 60 / 24) * 24, 0)
            float_minutes = (math.modf((remaining_time % 60 / 60 / 24) * 24)[0]) * 60
            remaining_minutes = np.round(float_minutes, 0)
            float_seconds = (math.modf(float_minutes)[0]) * 60
            remaining_seconds = np.round(float_seconds, 0)
            print(f"estimated remaining time: {remaining_days} days, {remaining_hours} hours, " +
                  f" {remaining_minutes} minutes and {remaining_seconds}")

        end_time = time.time()
        print(f"total minutes of execution time: {np.round((end_time - program_start_time) / 60)}")
        print(f'Number eta groups optimized: {len(eta_groups_idx_to_optimize)}' +
              f'from the total eta groups: {len(self._eta_groups)} ')
        optimal_configurations['eta_groups'] = self._eta_groups

        return optimal_configurations

    def _select_eta_groups_to_optimize(self, clone_setup: Optional[CloneSetup]) -> Tuple[List[int],
                                                                                         OptimalConfigurations]:
        """ from the given clone setup, select the eta groups to be optimized,
            and set the non computed pairs configuration as default values   """
        eta_group_idx_init, eta_group_idx_end = self._set_eta_group_index_bounds(clone_setup)
        index_dict = self._build_eta_group_index_lists(eta_group_idx_init, eta_group_idx_end)
        default_optimal_configurations = self._set_default_optimal_configurations(
            index_dict['eta_group_idx_to_skip'])

        return (index_dict['eta_group_idx_to_compute'],
                default_optimal_configurations)

    def _set_default_optimal_configurations(self, eta_group_idx_to_skip: List[int]) -> OptimalConfigurations:
        """ Return the optimal configurations set to default values for the indexes to be skipped """
        elements_to_skip = len(eta_group_idx_to_skip)

        configurations: List[ChannelConfiguration] = []
        for eta_group_idx in eta_group_idx_to_skip:
            one_configuration = OneShotConfiguration({'state_probability': 0,
                                                      'angle_rx': 0,
                                                      'angle_ry': 0,
                                                      'eta_group': self._eta_groups[eta_group_idx]})
            configurations.append(cast(ChannelConfiguration, one_configuration))

        return {'eta_groups': [],
                'best_algorithm': ['NA'] * elements_to_skip,
                'probabilities': [0] * elements_to_skip,
                'configurations': configurations,
                'number_calls_made': [0] * elements_to_skip}

    def _set_eta_group_index_bounds(self, clone_setup: Optional[CloneSetup]) -> Tuple[int, int]:
        """ set the first and last eta pair index from which to optimize the configuration """
        total_eta_groups = len(self._eta_groups)

        if clone_setup is None or clone_setup['total_clones'] <= 1:
            return (0, total_eta_groups)

        eta_group_idx_init = int(np.floor(clone_setup['id_clone'] * total_eta_groups / clone_setup['total_clones']))
        eta_group_idx_end = min(int((clone_setup['id_clone'] + 1) *
                                    total_eta_groups / clone_setup['total_clones']), total_eta_groups)
        return (eta_group_idx_init, eta_group_idx_end)

    def _build_eta_group_index_lists(self, eta_group_idx_init: int, eta_group_idx_end: int) -> Dict:
        """ create two lists with the the eta pair index to be computed and the index to be skipped """
        first_part_to_skip = list(range(0, eta_group_idx_init))
        last_part_to_skip = list(range(eta_group_idx_end, len(self._eta_groups)))

        return {
            'eta_group_idx_to_compute': list(range(eta_group_idx_init, eta_group_idx_end)),
            'eta_group_idx_to_skip': first_part_to_skip + last_part_to_skip
        }

    def _best_configuration_to_print(self, best_configuration: List[float]) -> str:
        return ("Parameters Found: state_probability = " + " = " + str(best_configuration[0]) +
                ", " + u"\u03D5" + "rx = " + str(int(math.degrees(best_configuration[1]))) + u"\u00B0" +
                ", " + u"\u03D5" + "ry = " + str(int(math.degrees(best_configuration[2]))) + u"\u00B0" +
                ", " + u"\u03B7" + u"\u2080" + " = " + str(int(math.degrees(self._global_eta_group[0]))) + u"\u00B0" +
                ", " + u"\u03B7" + u"\u2081" + " = " + str(int(math.degrees(self._global_eta_group[1]))) + u"\u00B0")

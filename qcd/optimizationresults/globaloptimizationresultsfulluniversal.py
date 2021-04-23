from abc import ABC
from qcd.dampingchannels.oneshotentangledfulluniversal import OneShotEntangledFullUniversalDampingChannel
from typing import Optional
from ..typings.configurations import OptimalConfigurations
from .aux import load_result_from_file
import time
import math
import numpy as np


class GlobalOptimizationResultsFullUniversal(ABC):
    """ Global class to load, process and plot any Optimization Results """

    @staticmethod
    def load_results(file_name: str, path: Optional[str] = ""):
        """
          1. Load results from file for all file names
          2. build probability matrices for each loaded result
          3. build amplitude matrix for each loaded result
        """

        results = load_result_from_file(file_name, path)
        return GlobalOptimizationResultsFullUniversal(results)

    def __init__(self, optimal_configurations: OptimalConfigurations) -> None:
        self._optimal_configurations = optimal_configurations

    @property
    def optimal_configurations(self):
        return self._optimal_configurations

    @property
    def validated_optimal_configurations(self):
        return self._validated_optimal_configurations

    def validate_optimal_configurations(self,
                                        plays: Optional[int] = 10000) -> None:
        """ Runs the circuit with the given optimal configurations computing the success average probability
            for each eta (and also the global), the selected eta for each measured state and finally the
            upper and lower bound fidelities
        """
        validated_configurations = self._optimal_configurations
        validated_configurations['validated_probabilities'] = []
        validated_configurations['eta_probabilities'] = []
        validated_configurations['measured_states_eta_assignment'] = []
        validated_configurations['fidelities'] = []

        eta_groups_length = len(validated_configurations['eta_groups'])
        print(f'number of eta groups to validate: {eta_groups_length}')

        program_start_time = time.time()
        probability_diffs_list = []
        for idx, configuration in enumerate(validated_configurations['configurations']):
            eta_group = self._optimal_configurations['eta_groups'][idx]
            validated_configuration = OneShotEntangledFullUniversalDampingChannel.validate_optimal_configuration(
                configuration, plays
            )
            probability_diffs = np.round((self._optimal_configurations['probabilities']
                                          [idx] - validated_configuration['validated_probability']) * 100, 2)
            probability_diffs_list.append(probability_diffs)
            end_time = time.time()
            if idx % 20 == 0:
                print(
                    f'Going to validate this eta group: ({int(math.degrees(eta_group[0]))}, ' +
                    f'{int(math.degrees(eta_group[1]))}, {int(math.degrees(eta_group[2]))})')
                print(f"Optimal Probability: {np.round(validated_configurations['probabilities'][idx]*100, 2)}% " +
                      f"Validated Probability: {np.round(validated_configuration['validated_probability']*100, 2)} % " +
                      f'Difference (absolute value): {probability_diffs}%')
                print(f"Group of etas # {idx} of {eta_groups_length}")
                print("total time taken so far: " +
                      f'{np.round(math.floor((end_time - program_start_time)/60), 0)} minutes' +
                      f' and {int((end_time - program_start_time) % 60)} seconds')
            validated_configurations['validated_probabilities'].append(validated_configuration['validated_probability'])
            validated_configurations['eta_probabilities'].append(validated_configuration['etas_probability'])
            validated_configurations['measured_states_eta_assignment'].append(
                validated_configuration['measured_states_eta_assignment'])
            validated_configurations['fidelities'].append(validated_configuration['fidelities'])
        self._validated_optimal_configurations = validated_configurations
        end_time = time.time()
        total_minutes = int((end_time - program_start_time) / 60)
        if total_minutes < 1:
            print(f"total seconds of validation: {int(end_time - program_start_time)}")
        if total_minutes >= 1:
            print(f"total minutes of validation: {total_minutes}")
        print(f'Probability differences --> MAX: {max(probability_diffs_list)}% , MIN: {min(probability_diffs_list)}%')

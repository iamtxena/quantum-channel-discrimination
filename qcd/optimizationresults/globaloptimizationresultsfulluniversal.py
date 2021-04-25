from abc import ABC
from qcd.typings.dicts import ResultsToPlot
from qcd.dampingchannels import OneShotEntangledFullUniversalDampingChannel
from qcd.configurations import OneShotEntangledFullUniversalConfiguration
from typing import List, Literal, Optional, Union, cast
from ..typings.configurations import MeasuredStatesEtaAssignment, OptimalConfigurations
from .aux import load_result_from_file, get_number_eta_pairs
import time
import math
import numpy as np
import matplotlib.pyplot as plt


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

    @property
    def results_to_plot(self):
        return self._results_to_plot

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
        validated_configurations['measured_states_counts'] = []

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
            validated_configurations['measured_states_counts'].append(validated_configuration['measured_states_counts'])
        self._validated_optimal_configurations = validated_configurations
        end_time = time.time()
        total_minutes = int((end_time - program_start_time) / 60)
        if total_minutes < 1:
            print(f"total seconds of validation: {int(end_time - program_start_time)}")
        if total_minutes >= 1:
            print(f"total minutes of validation: {total_minutes}")
        print(f'Probability differences --> MAX: {max(probability_diffs_list)}% , MIN: {min(probability_diffs_list)}%')
        print("Preparing results to plot...")
        self._prepare_results_to_plot()
        print("Results ready to be displayed. 😎 ")

    def _prepare_results_to_plot(self) -> None:
        """ prepare data structures to be easily plotted """
        error_probabilities = []
        error_probabilities_validated = []
        etas_third_channel = []
        upper_fidelities = []
        lower_fidelities = []
        eta0_success_probabilities = []
        eta1_success_probabilities = []
        eta2_success_probabilities = []
        eta_assigned_state_00 = []
        eta_assigned_state_01 = []
        eta_assigned_state_10 = []
        eta_assigned_state_11 = []
        success_probabilities_validated = []
        counts_00_eta0: List[str] = []
        counts_00_eta1: List[str] = []
        counts_00_eta2: List[str] = []
        counts_01_eta0: List[str] = []
        counts_01_eta1: List[str] = []
        counts_01_eta2: List[str] = []
        counts_10_eta0: List[str] = []
        counts_10_eta1: List[str] = []
        counts_10_eta2: List[str] = []
        counts_11_eta0: List[str] = []
        counts_11_eta1: List[str] = []
        counts_11_eta2: List[str] = []
        total_counts = []

        for idx, configuration in enumerate(self._validated_optimal_configurations['configurations']):
            etas_third_channel.append(
                int(math.degrees(cast(OneShotEntangledFullUniversalConfiguration, configuration).eta_group[2])))
            error_probabilities.append(1 - self._validated_optimal_configurations['probabilities'][idx])
            error_probabilities_validated.append(
                1 - self._validated_optimal_configurations['validated_probabilities'][idx])
            success_probabilities_validated.append(
                self._validated_optimal_configurations['validated_probabilities'][idx])
            upper_fidelities.append(self._validated_optimal_configurations['fidelities'][idx]['upper_bound_fidelity'])
            lower_fidelities.append(self._validated_optimal_configurations['fidelities'][idx]['lower_bound_fidelity'])
            eta0_success_probabilities.append(self._validated_optimal_configurations['eta_probabilities'][idx][0])
            eta1_success_probabilities.append(self._validated_optimal_configurations['eta_probabilities'][idx][1])
            eta2_success_probabilities.append(self._validated_optimal_configurations['eta_probabilities'][idx][2])
            eta_assigned_state_00.append(self._assign_eta(
                'state_00', self._validated_optimal_configurations['measured_states_eta_assignment'][idx]))
            eta_assigned_state_01.append(self._assign_eta(
                'state_01', self._validated_optimal_configurations['measured_states_eta_assignment'][idx]))
            eta_assigned_state_10.append(self._assign_eta(
                'state_10', self._validated_optimal_configurations['measured_states_eta_assignment'][idx]))
            eta_assigned_state_11.append(self._assign_eta(
                'state_11', self._validated_optimal_configurations['measured_states_eta_assignment'][idx]))
            if (np.round(sum(self._validated_optimal_configurations['eta_probabilities'][idx]), 3) !=
                    np.round(self._validated_optimal_configurations['validated_probabilities'][idx], 3)):
                raise ValueError('invalid probabilities!')
            self._assign_counts(counts_00_eta0, counts_00_eta1, counts_00_eta2, counts_01_eta0, counts_01_eta1,
                                counts_01_eta2, counts_10_eta0, counts_10_eta1, counts_10_eta2, counts_11_eta0,
                                counts_11_eta1, counts_11_eta2, idx)
            total_counts.append(self._validated_optimal_configurations['measured_states_counts'][idx]['total_counts'])

        number_eta_pairs, eta_unique_pairs = get_number_eta_pairs(
            eta_groups=self._validated_optimal_configurations['eta_groups'])
        number_third_channels = int(len(self._validated_optimal_configurations['eta_groups']) / number_eta_pairs)

        self._results_to_plot = [ResultsToPlot(
            {'error_probabilities': error_probabilities[idx * number_third_channels: (idx + 1) * number_third_channels],
             'error_probabilities_validated': error_probabilities_validated[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'success_probabilities_validated': success_probabilities_validated[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'etas_third_channel': etas_third_channel[idx * number_third_channels: (idx + 1) * number_third_channels],
             'upper_fidelities': upper_fidelities[idx * number_third_channels: (idx + 1) * number_third_channels],
             'lower_fidelities': lower_fidelities[idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta0_success_probabilities': eta0_success_probabilities[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta1_success_probabilities': eta1_success_probabilities[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta2_success_probabilities': eta2_success_probabilities[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta_assigned_state_00': eta_assigned_state_00[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta_assigned_state_01': eta_assigned_state_01[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta_assigned_state_10': eta_assigned_state_10[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta_assigned_state_11': eta_assigned_state_11[
                idx * number_third_channels: (idx + 1) * number_third_channels],
             'eta_pair': eta_pair,
             'counts_00_eta0': counts_00_eta0[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_00_eta1': counts_00_eta1[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_00_eta2': counts_00_eta2[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_01_eta0': counts_01_eta0[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_01_eta1': counts_01_eta1[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_01_eta2': counts_01_eta2[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_10_eta0': counts_10_eta0[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_10_eta1': counts_10_eta1[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_10_eta2': counts_10_eta2[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_11_eta0': counts_11_eta0[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_11_eta1': counts_11_eta1[idx * number_third_channels: (idx + 1) * number_third_channels],
             'counts_11_eta2': counts_11_eta2[idx * number_third_channels: (idx + 1) * number_third_channels],
             'total_counts': total_counts[idx * number_third_channels: (idx + 1) * number_third_channels]})
            for idx, eta_pair in enumerate(eta_unique_pairs)]

    def _assign_counts(self, counts_00_eta0, counts_00_eta1, counts_00_eta2, counts_01_eta0, counts_01_eta1,
                       counts_01_eta2, counts_10_eta0, counts_10_eta1, counts_10_eta2, counts_11_eta0, counts_11_eta1,
                       counts_11_eta2, idx):

        self._assign_max_counts(self._validated_optimal_configurations['measured_states_counts'][idx]['state_00'][0],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_00'][1],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_00'][2],
                                counts_00_eta0, counts_00_eta1, counts_00_eta2)
        self._assign_max_counts(self._validated_optimal_configurations['measured_states_counts'][idx]['state_01'][0],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_01'][1],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_01'][2],
                                counts_01_eta0, counts_01_eta1, counts_01_eta2)
        self._assign_max_counts(self._validated_optimal_configurations['measured_states_counts'][idx]['state_10'][0],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_10'][1],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_10'][2],
                                counts_10_eta0, counts_10_eta1, counts_10_eta2)
        self._assign_max_counts(self._validated_optimal_configurations['measured_states_counts'][idx]['state_11'][0],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_11'][1],
                                self._validated_optimal_configurations['measured_states_counts'][idx]['state_11'][2],
                                counts_11_eta0, counts_11_eta1, counts_11_eta2)

    def _assign_max_counts(self, counts_eta0: int, counts_eta1: int, counts_eta2: int,
                           counts_state_eta0: List[str], counts_state_eta1: List[str],
                           counts_state_eta2: List[str]) -> None:
        max_counts = counts_eta0
        index_eta = 0
        if (counts_eta1 > max_counts):
            max_counts = counts_eta1
            index_eta = 1
        if (counts_eta2 > max_counts):
            max_counts = counts_eta2
            index_eta = 2
        str_counts_eta0 = ''
        str_counts_eta1 = ''
        str_counts_eta2 = ''
        if index_eta == 0:
            str_counts_eta0 += '**'
        if index_eta == 1:
            str_counts_eta1 += '**'
        if index_eta == 2:
            str_counts_eta2 += '**'
        str_counts_eta0 += str(counts_eta0)
        str_counts_eta1 += str(counts_eta1)
        str_counts_eta2 += str(counts_eta2)
        counts_state_eta0.append(str_counts_eta0)
        counts_state_eta1.append(str_counts_eta1)
        counts_state_eta2.append(str_counts_eta2)

    def _assign_eta(self,
                    state_str: Union[
                        Literal['state_00'], Literal['state_01'], Literal['state_10'], Literal['state_11']],
                    measured_states_eta_assignment: MeasuredStatesEtaAssignment) -> str:
        if measured_states_eta_assignment[state_str] == -1:
            return 'None'
        if measured_states_eta_assignment[state_str] == 0:
            return '$\eta0$'
        if measured_states_eta_assignment[state_str] == 1:
            return '$\eta1$'
        if measured_states_eta_assignment[state_str] == 2:
            return '$\eta2$'
        raise ValueError('Invalid input state')

    def plot_3channel_results(self, algorithm: str = '') -> None:
        if self._results_to_plot is None:
            raise ValueError('Results not available. Please call validate_optimal_configurations first.')

        fig = plt.figure(figsize=(25, 10))
        sup_title = 'Error Probabilities'
        sup_title += f' with {algorithm}' if algorithm != '' else ''
        fig.suptitle(sup_title, fontsize=20)

        for idx, parsed_result in enumerate(self._results_to_plot):
            title = f"$\eta$ pair ({parsed_result['eta_pair'][0]}\u00B0, {parsed_result['eta_pair'][1]}\u00B0)"
            ax = fig.add_subplot(2, 3, idx + 1 % 3)
            ax.set_ylim([0, 1])
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('$\eta_2$ angle')
            ax.set_ylabel('Probability Error')
            ax.axvline(x=parsed_result['eta_pair'][0], linestyle='dotted',
                       color='lightcoral', label=f"$\eta_0$: {parsed_result['eta_pair'][0]}\u00B0")
            ax.axvline(x=parsed_result['eta_pair'][1], linestyle='dotted', color='firebrick',
                       label=f"$\eta_1$: {parsed_result['eta_pair'][1]}\u00B0")
            ax.plot(parsed_result['etas_third_channel'], parsed_result['error_probabilities'], label='Perr')
            ax.plot(parsed_result['etas_third_channel'], parsed_result['upper_fidelities'], label='Upper Bound')
            ax.plot(parsed_result['etas_third_channel'], parsed_result['lower_fidelities'], label='Lower Bound')
            ax.legend()
        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def plot_global_probabilities(self, algorithm: str = '') -> None:
        if self._results_to_plot is None:
            raise ValueError('Results not available. Please call validate_optimal_configurations first.')

        fig = plt.figure(figsize=(25, 10))
        sup_title = 'Error Probabilities: Optimization vs Validation'
        sup_title += f' with {algorithm}' if algorithm != '' else ''
        fig.suptitle(sup_title, fontsize=20)

        for idx, parsed_result in enumerate(self._results_to_plot):
            title = f"$\eta$ pair ({parsed_result['eta_pair'][0]}\u00B0, {parsed_result['eta_pair'][1]}\u00B0)"
            ax = fig.add_subplot(2, 3, idx + 1 % 3)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('$\eta_2$ angle')
            ax.set_ylabel('Probability Error')
            ax.axvline(x=parsed_result['eta_pair'][0], linestyle='dotted',
                       color='lightcoral', label=f"$\eta_0$: {parsed_result['eta_pair'][0]}\u00B0")
            ax.axvline(x=parsed_result['eta_pair'][1], linestyle='dotted', color='firebrick',
                       label=f"$\eta_1$: {parsed_result['eta_pair'][1]}\u00B0")
            ax.plot(parsed_result['etas_third_channel'],
                    parsed_result['error_probabilities'], label='Perr Optimization')
            ax.plot(parsed_result['etas_third_channel'],
                    parsed_result['error_probabilities_validated'], label='Perr Validation')
            ax.legend()
        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def plot_eta_success_probabilities(self, algorithm: str = '') -> None:
        if self._results_to_plot is None:
            raise ValueError('Results not available. Please call validate_optimal_configurations first.')

        fig = plt.figure(figsize=(25, 10))
        sup_title = '$\eta$ Success Probabilities'
        sup_title += f' with {algorithm}' if algorithm != '' else ''
        fig.suptitle(sup_title, fontsize=20)
        width = 2
        for idx, parsed_result in enumerate(self._results_to_plot):
            title = f"$\eta$ pair ({parsed_result['eta_pair'][0]}\u00B0, {parsed_result['eta_pair'][1]}\u00B0)"
            ax = fig.add_subplot(2, 3, idx + 1 % 3)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('$\eta_2$ angle')
            ax.set_ylabel('Probability Success')
            ax.set_ylim([0, 1])
            base_bottom_probs = [eta0_prob + eta1_prob
                                 for eta0_prob, eta1_prob in zip(parsed_result['eta0_success_probabilities'],
                                                                 parsed_result['eta1_success_probabilities'])]
            ax.axvline(x=parsed_result['eta_pair'][0], linestyle='dashed',
                       color='cornflowerblue', label=f"$\eta_0$: {parsed_result['eta_pair'][0]}\u00B0")
            ax.axvline(x=parsed_result['eta_pair'][1], linestyle='dashed', color='darkorange',
                       label=f"$\eta_1$: {parsed_result['eta_pair'][1]}\u00B0")
            ax.plot(parsed_result['etas_third_channel'],
                    parsed_result['success_probabilities_validated'],
                    linestyle='dotted', color='grey', label='Psucc Validation')
            ax.bar(parsed_result['etas_third_channel'],
                   parsed_result['eta0_success_probabilities'], width=width,
                   align='edge', label='Psucc $\eta_0$', color='cornflowerblue')
            ax.bar(parsed_result['etas_third_channel'], parsed_result['eta1_success_probabilities'],
                   width=width, align='edge',
                   bottom=parsed_result['eta0_success_probabilities'],
                   label='Psucc $\eta_1$', color='darkorange')
            ax.bar(parsed_result['etas_third_channel'], parsed_result['eta2_success_probabilities'],
                   width=width, align='edge',
                   bottom=base_bottom_probs,
                   label='Psucc $\eta_2$', color='green')
            ax.legend()
        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def plot_eta_assignments(self, eta_pair_index: int = -1, algorithm: str = '') -> None:
        fig = plt.figure(figsize=(35, 30)) if eta_pair_index < 0 else plt.figure(figsize=(25, 10))
        sup_title = '$\eta$ Success Probabilities with $\eta$ assigments for each measurement state'
        sup_title += f' with {algorithm}' if algorithm != '' else ''
        fig.suptitle(sup_title, fontsize=20)
        width = 3

        for idx, parsed_result in enumerate(self._results_to_plot):
            if eta_pair_index >= 0 and eta_pair_index != idx:
                continue
            title = f"$\eta$ pair ({parsed_result['eta_pair'][0]}\u00B0, {parsed_result['eta_pair'][1]}\u00B0)"
            ax = fig.add_subplot(3, 2, idx + 1 % 2) if eta_pair_index < 0 else fig.add_subplot(111)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel('Probability Success')
            ax.set_ylim([0, 1])
            ax.set(xlabel=None, xticklabels=[])
            base_bottom_probs = [eta0_prob + eta1_prob
                                 for eta0_prob, eta1_prob in zip(parsed_result['eta0_success_probabilities'],
                                                                 parsed_result['eta1_success_probabilities'])]
            ax.axvline(x=parsed_result['eta_pair'][0], linestyle='dashed',
                       color='cornflowerblue', label=f"$\eta_0$: {parsed_result['eta_pair'][0]}\u00B0")
            ax.axvline(x=parsed_result['eta_pair'][1], linestyle='dashed', color='darkorange',
                       label=f"$\eta_1$: {parsed_result['eta_pair'][1]}\u00B0")
            ax.plot(parsed_result['etas_third_channel'],
                    parsed_result['success_probabilities_validated'],
                    linestyle='dotted', color='grey', label='Psucc Validation')
            ax.bar(parsed_result['etas_third_channel'],
                   parsed_result['eta0_success_probabilities'],
                   width=width, align='edge', label='Psucc $\eta_0$', color='cornflowerblue')
            ax.bar(parsed_result['etas_third_channel'], parsed_result['eta1_success_probabilities'],
                   width=width,
                   align='edge',
                   bottom=parsed_result['eta0_success_probabilities'],
                   label='Psucc $\eta_1$', color='darkorange')
            ax.bar(parsed_result['etas_third_channel'], parsed_result['eta2_success_probabilities'],
                   width=width,
                   align='edge',
                   bottom=base_bottom_probs,
                   label='Psucc $\eta_2$', color='green')
            data = [parsed_result['eta_assigned_state_00'],
                    parsed_result['eta_assigned_state_01'],
                    parsed_result['eta_assigned_state_10'],
                    parsed_result['eta_assigned_state_11'],
                    parsed_result['counts_00_eta0'], parsed_result['counts_00_eta1'], parsed_result['counts_00_eta2'],
                    parsed_result['counts_01_eta0'], parsed_result['counts_01_eta1'], parsed_result['counts_01_eta2'],
                    parsed_result['counts_10_eta0'], parsed_result['counts_10_eta1'], parsed_result['counts_10_eta2'],
                    parsed_result['counts_11_eta0'], parsed_result['counts_11_eta1'], parsed_result['counts_11_eta2'],
                    parsed_result['total_counts']]
            columns = parsed_result['etas_third_channel']
            rows = ('state 00', 'state 01', 'state 10', 'state 11',
                    'counts_00_eta0', 'counts_00_eta1', 'counts_00_eta2',
                    'counts_01_eta0', 'counts_01_eta1', 'counts_01_eta2',
                    'counts_10_eta0', 'counts_10_eta1', 'counts_10_eta2',
                    'counts_11_eta0', 'counts_11_eta1', 'counts_11_eta2',
                    'total_counts')
            ax.table(cellText=data,
                     rowLabels=rows,
                     colLabels=columns,
                     loc='bottom')
            ax.legend()
        if eta_pair_index < 0:
            plt.subplots_adjust(hspace=0.6)
        plt.show()

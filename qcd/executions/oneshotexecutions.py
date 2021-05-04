from . import Execution
from typing import Optional, Union, List, cast
from ..typings import OneShotResults
from .aux import draw_cube
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


class OneShotExecution(Execution):
    """ Representation of the One Shot Execution """

    def __init__(self, results: Union[OneShotResults, List[OneShotResults]]):
        self._results = results

    def plot_surface_probabilities(self, results_index: Optional[int] = 0) -> None:
        """ Displays the output probabilities for all circuits in a 3D plot """
        results = self._get_one_result(results_index)
        fig = plt.figure(figsize=(25, 35))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(results['probabilities']['x_input_0'],
                        results['attenuation_factor_per_state'],
                        results['probabilities']['z_output_0'],
                        cmap=cm.coolwarm,
                        linewidth=1,
                        antialiased=True)
        ax.set_title("Output Probabilities for $\\vert0\\rangle$", fontsize=30)
        plt.ylabel("Attenuation factor $\lambda$")
        plt.xlabel("Input State ||" + "$\\alpha||^2 |0\\rangle$")

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        ax.plot_surface(results['probabilities']['x_input_1'],
                        results['attenuation_factor_per_state'],
                        results['probabilities']['z_output_1'],
                        cmap=cm.coolwarm,
                        linewidth=1,
                        antialiased=True)
        ax.set_title("Output Probabilities for $\\vert1\\rangle$", fontsize=30)
        plt.ylabel("Attenuation factor $\lambda$")
        plt.xlabel("Input State ||" + "$\\beta||^2 |1\\rangle$")

        plt.show()

    def plot_wireframe_blochs(self,
                              results_index: Optional[int] = 0,
                              in_rows: Optional[int] = 3,
                              in_cols: Optional[int] = 3,
                              show_sample_states: Optional[bool] = False) -> None:
        """ Displays the resulting Bloch Spheres after the input states travels through the channel  """
        results = self._get_one_result(results_index)
        rows = in_rows
        cols = in_cols
        if rows is None:
            rows = 3
        if cols is None:
            cols = 3

        fig = plt.figure(figsize=(20, 25))
        # ===============
        #  First subplot
        # ===============
        # set up the axes for the first plot
        ax = fig.add_subplot(rows, cols, 1, projection='3d')
        draw_cube(ax)

        # draw initial states
        wf = ax.plot_wireframe(results['initial_states_reshaped']['reshapedCoordsX'],
                               results['initial_states_reshaped']['reshapedCoordsY'],
                               results['initial_states_reshaped']['reshapedCoordsZ'],
                               color="c")
        ax.set_title("Input States")
        if not show_sample_states:
            # draw center
            ax.scatter([0], [0], [0], color="g", s=50)
        # draw one state
        else:
            ax.scatter([results['initial_states_reshaped']['reshapedCoordsX'][5][7]],
                       [results['initial_states_reshaped']['reshapedCoordsY'][5][7]],
                       [results['initial_states_reshaped']['reshapedCoordsZ'][5][7]],
                       color="k", s=100, zorder=2)

            ax.scatter([results['initial_states_reshaped']['reshapedCoordsX'][13][8]],
                       [results['initial_states_reshaped']['reshapedCoordsY'][13][8]],
                       [results['initial_states_reshaped']['reshapedCoordsZ'][13][8]],
                       color="b", s=100, zorder=3)
        wf.set_zorder(1)
        # ===============
        # Next subplots
        # ===============
        attenuation_factors = results['attenuation_factors']
        # modulus_number = np.round(len(results['final_states_reshaped']) / (rows * cols - 1))
        index_to_print = 2
        for idx, final_state_reshaped in enumerate(results['final_states_reshaped']):
            if (idx == 0 or idx == 5 or idx == 6 or idx == 8 or idx == 10 or
                    idx == 12 or idx == 14 or idx == 16 or idx == 20):
                # if ((index_to_print == 0 or len(results['final_states_reshaped']) < modulus_number) or
                #         (index_to_print != 0 and idx % modulus_number == 0 and
                #          index_to_print < (rows * cols - 1)) or
                #         (idx == len(results['final_states_reshaped']) - 1)):
                # set up the axes for the second plot
                ax = fig.add_subplot(rows, cols, index_to_print, projection='3d')
                draw_cube(ax)

                # draw final states
                in_wf = ax.plot_wireframe(final_state_reshaped['reshaped_coords_x'],
                                          final_state_reshaped['reshaped_coords_y'],
                                          final_state_reshaped['reshaped_coords_z'],
                                          color="r")
                title = f'Output States\n Channel $\lambda= {np.round(attenuation_factors[idx], 2)}$'
                ax.set_title(title)
                if not show_sample_states:
                    # draw center
                    ax.scatter([0], [0], final_state_reshaped["center"], color="g", s=50)
                else:
                    # final one state reshaped
                    ax.scatter([final_state_reshaped['reshaped_coords_x'][5][7]],
                               [final_state_reshaped['reshaped_coords_y'][5][7]],
                               [final_state_reshaped['reshaped_coords_z'][5][7]],
                               color="k", s=100, zorder=2)

                    ax.scatter([final_state_reshaped['reshaped_coords_x'][13][8]],
                               [final_state_reshaped['reshaped_coords_y'][13][8]],
                               [final_state_reshaped['reshaped_coords_z'][13][8]],
                               color="b", s=100, zorder=3)
                in_wf.set_zorder(1)
                index_to_print += 1

        plt.show()

    def plot_wireframe_blochs_one_lambda(self,
                                         rows: Optional[int] = 3,
                                         cols: Optional[int] = 3) -> None:
        """ Displays the resulting Bloch Spheres after the input states travels through the channel
            using only the provided attenuation level (lambda) """
        results = self._get_results()
        fig = plt.figure(figsize=(20, 25))
        for idx, result in enumerate(results):
            # set up the axes for the second plot
            ax = fig.add_subplot(rows, cols, 1 + idx, projection='3d')
            draw_cube(ax)

            final_state_reshaped = result['final_states_reshaped'][0]
            # draw final states
            ax.plot_wireframe(final_state_reshaped['reshaped_coords_x'],
                              final_state_reshaped['reshaped_coords_y'],
                              final_state_reshaped['reshaped_coords_z'],
                              color="r")

            attenuation_factor = np.round(result['attenuation_factors'][0], 1)
            title = f"Output States executed on {result['backend_name']}\n Channel $\lambda= {attenuation_factor}$"
            ax.set_title(title)
            # draw center
            ax.scatter([0], [0], result['final_states_reshaped'][0]["center"], color="g", s=50)

        plt.show()

    def _get_one_result(self, results_index: Optional[int] = 0) -> OneShotResults:
        idx = results_index
        if idx is None:
            idx = 0
        if isinstance(self._results, list):
            return cast(List[OneShotResults], self._results)[idx]
        return cast(OneShotResults, self._results)

    def _get_results(self) -> List[OneShotResults]:
        if isinstance(self._results, list):
            return cast(List[OneShotResults], self._results)
        return [cast(OneShotResults, self._results)]

from . import Execution
from typing import Optional
from ..typings import OneShotResults
from matplotlib import cm
import matplotlib.pyplot as plt


class OneShotExecution(Execution):
    """ Representation of the One Shot Execution """

    def __init__(self, results: OneShotResults):
        self._results = results

    def plot_surface_probabilities(self) -> None:
        """ Displays the output probabilities for all circuits in a 3D plot """
        # Representation of output probabilities for all circuit in a 3d plot
        fig = plt.figure(figsize=(25, 35))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(self._results['probabilities']['x_input_0'],
                        self._results['attenuation_factor_per_state'],
                        self._results['probabilities']['z_output_0'],
                        cmap=cm.coolwarm,
                        linewidth=1,
                        antialiased=True)
        ax.set_title("Output Probabilities for $\\vert0\\rangle$", fontsize=30)
        plt.ylabel("Attenuation factor $\lambda$")
        plt.xlabel("Input State ||" + "$\\alpha||^2 |0\\rangle$")

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        ax.plot_surface(self._results['probabilities']['x_input_1'],
                        self._results['attenuation_factor_per_state'],
                        self._results['probabilities']['z_output_1'],
                        cmap=cm.coolwarm,
                        linewidth=1,
                        antialiased=True)
        ax.set_title("Output Probabilities for $\\vert1\\rangle$", fontsize=30)
        plt.ylabel("Attenuation factor $\lambda$")
        plt.xlabel("Input State ||" + "$\\beta||^2 |1\\rangle$")

        plt.show()

    def plot_wireframe_blochs(self, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel  """
        raise NotImplementedError('Method not implemented')

    def plot_wireframe_blochs_one_lambda(self, one_lambda: int, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel
            using only the provided attenuation level (lambda) """
        raise NotImplementedError('Method not implemented')

    def plot_fidelity(self):
        """ Displays the channel fidelity for 11 discrete attenuation levels ranging from
            0 (minimal attenuation) to 1 (maximal attenuation) """
        raise NotImplementedError('Method not implemented')

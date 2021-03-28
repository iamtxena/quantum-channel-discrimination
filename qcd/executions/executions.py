from abc import ABC, abstractmethod
from typing import Optional


class Execution(ABC):
    """ Generic class acting as an interface for any Execution Output """

    @abstractmethod
    def plot_surface_probabilities(self):
        """ Displays the output probabilities for all circuits in a 3D plot """
        pass

    @abstractmethod
    def plot_wireframe_blochs(self, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel  """
        pass

    @abstractmethod
    def plot_wireframe_blochs_one_lambda(self, one_lambda: int, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel
            using only the provided attenuation level (lambda) """
        pass

    @abstractmethod
    def plot_fidelity(self):
        """ Displays the channel fidelity for 11 discrete attenuation levels ranging from
            0 (minimal attenuation) to 1 (maximal attenuation) """
        pass

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
    def plot_wireframe_blochs_one_lambda(self, rows: Optional[int] = 3, cols: Optional[int] = 3):
        """ Displays the resulting Bloch Spheres after the input states travels through the channel
            using only the provided attenuation level (lambda) """
        pass

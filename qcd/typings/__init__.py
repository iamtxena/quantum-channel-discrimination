""" Typings from all qcd module """
from typing import List, TypedDict


class ResultStates(TypedDict):
    zero_amplitude: List[float]
    one_amplitude: List[complex]


class ResultState(TypedDict):
    zero_amplitude: float
    one_amplitude: complex


class ResultStatesReshaped(TypedDict):
    reshaped_coords_x: List[float]
    reshaped_coords_y: List[float]
    reshaped_coords_z: List[float]
    center: float

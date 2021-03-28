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


class ResultProbabilitiesOneChannel(TypedDict):
    x_input_0: List[float]
    x_input_1: List[float]
    z_output_0: List[float]
    z_output_1: List[complex]


class ResultProbabilities(TypedDict):
    x_input_0: List[List[float]]
    x_input_1: List[List[float]]
    z_output_0: List[List[float]]
    z_output_1: List[List[complex]]


class OneShotResults(TypedDict):
    final_states: List[ResultStates]
    final_states_reshaped: List[ResultStatesReshaped]
    probabilities: ResultProbabilities
    attenuation_factor_per_state: List[List[float]]
    backend_name: str

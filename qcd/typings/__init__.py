""" Typings from all qcd module """
from typing import List, TypedDict, Tuple
import enum


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
    attenuation_factors: List[float]
    attenuation_factor_per_state: List[List[float]]
    backend_name: str


class OptimizationSetup(TypedDict, total=False):
    optimizer_algorithms: List[str]
    optimizer_iterations: List[int]
    eta_partitions: int
    number_channels_to_discriminate: int
    plays: int
    initial_parameters: List[float]
    variable_bounds: List[Tuple[float, float]]


class TheoreticalOptimizationSetup(TypedDict):
    eta_groups: List[List[float]]


class GuessStrategy(enum.Enum):
    one_bit_same_as_measured = 1
    two_bit_base = 2
    two_bit_neural_network = 3


class CloneSetup(TypedDict, total=False):
    total_clones: int
    id_clone: int
    file_name: str
    path: str

""" Typings from all qcd module """
from typing import List, TypedDict, Tuple
from ..configurations.configuration import ChannelConfiguration


class OptimalConfiguration(TypedDict):
    best_algorithm: str
    best_probability: float
    best_configuration: ChannelConfiguration
    number_calls_made: int


class OptimalConfigurations(TypedDict, total=False):
    eta_pairs: List[Tuple[float, float]]
    best_algorithm: List[str]
    probabilities: List[float]
    configurations: List[ChannelConfiguration]
    number_calls_made: List[int]
    legacy: bool


class TheoreticalOneShotOptimalConfiguration(OptimalConfiguration):
    best_theoretical_amplitude: float


class TheoreticalOneShotEntangledOptimalConfiguration(TheoreticalOneShotOptimalConfiguration):
    improvement: float


class TheoreticalOneShotOptimalConfigurations(OptimalConfigurations):
    list_theoretical_amplitude: List[float]


class TheoreticalOneShotEntangledOptimalConfigurations(TheoreticalOneShotOptimalConfigurations):
    improvements: List[float]

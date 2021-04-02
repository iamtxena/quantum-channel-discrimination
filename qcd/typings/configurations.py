""" Typings from all qcd module """
from typing import List, TypedDict, Tuple
from ..configurations.configuration import ChannelConfiguration


class OptimalConfiguration(TypedDict):
    best_algorithm: str
    best_probability: float
    best_configuration: ChannelConfiguration
    number_calls_made: int


class OptimalConfigurations(TypedDict):
    eta_pairs: List[Tuple[float, float]]
    best_algorithm: List[str]
    probabilities: List[float]
    configurations: List[ChannelConfiguration]
    number_calls_made: List[int]


class TheoreticalOptimalConfiguration(TypedDict):
    best_probability: float


class TheoreticalOneShotOptimalConfiguration(TheoreticalOptimalConfiguration):
    best_theoretical_amplitude: float


class TheoreticalOneShotEntangledOptimalConfiguration(TheoreticalOneShotOptimalConfiguration):
    improvement: float


class TheoreticalOptimalConfigurations(TypedDict):
    eta_pairs: List[Tuple[float, float]]
    probabilities: List[float]


class TheoreticalOneShotOptimalConfigurations(TheoreticalOptimalConfigurations):
    list_theoretical_amplitude: List[float]


class TheoreticalOneShotEntangledOptimalConfigurations(TheoreticalOneShotOptimalConfigurations):
    improvements: List[float]

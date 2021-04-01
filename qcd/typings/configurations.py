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

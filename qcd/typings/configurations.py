""" Typings from all qcd module """
from typing import List, TypedDict, Tuple
from ..configurations import ChannelConfiguration


class OneShotConfigurationDict(TypedDict):
    theta: float
    phase: float
    angle_rx: float
    angle_ry: float
    attenuation_pair: Tuple[float, float]


class OptimalConfiguration(TypedDict):
    best_algorithm: str
    best_probability: float
    best_configuration: ChannelConfiguration


class OptimalConfigurations(TypedDict):
    attenuation_pairs: List[Tuple[float, float]]
    best_algorithm: List[str]
    probabilities: List[float]
    configurations: List[ChannelConfiguration]

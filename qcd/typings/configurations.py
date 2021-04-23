""" Typings from all qcd module """
from typing import List, Tuple, TypedDict
from ..configurations.configuration import ChannelConfiguration
from enum import Enum


class Eta(Enum):
    ETA0 = 0
    ETA1 = 1
    ETA2 = 2


class MeasuredStatesEtaAssignment(TypedDict):
    state_00: Eta
    state_01: Eta
    state_10: Eta
    state_11: Eta


class OptimalConfiguration(TypedDict):
    best_algorithm: str
    best_probability: float
    best_configuration: ChannelConfiguration
    number_calls_made: int


class OptimalConfigurations(TypedDict, total=False):
    eta_groups: List[List[float]]
    best_algorithm: List[str]
    probabilities: List[float]
    configurations: List[ChannelConfiguration]
    number_calls_made: List[int]
    legacy: bool
    validated_probabilities: List[float]
    eta_probabilities: List[Tuple[float, float, float]]
    measured_states_eta_assignment: List[MeasuredStatesEtaAssignment]


class TheoreticalOneShotOptimalConfiguration(OptimalConfiguration):
    best_theoretical_amplitude: float


class TheoreticalOneShotEntangledOptimalConfiguration(TheoreticalOneShotOptimalConfiguration):
    improvement: float


class TheoreticalOneShotOptimalConfigurations(OptimalConfigurations):
    list_theoretical_amplitude: List[float]


class TheoreticalOneShotEntangledOptimalConfigurations(TheoreticalOneShotOptimalConfigurations):
    improvements: List[float]

""" Typings from all qcd module """
from typing import List, TypedDict
from ..configurations.configuration import ChannelConfiguration


class MeasuredStatesEtaAssignment(TypedDict):
    state_00: int
    state_01: int
    state_10: int
    state_11: int


class MeasuredStatesCounts(TypedDict):
    state_00: List[int]
    state_01: List[int]
    state_10: List[int]
    state_11: List[int]
    total_counts: int


class Fidelities(TypedDict):
    upper_bound_fidelity: float
    lower_bound_fidelity: float


class OptimalConfiguration(TypedDict, total=False):
    best_algorithm: str
    best_probability: float
    best_configuration: ChannelConfiguration
    number_calls_made: int


class ValidatedConfiguration(OptimalConfiguration, total=False):
    validated_probability: float
    etas_probability: List[float]
    measured_states_eta_assignment: MeasuredStatesEtaAssignment
    fidelities: Fidelities
    measured_states_counts: MeasuredStatesCounts


class OptimalConfigurations(TypedDict, total=False):
    eta_groups: List[List[float]]
    best_algorithm: List[str]
    probabilities: List[float]
    configurations: List[ChannelConfiguration]
    number_calls_made: List[int]
    legacy: bool
    validated_probabilities: List[float]
    eta_probabilities: List[List[float]]
    measured_states_eta_assignment: List[MeasuredStatesEtaAssignment]
    fidelities: List[Fidelities]
    measured_states_counts: List[MeasuredStatesCounts]


class TheoreticalOneShotOptimalConfiguration(OptimalConfiguration):
    best_theoretical_amplitude: float


class TheoreticalOneShotEntangledOptimalConfiguration(TheoreticalOneShotOptimalConfiguration):
    improvement: float


class TheoreticalOneShotOptimalConfigurations(OptimalConfigurations):
    list_theoretical_amplitude: List[float]


class TheoreticalOneShotEntangledOptimalConfigurations(TheoreticalOneShotOptimalConfigurations):
    improvements: List[float]

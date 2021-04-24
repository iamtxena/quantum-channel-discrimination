from typing import Tuple, TypedDict, List


class OneShotConfigurationDict(TypedDict, total=False):
    state_probability: float
    angle_rx: float
    angle_ry: float
    eta_group: List[float]
    theta: float


class OneShotEntangledConfigurationDict(OneShotConfigurationDict, total=False):
    angle_rx0: float
    angle_ry0: float
    angle_rx1: float
    angle_ry1: float


class OneShotEntangledFullInputConfigurationDict(OneShotEntangledConfigurationDict, total=False):
    angle_rx_input0: float
    angle_ry_input0: float
    angle_rx_input1: float
    angle_ry_input1: float


class OneShotEntangledUniversalConfigurationDict(OneShotEntangledFullInputConfigurationDict, total=False):
    theta0: float
    phi0: float
    lambda0: float
    theta1: float
    phi1: float
    lambda1: float
    theta2: float
    phi2: float
    lambda2: float
    theta3: float
    phi3: float
    lambda3: float
    theta4: float
    phi4: float
    lambda4: float
    theta5: float
    phi5: float
    lambda5: float
    theta6: float
    phi6: float
    lambda6: float
    theta7: float
    phi7: float
    lambda7: float


class OneShotEntangledFullUniversalConfigurationDict(OneShotEntangledUniversalConfigurationDict, total=False):
    input_theta0: float
    input_phi0: float
    input_lambda0: float
    input_theta1: float
    input_phi1: float
    input_lambda1: float
    input_theta2: float
    input_phi2: float
    input_lambda2: float
    input_theta3: float
    input_phi3: float
    input_lambda3: float
    input_theta4: float
    input_phi4: float
    input_lambda4: float
    input_theta5: float
    input_phi5: float
    input_lambda5: float
    input_theta6: float
    input_phi6: float
    input_lambda6: float
    input_theta7: float
    input_phi7: float
    input_lambda7: float


class ResultsToPlot(TypedDict):
    error_probabilities: List[float]
    error_probabilities_validated: List[float]
    etas_third_channel: List[int]
    upper_fidelities: List[float]
    lower_fidelities: List[float]
    eta0_error_probabilities: List[float]
    eta1_error_probabilities: List[float]
    eta2_error_probabilities: List[float]
    eta_assigned_state_00: List[str]
    eta_assigned_state_01: List[str]
    eta_assigned_state_10: List[str]
    eta_assigned_state_11: List[str]
    eta_pair: Tuple[int, int]

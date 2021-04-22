from typing import TypedDict, List


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

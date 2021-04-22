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

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

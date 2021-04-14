from typing import TypedDict, List


class OneShotConfigurationDict(TypedDict, total=False):
    state_probability: float
    angle_rx: float
    angle_ry: float
    eta_group: List[float]
    theta: float

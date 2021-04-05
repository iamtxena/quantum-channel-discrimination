from typing import TypedDict, Tuple


class OneShotConfigurationDict(TypedDict, total=False):
    state_probability: float
    angle_rx: float
    angle_ry: float
    eta_pair: Tuple[float, float]
    theta: float

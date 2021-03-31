from typing import TypedDict, Tuple


class OneShotConfigurationDict(TypedDict):
    theta: float
    angle_rx: float
    angle_ry: float
    eta_pair: Tuple[float, float]

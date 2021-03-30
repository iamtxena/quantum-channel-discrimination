from typing import TypedDict, Tuple


class OneShotConfigurationDict(TypedDict):
    theta: float
    phase: float
    angle_rx: float
    angle_ry: float
    attenuation_pair: Tuple[float, float]

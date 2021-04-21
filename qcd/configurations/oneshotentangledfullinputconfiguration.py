from . import OneShotEntangledConfiguration
from ..typings.dicts import OneShotEntangledFullInputConfigurationDict
from typing import Dict


class OneShotEntangledFullInputConfiguration(OneShotEntangledConfiguration):
    """ Definition for One Shot Entangled channel configuration with Full Input"""

    def __init__(self, configuration: OneShotEntangledFullInputConfigurationDict) -> None:
        self._angle_rx_input0 = configuration['angle_rx_input0']
        self._angle_ry_input0 = configuration['angle_ry_input0']
        self._angle_rx_input1 = configuration['angle_rx_input1']
        self._angle_ry_input1 = configuration['angle_ry_input1']
        super().__init__(configuration)

    @property
    def angle_rx_input0(self) -> float:
        return self._angle_rx_input0

    @property
    def angle_ry_input0(self) -> float:
        return self._angle_ry_input0

    @property
    def angle_rx_input1(self) -> float:
        return self._angle_rx_input1

    @property
    def angle_ry_input1(self) -> float:
        return self._angle_ry_input1

    def to_dict(self) -> Dict:
        return {'angle_rx_input0': self._angle_rx_input0,
                'angle_ry_input0': self._angle_ry_input0,
                'angle_rx_input1': self._angle_rx_input1,
                'angle_ry_input1': self._angle_ry_input1,
                'angle_rx0': self._angle_rx0,
                'angle_ry0': self._angle_ry0,
                'angle_rx1': self._angle_rx1,
                'angle_ry1': self._angle_ry1,
                'eta_group': self._eta_group}

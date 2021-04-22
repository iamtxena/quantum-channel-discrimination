from . import OneShotConfiguration
from ..typings.dicts import OneShotEntangledConfigurationDict
from typing import Dict


class OneShotEntangledConfiguration(OneShotConfiguration):
    """ Definition for One Shot Entangled channel configuration """

    def __init__(self, configuration: OneShotEntangledConfigurationDict) -> None:
        if 'angle_rx0' in configuration:
            self._angle_rx0 = configuration['angle_rx0']
        if 'angle_ry0' in configuration:
            self._angle_ry0 = configuration['angle_ry0']
        if 'angle_rx1' in configuration:
            self._angle_rx1 = configuration['angle_rx1']
        if 'angle_ry0' in configuration:
            self._angle_ry1 = configuration['angle_ry1']
        super().__init__(configuration)

    @property
    def angle_rx0(self) -> float:
        return self._angle_rx0

    @property
    def angle_ry0(self) -> float:
        return self._angle_ry0

    @property
    def angle_rx1(self) -> float:
        return self._angle_rx1

    @property
    def angle_ry1(self) -> float:
        return self._angle_ry1

    def to_dict(self) -> Dict:
        return {'state_probability': self._state_probability if hasattr(self, 'state_probability') else self._theta,
                'angle_rx0': self._angle_rx0,
                'angle_ry0': self._angle_ry0,
                'angle_rx1': self._angle_rx1,
                'angle_ry1': self._angle_ry1,
                'eta_group': self._eta_group}

from . import ChannelConfiguration
from ..typings.dicts import OneShotConfigurationDict
from typing import cast, Dict


class OneShotConfiguration(ChannelConfiguration):
    """ Definition for One Shot channel configuration """

    def __init__(self, configuration: OneShotConfigurationDict) -> None:
        self._state_probability = configuration['state_probability']
        self._angle_rx = configuration['angle_rx']
        self._angle_ry = configuration['angle_ry']
        if 'theta' in configuration:
            self._theta = configuration['theta']
        super().__init__(cast(dict, configuration))

    @property
    def state_probability(self) -> float:
        return self._state_probability

    @property
    def angle_rx(self) -> float:
        return self._angle_rx

    @property
    def angle_ry(self) -> float:
        return self._angle_ry

    @property
    def theta(self) -> float:
        if self._theta is None:
            raise ValueError('theta not defined')
        return self._theta

    def to_dict(self) -> Dict:
        return {'state_probability': self._state_probability if hasattr(self, 'state_probability') else self._theta,
                'angle_rx': self._angle_rx,
                'angle_ry': self._angle_ry,
                'eta_pair': self._eta_pair,
                }

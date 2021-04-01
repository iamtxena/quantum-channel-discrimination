from . import ChannelConfiguration
from ..typings.dicts import OneShotConfigurationDict
from typing import cast


class OneShotConfiguration(ChannelConfiguration):
    """ Definition for One Shot channel configuration """

    def __init__(self, configuration: OneShotConfigurationDict) -> None:
        self._theta = configuration['theta']
        self._angle_rx = configuration['angle_rx']
        self._angle_ry = configuration['angle_ry']
        super().__init__(cast(dict, configuration))

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def angle_rx(self) -> float:
        return self._angle_rx

    @property
    def angle_ry(self) -> float:
        return self._angle_ry

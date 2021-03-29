from abc import ABC
from typing import Tuple, cast


class ChannelConfiguration(ABC):
    """ Generic class acting as an interface for any Channel Configuration """

    def __init__(self, configuration: dict) -> None:
        self._attenuation_pair = cast(Tuple[float, float], configuration['attenuation_pair'])

    @property
    def attenuation_pair(self) -> Tuple[float, float]:
        return self._attenuation_pair

from abc import ABC
from typing import Tuple, cast


class ChannelConfiguration(ABC):
    """ Generic class acting as an interface for any Channel Configuration """

    def __init__(self, configuration: dict) -> None:
        self._eta_pair = cast(Tuple[float, float], configuration['eta_pair'])

    @property
    def eta_pair(self) -> Tuple[float, float]:
        return self._eta_pair

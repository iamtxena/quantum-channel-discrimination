from abc import ABC, abstractmethod
from typing import Tuple, cast, Dict


class ChannelConfiguration(ABC):
    """ Generic class acting as an interface for any Channel Configuration """

    def __init__(self, configuration: dict) -> None:
        self._eta_pair = cast(Tuple[float, float], configuration['eta_pair'])

    @property
    def eta_pair(self) -> Tuple[float, float]:
        return self._eta_pair

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

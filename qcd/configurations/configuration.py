from abc import ABC, abstractmethod
from typing import List, cast, Dict


class ChannelConfiguration(ABC):
    """ Generic class acting as an interface for any Channel Configuration """

    def __init__(self, configuration: dict) -> None:
        self._eta_group = cast(List[float], configuration['eta_group'])

    @property
    def eta_group(self) -> List[float]:
        return self._eta_group

    def reorder_eta_group(self):
        self._eta_group.sort(reverse=True)
        return self

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

from . import OneShotConfiguration
from ..typings.dicts import OneShotConfigurationDict


class OneShotEntangledConfiguration(OneShotConfiguration):
    """ Definition for One Shot Entangled channel configuration """

    def __init__(self, configuration: OneShotConfigurationDict) -> None:
        super().__init__(configuration)

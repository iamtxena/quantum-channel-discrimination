""" Quantum Damping Channels """

from .dampingchannel import DampingChannel
from .oneshotbasechannel import OneShotDampingChannel, ResultStatesReshaped
from .oneshotentangledchannel import OneShotEntangledDampingChannel
from .oneshotentangledfullinputchannel import OneShotEntangledFullInputDampingChannel
from .oneshotentangleduniversal import OneShotEntangledUniversalDampingChannel
from .oneshotentangledfulluniversal import OneShotEntangledFullUniversalDampingChannel
from .oneshotprojectivem import OneShotDampingChannelProjectivMeasurement
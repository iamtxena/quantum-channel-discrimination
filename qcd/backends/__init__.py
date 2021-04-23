""" Defined Device Backends """

from .devicebackend import DeviceBackend
from .fakenoisysimulator import FakeSimulatorBackend
from .ibmq import DeviceIBMQ
from .simulator import SimulatorBackend
from .statevector_simulator import StateVectorSimulatorBackend

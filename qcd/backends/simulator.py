from . import DeviceBackend
from qiskit import Aer


class SimulatorBackend(DeviceBackend):

    """ Representation of the Aer Simulator backend """

    def __init__(self) -> None:
        self._backend = Aer.get_backend('qasm_simulator')
        super().__init__(self._backend)

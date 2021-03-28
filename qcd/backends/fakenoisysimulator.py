from . import DeviceBackend
from qiskit.test.mock import FakeAthens
from qiskit.providers.aer import QasmSimulator


class FakeSimulatorBackend(DeviceBackend):

    """ Representation of the Fake Noisy Simulator backend """

    def __init__(self) -> None:
        """ using FakeAthens as default """
        self._backend = QasmSimulator.from_backend(FakeAthens())
        super().__init__(self._backend)

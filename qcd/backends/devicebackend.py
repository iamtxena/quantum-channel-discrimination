from abc import ABC
from qiskit.providers import BackendV1 as Backend


class DeviceBackend(ABC):
    """ Generic class acting as an interface for any Device Backend """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @property
    def backend(self) -> Backend:
        return self._backend

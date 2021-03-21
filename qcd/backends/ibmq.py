from . import DeviceBackend
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy


class DeviceIBMQ(DeviceBackend):

    """ Representation of the IBMQ Real backend """

    def __init__(self) -> None:
        provider = IBMQ.load_account()
        # Load IBM Q account and get the least busy backend device
        self._backend = least_busy(provider.backends(
            filters=lambda x: x.configuration().n_qubits >= 2, simulator=False))
        print("Running on current least busy device: ", self._backend)
        super().__init__(self._backend)

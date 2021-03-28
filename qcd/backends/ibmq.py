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

    def get_max_experiments(self) -> int:
        """ Return the maximum number of experiments allowed for the device """
        return self._backend.configuration().to_dict().get('max_experiments')

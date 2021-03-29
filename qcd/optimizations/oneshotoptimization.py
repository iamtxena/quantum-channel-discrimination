
from . import Optimization
from ..typings import OptimizationSetup, OneShotConfigurationDict
import numpy as np
from typing import Tuple, cast
from ..configurations import ChannelConfiguration, OneShotConfiguration
import math
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute


class OneShotOptimization(Optimization):
    """ Representation of the One Shot Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: np.ndarray[float],
                                                            attenuation_pair: Tuple[float, float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotConfiguration(
            OneShotConfigurationDict(
                theta=configuration[0],
                phase=configuration[1],
                angle_rx=configuration[2],
                angle_ry=configuration[3],
                attenuation_pair=attenuation_pair
            )
        )

    def _prepare_initial_state(self, theta: float, phase: float) -> Tuple[complex, complex]:
        """ Prepare initial state """
        return (math.cos(theta) * (1 + 0j),
                (math.sin(theta) * math.e**(1j * phase) + 0 + 0j))

    def _convert_counts_to_final_result(self, counts: str) -> int:
        """ Convert the execution result to the final measured value: 0 or 1 """
        if "0" in counts:
            return 0
        return 1

    def _compute_damping_channel(self, channel_configuration: ChannelConfiguration, attenuation_index: int) -> int:
        """ one-time execution of the amplitude damping circuit using the passed parameters
            Returns: the execution measured result: either 0 or 1
        """
        configuration = cast(OneShotConfiguration, channel_configuration)
        backend = backend = Aer.get_backend('qasm_simulator')
        attenuation_factor = configuration.attenuation_pair[attenuation_index]
        qreg_q = QuantumRegister(2, 'q')
        creg_c = ClassicalRegister(1, 'c')

        initial_state = self._prepare_initial_state(configuration.theta, configuration.phase)

        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.initialize([initial_state[0],
                            initial_state[1]], qreg_q[0])
        circuit.reset(qreg_q[1])
        circuit.cry(2 * np.arcsin(np.sqrt(attenuation_factor)), qreg_q[0], qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.rx(configuration.angle_rx, qreg_q[0])
        circuit.ry(configuration.angle_ry, qreg_q[0])
        circuit.measure(qreg_q[0], creg_c[0])

        counts = execute(circuit, backend, shots=1).result().get_counts(circuit)
        return self._convert_counts_to_final_result(counts)

    def _guess_lambda_used(self, real_measured_result: int) -> int:
        """ Decides which lambda was used on the real execution.
            It is a silly guess.
            It returns the same lambda used as the measured result
        """
        return real_measured_result

    def _cost_function(self, params: np.ndarray[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
            defined in the optimization setup.
            Cost is computed as 1 (perfect probability) - average success probability for
            all the plays with the given configuration
            Returns the Cost (error probability).
        """
        configuration = OneShotConfiguration(
            OneShotConfigurationDict(
                theta=params[0],
                phase=params[1],
                angle_rx=params[2],
                angle_ry=params[3],
                attenuation_pair=self._global_attenuation_pair
            )
        )

        success_counts = 0
        for play in range(self._setup['plays']):
            success_counts += self._play_and_guess_one_case(configuration)

        return 1 - (success_counts / self._setup['plays'])

from . import Optimization
from ..typings import OptimizationSetup, GuessStrategy
from typing import Tuple, cast, List
from ..configurations import ChannelConfiguration, OneShotConfiguration
from .aux import get_measured_value_from_counts
import math
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute


class OneShotOptimization(Optimization):
    """ Representation of the One Shot Channel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)

    def _convert_optimizer_results_to_channel_configuration(self,
                                                            configuration: List[float],
                                                            eta_pair: Tuple[float, float]
                                                            ) -> ChannelConfiguration:
        """ Convert the results of an optimization to a One Shot channel configuration """
        return OneShotConfiguration({
            'theta': configuration[0],
            'angle_rx': configuration[1],
            'angle_ry': configuration[2],
            'eta_pair': eta_pair})

    def _prepare_initial_state(self, theta: float) -> Tuple[complex, complex]:
        """ Prepare initial state """
        return (math.cos(theta), math.sin(theta))

    def _guess_eta_used_one_bit_strategy(self, counts: str) -> int:
        """ Decides which eta was used on the real execution from the one bit 'counts' measured
            It is a silly guess.
            It returns the same eta index used as the measured result
        """
        if len(counts) != 1:
            raise ValueError('counts MUST be a one character length string')
        if "0" in counts:
            return 0
        return 1

    def _convert_counts_to_eta_used(self,
                                    counts_dict: dict,
                                    guess_strategy: GuessStrategy) -> int:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        if guess_strategy != GuessStrategy.one_bit_same_as_measured:
            raise ValueError('Invalid Guess Strategy. Only GuessStrategy.one_bit_same_as_measured supported')

        counts = get_measured_value_from_counts(counts_dict)
        return self._guess_eta_used_one_bit_strategy(counts)

    def _compute_damping_channel(self, channel_configuration: ChannelConfiguration, eta_index: int) -> int:
        """ one-time execution of the amplitude damping circuit using the passed parameters
            Returns: the execution measured result: either 0 or 1
        """
        configuration = cast(OneShotConfiguration, channel_configuration)
        backend = backend = Aer.get_backend('qasm_simulator')
        eta = configuration.eta_pair[eta_index]
        qreg_q = QuantumRegister(2, 'q')
        creg_c = ClassicalRegister(1, 'c')

        initial_state = self._prepare_initial_state(configuration.theta)

        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.initialize([initial_state[0],
                            initial_state[1]], qreg_q[0])
        circuit.reset(qreg_q[1])
        circuit.cry(2 * eta, qreg_q[0], qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.rx(configuration.angle_rx, qreg_q[0])
        circuit.ry(configuration.angle_ry, qreg_q[0])
        circuit.measure(qreg_q[0], creg_c[0])

        counts = execute(circuit, backend, shots=1).result().get_counts(circuit)
        return self._convert_counts_to_eta_used(counts, guess_strategy=GuessStrategy.one_bit_same_as_measured)

    def _cost_function(self, params: List[float]) -> float:
        """ Computes the cost of running a specific configuration for the number of plays
            defined in the optimization setup.
            Cost is computed as 1 (perfect probability) - average success probability for
            all the plays with the given configuration
            Returns the Cost (error probability).
        """
        configuration = OneShotConfiguration({
            'theta': params[0],
            'angle_rx': params[1],
            'angle_ry': params[2],
            'eta_pair': self._global_eta_pair})

        success_counts = 0
        for play in range(self._setup['plays']):
            success_counts += self._play_and_guess_one_case(configuration)

        return 1 - (success_counts / self._setup['plays'])

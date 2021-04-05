from qcd.circuits import OneShotEntangledCircuit
from qcd.typings import OptimizationSetup
from . import OneShotOptimization


class OneShotEntangledOptimization(OneShotOptimization):
    """ Representation of the One Shot EntangledChannel Optimization """

    def __init__(self, optimization_setup: OptimizationSetup):
        super().__init__(optimization_setup)
        self._one_shot_circuit = OneShotEntangledCircuit()

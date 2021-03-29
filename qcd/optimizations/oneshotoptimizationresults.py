from . import OptimizationResults
from typing import Optional
from ..typings import OptimalConfigurations


class OneShotOptimizationResults(OptimizationResults):
    """ Representation of the One Shot Optimization Results """

    def __init__(self, optimal_configurations: Optional[OptimalConfigurations] = None) -> None:
        super().__init__(optimal_configurations)

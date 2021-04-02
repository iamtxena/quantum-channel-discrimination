from typing import cast
from . import OneShotOptimizationResult
from ..typings.configurations import OptimalConfigurations
from ..configurations import OneShotConfiguration


def build_optimization_result(
        optimal_configurations: OptimalConfigurations) -> OneShotOptimizationResult:
    """ Create a Optimization Result subtype class based on its structure """

    configurations = optimal_configurations.get('configurations')
    if configurations is None:
        raise ValueError('configurations MUST be specified')

    if cast(OneShotConfiguration, configurations[0]).theta is not None:
        return OneShotOptimizationResult(optimal_configurations)

    raise ValueError('Supported type for optimal_configurations is only OneShotConfiguration')

from . import OptimizationResult
from ..typings.configurations import OptimalConfigurations


def build_optimization_result(
        optimal_configurations: OptimalConfigurations) -> OptimizationResult:
    """ Create a Optimization Result subtype class based on its structure """

    configurations = optimal_configurations.get('configurations')
    if configurations is None:
        raise ValueError('configurations MUST be specified')

    return OptimizationResult(optimal_configurations)

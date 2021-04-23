""" Auxiliary static methods """
from functools import reduce
from qcd.configurations.oneshotbaseconfiguration import OneShotConfiguration
from qcd.typings.configurations import OptimalConfigurations
from typing import Dict, List, Tuple, cast
import random
import math


def get_measured_value_from_counts(counts_dict: dict) -> str:
    """ converts the dictionary counts measured to the
          value measured in string format
    """
    if len(list(counts_dict.keys())) != 1:
        raise ValueError('Circuit execution shots MUST be 1')
    return list(counts_dict.keys())[0]


def set_random_eta(eta_pair: Tuple[float, float]) -> int:
    """ return a random choice from attenuation pair with the correspondent index value """
    eta_value = random.choice(eta_pair)
    if eta_value == eta_pair[0]:
        return 0
    return 1


def check_value(real_index_eta: int, guess_index_eta: int):
    if real_index_eta == guess_index_eta:
        return 1
    return 0


def set_only_eta_groups(results: List[Dict]) -> List[OptimalConfigurations]:
    """ Return a fixed results setting eta_groups instead of eta_pairs or lambda_pairs """
    fixed_results = []
    for result in results:
        if 'lambda_pairs' in result:
            result['eta_groups'] = result['lambda_pairs']
            del result['lambda_pairs']
        if 'eta_pairs' in result:
            result['eta_groups'] = result['eta_pairs']
            del result['eta_pairs']
        fixed_results.append(cast(OptimalConfigurations, result))
    return fixed_results


def fix_configurations(result: OptimalConfigurations) -> OptimalConfigurations:
    """ Creates OneShotConfigurations setting and additional key to mark as legacy """

    if len(result['configurations']) <= 0:
        return result
    if (len(result['configurations']) > 0 and
            hasattr(cast(OneShotConfiguration, result['configurations'][0]), 'state_probability')):
        return result
    if (len(result['configurations']) > 0 and
            hasattr(cast(OneShotConfiguration, result['configurations'][0]), 'theta')):
        return _adapt_result(result)

    """ We have to fix configurations, creating a new configuration object """
    fixed_result = result
    for idx, configuration in enumerate(cast(List[List[float]], result['configurations'])):
        new_configuration = OneShotConfiguration({
            'state_probability': (math.sin(configuration[0]))**2,
            'angle_rx': configuration[1],
            'angle_ry': configuration[2],
            'eta_group': cast(dict, result)['eta_group'][idx]
            if isinstance(cast(dict, result)['eta_pairs'][idx][0], float)
            else (math.radians(int(cast(dict, result)['eta_pairs'][idx][0])),
                  math.radians(int(cast(dict, result)['eta_pairs'][idx][1])))
        })
        fixed_result['configurations'][idx] = new_configuration
        fixed_result['eta_groups'][idx] = new_configuration._eta_group

    fixed_result['legacy'] = True

    return fixed_result


def _adapt_result(result):
    adapted_result = result
    for idx, configuration in enumerate(cast(List[OneShotConfiguration], result['configurations'])):
        new_configuration = OneShotConfiguration({
            'state_probability': (math.sin(configuration.theta))**2,
            'angle_rx': configuration.angle_rx,
            'angle_ry': configuration.angle_ry,
            'eta_pair': configuration.eta_pair
        })
        adapted_result['configurations'][idx] = new_configuration
    return adapted_result


def upper_bound_fidelity(fidelities: List[float]) -> float:
    return reduce(lambda total, current: total + (1 / 3) * current, fidelities, 0.0)


def lower_bound_fidelity(fidelities: List[float]) -> float:
    return 1 / 2 * reduce(lambda total, current: total + (1 / 9) * current**2, fidelities, 0.0)

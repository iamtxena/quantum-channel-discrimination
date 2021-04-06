""" Auxiliary static methods """
from typing import Tuple, List, Union, cast
import random
import itertools
import numpy as np
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


def reorder_pairs(pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """ reorder received pairs setting first the element of the tuple
        as greater or equal de second one
    """
    reordered_pairs = pairs
    for idx, pair in enumerate(pairs):
        reordered_pairs[idx] = reorder_pair(pair)
    return reordered_pairs


def reorder_pair(pair: Tuple[float, float]) -> Tuple[float, float]:
    if pair[0] < pair[1]:
        return (pair[1], pair[0])
    return pair


def get_combinations_two_etas_without_repeats_from_lambdas(
        attenuation_factors: List[float]) -> List[Tuple[float, float]]:
    """ from a given list of attenuations factors create a
        list of all combinatorial pairs of possible etas
        without repeats
        For us it is the same testing first eta 0.1 and second eta 0.2
        than first eta 0.2 and second eta 0.1
        Though, we will always put the greater value as the first pair element
    """
    angles_etas = list(map(lambda attenuation_factor: np.arcsin(np.sqrt(attenuation_factor)), attenuation_factors))

    # when there is only one element, we add the same element
    if len(angles_etas) == 1:
        angles_etas.append(angles_etas[0])
    # get combinations of two etas without repeats
    eta_pairs = list(itertools.combinations(angles_etas, 2))

    return reorder_pairs(eta_pairs)


def get_combinations_two_etas_without_repeats_from_etas(angles_etas: List[float]) -> List[Tuple[float, float]]:
    """ from a given list of attenuations factors create a
        list of all combinatorial pairs of possible etas
        without repeats
        For us it is the same testing first eta 0.1 and second eta 0.2
        than first eta 0.2 and second eta 0.1
        Though, we will always put the greater value as the first pair element
    """
    # when there is only one element, we add the same element
    if len(angles_etas) == 1:
        angles_etas.append(angles_etas[0])
    # get combinations of two etas without repeats
    eta_pairs = list(itertools.combinations(angles_etas, 2))

    return reorder_pairs(eta_pairs)


def convert_eta_pairs_from_string_degrees_to_float_radians(
        eta_pairs: List[Tuple[str, str]]) -> List[Tuple[float, float]]:
    return list(map(lambda eta_pair: (math.radians(int(eta_pair[0])), math.radians(int(eta_pair[1]))), eta_pairs))


def parse_eta_pairs(eta_pairs: Union[List[Tuple[float, float]], List[Tuple[str, str]]]) -> List[Tuple[float, float]]:
    """ makes sure that eta values are the float radian values of an angle.
        as some older results executions might have the values in strings format for the angle in degrees.
    """
    if isinstance(eta_pairs[0][0], str):
        return reorder_pairs(
            convert_eta_pairs_from_string_degrees_to_float_radians(cast(List[Tuple[str, str]], eta_pairs)))
    return cast(List[Tuple[float, float]], eta_pairs)

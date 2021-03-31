""" Auxiliary static methods """
from typing import Tuple, List
import random


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
        if pair[0] < pair[1]:
            reordered_pairs[idx] = (pair[1], pair[0])
    return reordered_pairs

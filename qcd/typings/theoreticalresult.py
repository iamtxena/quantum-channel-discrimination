from typing import TypedDict, Literal
from ..optimizationresults.theoreticaloptimizationresults import TheoreticalOneShotOptimizationResult
from ..optimizationresults.theoreticaloptimizationresults import TheoreticalOneShotEntangledOptimizationResult
import enum


class DampingChannelStrategy(enum.Enum):
    one_shot = 'one_shot'
    one_shot_side_entanglement = 'one_shot_side_entanglement'


STRATEGY = Literal['one_shot', 'one_shot_side_entanglement']


class TheoreticalResult(TypedDict):
    one_shot: TheoreticalOneShotOptimizationResult
    one_shot_side_entanglement: TheoreticalOneShotEntangledOptimizationResult

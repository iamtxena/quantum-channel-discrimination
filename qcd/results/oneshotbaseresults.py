from . import ExecutionResults
from typing import List
from ..typings import ResultStatesReshaped


class OneShotResults(ExecutionResults):
    """ Representation of the One Shot Execution Results """

    def __init__(self, results: List[ResultStatesReshaped], backend_name: str):
        super().__init__(results, backend_name)

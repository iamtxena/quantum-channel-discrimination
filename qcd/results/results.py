from abc import ABC
from typing import List
from ..typings import ResultStatesReshaped


class ExecutionResults(ABC):
    """ Generic class acting as an interface for any Execution Results """

    def __init__(self, results: List[ResultStatesReshaped], backend_name: str):
        self._results = results
        self._backend_name = backend_name

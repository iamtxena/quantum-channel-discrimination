from abc import ABC


class ExecutionResults(ABC):
    """ Generic class acting as an interface for any Execution Results """

    def __init__(self, results):
        self._results = results

from . import ExecutionResults


class OneShotResults(ExecutionResults):
    """ Representation of the One Shot Execution Results """

    def __init__(self, results):
        super().__init__(results)

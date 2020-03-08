# -------------------------Error Messages---------------------------------------
class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class SynapseParameterError(Error):
    def __init__(self, message):
        self.message = message


class InputPotentialError(Error):
    def __init__(self, message):
        self.message = message


class DecreasePercentageError(Error):
    def __init__(self, message):
        self.message = message


from abc import ABC

import flwr as fl

from florl.common import Knowledge

class FlorlStrategy(fl.server.strategy.Strategy, ABC):
    """ A strategy interface specific to reinforcement learning
    """
    def __init__(self, knowledge: Knowledge) -> None:
        super().__init__()
        self._knowledge = knowledge

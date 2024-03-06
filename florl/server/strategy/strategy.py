from abc import ABC
from typing import Dict, List, Tuple

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from florl.common import Knowledge


class FlorlStrategy(Strategy, ABC):
    """A strategy interface specific to reinforcement learning"""

    def __init__(self, knowledge: Knowledge) -> None:
        super().__init__()
        self._knowledge = knowledge

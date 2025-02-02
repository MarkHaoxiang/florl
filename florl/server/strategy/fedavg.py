from typing import Callable, Tuple
from logging import WARNING

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server import strategy
from flwr.server.strategy import aggregate
from flwr.common import (
    Scalar,
    FitRes,
    GetParametersIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from florl.common import StateDict


class FedAvg(strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        return super().aggregate_fit(server_round, results, failures)

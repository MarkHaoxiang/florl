from abc import ABC
from typing import Dict, List, Tuple
from flwr.server.client_manager import ClientManager

from flwr.server.strategy import Strategy
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, GetParametersIns
from flwr.server.client_proxy import ClientProxy

from florl.common import Knowledge


class FlorlStrategy(Strategy, ABC):
    """A strategy interface specific to reinforcement learning"""

    def __init__(self, knowledge: Knowledge) -> None:
        super().__init__()
        self._knowledge = knowledge

class AggregateFitWrapper(Strategy):
    """ Hack for adapting classical FL strategy
    """
    def __init__(self, strategy: Strategy) -> None:
        self.strategy = strategy
    
    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy | FitRes]],
                      failures: List[Tuple[ClientProxy, FitRes] | BaseException]
        ) -> Tuple[Parameters | Dict[str, bool | bytes | float | int | str] | None]:
        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(server_round, results, failures)
        if len(results) > 0:
            parameters_aggregated.tensor_type = results[0][1].parameters.tensor_type
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy | EvaluateRes]], failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException]) -> Tuple[float | Dict[str, bool | bytes | float | int | str] | None]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy | EvaluateIns]]:
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy | FitIns]]:
        return self.strategy.configure_fit(server_round, parameters, client_manager)
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, bool | bytes | float | int | str]] | None:
        return self.strategy.evaluate(server_round, parameters)
    
    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.strategy.initialize_parameters(client_manager)

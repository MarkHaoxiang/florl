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

from florl.common import Knowledge, KnowledgeShard
from florl.server.strategy import FlorlStrategy


class FedAvg(strategy.FedAvg, FlorlStrategy):
    """Custom FedAvg with adaptation to Florl specific features"""

    def __init__(
        self, knowledge: Knowledge, evaluate_fn: Callable | None = None, *args, **kwargs
    ):
        """Custom FedAvg with adaptation to Florol specific features

        Args:

            knowledge (Knowledge): Knowledge representation used by the agent.
            args, kwargs (Any): Pass in arguments to FedAvg
        """
        FedAvg.__init__(self, *args, **kwargs)
        FlorlStrategy.__init__(self, knowledge)
        self._evaluate_fn = evaluate_fn

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate fit results with weighted average (client provided weights)

        num_examples is just a generalised client weighting suggestion, doesn't have a RL correspondence
        """
        # From FedAvg
        # Do not aggregate if there are failures and failures are not accepted
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # ... Custom ...
        # Split FitRes into Modules
        shards_registry: dict[str, list[Tuple[KnowledgeShard, int]]] = {}
        for _, fit_res in results:
            fit_res: FitRes = fit_res
            shards = Knowledge.unpack(fit_res.parameters)
            for shard in shards:
                if shard.name not in shards_registry:
                    shards_registry[shard.name] = []
                shards_registry[shard.name].append((shard, fit_res.num_examples))
        # Combine each module individually
        aggregated_shards: dict[str, KnowledgeShard] = {}
        for name, shards in shards_registry.items():
            if self.inplace:
                raise NotImplementedError("Inplace not tested.")
            weights_results = [
                (parameters_to_ndarrays(shard.parameters), num_examples)
                for (shard, num_examples) in shards
            ]
            aggregated_ndarrays = aggregate.aggregate(weights_results)
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
            aggregated_shards[name] = KnowledgeShard(
                name=name, parameters=parameters_aggregated
            )
        # Update own knowledge base
        self._knowledge.update_knowledge(shards=list(aggregated_shards.values()))
        parameters_aggregated = self._knowledge.get_parameters(
            GetParametersIns({})
        ).parameters

        # From FedAvg
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Tuple[float, dict[str, Scalar]] | None:
        """Evaluate model parameters using an evaluation function"""
        if self._evaluate_fn is None:
            return None
        return self._evaluate_fn(server_round, parameters)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        return super().configure_evaluate(server_round, parameters, client_manager)

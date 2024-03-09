import os
import pickle
import time
from typing import Callable, List, Tuple, OrderedDict, Optional
from collections import defaultdict
import numbers
import flwr
from flwr.common import (
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
)
from flwr.client import Client, ClientFn

from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.history import History
from flwr.server.client_manager import ClientManager


from multiprocessing import Process
from multiprocessing.pool import Pool

import numpy as np
import torch


class StatefulClient(Client):
    """A wrapper to enabled pickled client states on disk as an alternative to Context for stateful execution

    # TODO: Add encryption to ensure security. But really this entire thing is a hack, how to save complex state is worth discussing.

    # TODO: This breaks has_get_... in flower.client.py - but that's more of a hack on Flower's side.
    """

    def __init__(
        self, cid: str, client_fn: Callable[[str], Client], ws: str = "florl_ws"
    ):
        cid = str(cid)

        # Create ws if not exists
        if not os.path.exists(ws):
            os.makedirs(ws)

        self._client_path = os.path.join(ws, f"{cid}.client")
        if not os.path.exists(self._client_path):
            # Create client if not exists
            self._client = client_fn(cid)
            self.save_client()
        else:
            self.load_client()

    def save_client(self):
        pickle.dump(self._client, open(self._client_path, "wb"))

    def load_client(self):
        self._client = pickle.load(open(self._client_path, "rb"))

    def get_context(self) -> Context:
        return self._client.get_context()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._client.get_parameters(ins)

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return self._client.get_properties(ins)

    def fit(self, ins: FitIns) -> FitRes:
        result = self._client.fit(ins)
        self.save_client()
        return result

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self._client.evaluate(ins)


def stateful_client(
    client_fn: Callable[[str], Client], ws: str = "florl_ws"
) -> Callable[[str], Client]:
    """Wraps a client constructor to a StatefulClient constructor

    Args:
        client_fn (Callable[[str], Client]): Builds a client from cid.
        ws (str, optional): Directory to save contexts. Defaults to "florl_ws".

    Returns:
        Callable[[str], Client]: Builds a stateful client from cid.
    """
    return lambda cid: StatefulClient(cid=cid, client_fn=client_fn, ws=ws)


_start_server = flwr.server.start_server


def _start_client(cid: int, **kwargs):
    """
    This needs to be top level for multiprocessing to work
    """
    client = kwargs["client_fn"](str(cid))
    kwargs.pop("client_fn")
    flwr.client.start_client(client=client, **kwargs)


def start_stateful_simulation(
    client_fn: ClientFn,
    num_clients: int,
    config: ServerConfig,
    strategy: Strategy,
    client_manager: Optional[ClientManager] = None,
    server_addr: str = "0.0.0.0:8080",
) -> History:
    server_args = {
        "server_address": server_addr,
        "strategy": strategy,
        "config": config,
        "client_manager": client_manager,
    }

    client_args = {"server_address": server_addr, "client_fn": client_fn}

    processes = []

    # server_process = Process(target=_start_server, kwargs=server_args)
    # server_process.start()
    server_pool = Pool()
    result = server_pool.apply_async(func=_start_server, kwds=server_args)

    # processes.append(server_process)
    time.sleep(3)  # TODO: ideally we want to trigger based on server launch events

    for cid in range(num_clients):
        client_process = Process(target=_start_client, args=(cid,), kwargs=client_args)
        # client_process = Process(target=_start_client, kwargs=client_args)
        client_process.start()
        processes.append(client_process)

    server_pool.close()
    server_pool.join()
    for p in processes:
        p.join()
        p.close()
    return result.get()
    # server_process.join()


# ==========
# Functions from https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html and the labs


def set_torch_parameters(net: torch.nn.Module, parameters: NDArrays):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_torch_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def aggregate_weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Generic function to combine results from multiple clients
    following training or evaluation.

    Args:
        metrics (List[Tuple[int, dict]]): collected clients metrics

    Returns:
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    other_dict: dict = defaultdict(list)

    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
            else:
                other_dict[key].append(val)
        total_examples += num_examples
    aggregated_metrics = {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }
    result = aggregated_metrics | other_dict
    return result


import os
import pickle
from typing import Callable, List, Tuple, OrderedDict
from collections import defaultdict
import numbers
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, NDArrays
from flwr.client import Client

import numpy as np
import torch


class StatefulClient(Client):
    """A wrapper to enabled pickled client states on disk as an alternative to Context for stateful execution

    # TODO: Add encryption to ensure security. But really this entire thing is a hack, how to save complex state is worth discussing.
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
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }

import os
import pickle
from logging import INFO, WARNING
from typing import Any, List, Tuple, Union, TypeAlias

import torch
import numpy as np
import gymnasium as gym
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Properties,
    Status,
    Config,
    Parameters,
)

from florl.client import FlorlClient
from florl.client.kitten import KittenClientWrapper
try:
    from strategy import RlFedAvg
except ModuleNotFoundError:
    from .strategy import RlFedAvg

CFG_FIT: TypeAlias = List[Tuple[ClientProxy, FitIns]]
RES_FIT: TypeAlias = List[Tuple[ClientProxy, FitRes]]
CFG_EVAL: TypeAlias = List[Tuple[ClientProxy, EvaluateIns]]
RES_FIT: TypeAlias = List[Tuple[ClientProxy, EvaluateRes]]
FAILURES: TypeAlias = List[Union[Tuple[ClientProxy, FitRes], BaseException]]

# This is a default
replay_buffer_workspace = "florl_ws"


def get_evaluation_fn(evaluation_client: FlorlClient):
    """Utility to to set centralised evaluation

    Args:
        evaluation_client (FlorlClient): client used to run evaluation rounds.
    """

    def evaluate(server_rounds: int, parameters: Parameters):
        ins = EvaluateIns(parameters=parameters, config={})
        evaluation_result = evaluation_client.evaluate(ins)
        return evaluation_result.loss, evaluation_result.metrics

    return evaluate


def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
    """Override of get_properties function;
    currently gets replay buffer from Kitten Clients"""
    results: Properties = {}
    if (cid := ins.config.get("cid")) is None:
        return GetPropertiesRes(
            status=Status(
                code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
                message="Please Pass in CID",
            ),
            properties={},
        )
    if ins.config.get("replay_buffer", False) and self._memory is not None:
        replay_path = os.path.join(
            replay_buffer_workspace, f"replay_{cid}_{self.step}.pt"
        )
        log(INFO, "new get property called on client side")
        torch.save(self._memory.storage, replay_path)
        results["cid"] = cid
        results["replay_buffer"] = replay_path

    # Add more properties if needed

    return GetPropertiesRes(
        status=Status(code=Code.OK, message="Successfully"), properties=results
    )


class EvalReplayFedAvg(RlFedAvg):
    """Fed Average which also stores replay buffers (objects) to History
    
    Deprecated. Use memory client instead.
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: RES_FIT,
        failures: FAILURES,
    ):
        parameters_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        property_request = {"replay_buffer": True, "cid": -1}
        for client_proxy, _ in results:
            # TODO: handle failed clients
            config = property_request.copy()
            config["cid"] = client_proxy.cid
            prop = client_proxy.get_properties(
                GetPropertiesIns(config=config),
                timeout=1000,
                group_id=client_proxy.cid,  # For some reason RayActorClient needs a group id
            )
            log(
                WARNING,
                f"Get Properties Request got {prop.status.code} : {prop.status.message}",
            )
            # FIXME: once we get the simulation to handle get_properties
            # correctly, uncomment this
            if metrics_aggregated.get("replay_buffer") is None:
                metrics_aggregated["replay_buffer"] = []
            metrics_aggregated["replay_buffer"].append(
                (
                    server_round,
                    client_proxy.cid,
                    prop.properties.get("replay_buffer", ""),
                )
            )
        return parameters_aggregated, metrics_aggregated

class MemoryClient(KittenClientWrapper):
    """ Records the memory state
    """
    def train(self, train_config: Config):
        n, metrics =  super().train(train_config)

        seen_states = self._client._memory.storage[0]

        metrics["id"] = self._client._seed
        metrics["rb"] = pickle.dumps(seen_states)
        metrics["rb_size"] = len(self._client._memory)

        frames = train_config["frames"]
        if frames <= self._client._memory.capacity:
            # Get newly collected frames
            append_index = self._client._memory._append_index
            newly_collected_frames = seen_states[max(0, append_index-frames):append_index]
            if append_index-frames < 0:
                newly_collected_frames = torch.cat((
                    newly_collected_frames,
                    seen_states[append_index-frames:]
                ))
            metrics["rb_new"] = pickle.dumps(newly_collected_frames)
        return n, metrics

class FixedResetWrapper(gym.Wrapper):
    """ Fix the reset to a single state
    """
    def __init__(self, env: gym.Env):
        """ Fix the reset to a single state
        """
        super().__init__(env)
        self._seed = np.random.rand()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[Any | dict[str, Any]]:        
        if seed is not None:
            self._seed = seed
        return super().reset(seed=self._seed, options=options)

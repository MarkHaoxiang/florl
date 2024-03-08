import os
from typing import List, Tuple, Union

import torch

from logging import WARNING
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Properties,
    Status,
)

from strategy import RlFedAvg


CFG_FIT = List[Tuple[ClientProxy, FitIns]]
RES_FIT = List[Tuple[ClientProxy, FitRes]]
CFG_EVAL = List[Tuple[ClientProxy, EvaluateIns]]
RES_FIT = List[Tuple[ClientProxy, EvaluateRes]]
FAILURES = List[Union[Tuple[ClientProxy, FitRes], BaseException]]

# This is a default
replay_buffer_workspace = "florl_ws"


def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
    """Override of get_properties function;
    currently gets replay buffer from Kitten Clients"""
    results: Properties = {}
    if cid := ins.config.get("cid") is None:
        return GetPropertiesRes(
            status=Status(
                code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
                message="Please Pass in CID",
            ),
            properties={},
        )
    if ins.config.get("replay_buffer", False) and self._memory is not None:
        replay_path = os.path.join(
            replay_buffer_workspace, f"replay_{cid}_{self.step()}.pt"
        )
        log(WARNING, "new get property called on client side")
        torch.save(self._memory.storage, replay_path)
        results["cid"] = cid
        results["replay_buffer"] = replay_path

    # Add more properties if needed

    return GetPropertiesRes(
        status=Status(code=Code.OK, message="Successfully"), properties=results
    )


class EvalReplayFedAvg(RlFedAvg):
    """Fed Average which also stores replay buffers (objects) to History"""

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

    ## Test if this work
    # def configure_evaluate(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ):
    #     super_res = super().configure_evaluate(server_round, parameters, client_manager)
    #     property_request = {"replay_buffer": True, "cid": -1}
    #     for client_proxy, _ in super_res:
    #         # TODO: handle failed clients
    #         config = property_request.copy()
    #         config["cid"] = client_proxy.cid
    #         prop = client_proxy.get_properties(
    #             GetPropertiesIns(config=config), 50, client_proxy.cid
    #         )
    #         log(
    #             WARNING,
    #             f"Get Properties Request got {prop.status.code} : {prop.status.message}",
    #         )
    #         self.pickeled_paths.append(
    #             (
    #                 server_round,
    #                 client_proxy.cid,
    #                 prop.properties.get("replay_buffer", ""),
    #             )
    #         )
    #     return super_res

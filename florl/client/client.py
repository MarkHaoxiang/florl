from abc import ABC
from typing import Dict, Tuple

import gymnasium as gym
import flwr as fl
from flwr.common.typing import (
    Parameters,
    GetParametersIns,
    GetParametersRes,
    Config,
    Status,
    Code,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Scalar,
)

from florl.common.knowledge import Knowledge


class FlorlClient(fl.client.Client, ABC):
    """A client interface specific to reinforcement learning"""

    def __init__(self, knowledge: Knowledge):
        self._knowl = knowledge

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._knowl.get_parameters(ins)

    def fit(self, ins: FitIns) -> FitRes:
        if len(ins.parameters.tensors) > 0:
            self._knowl.set_parameters(ins.parameters)
        try:
            n, metrics = self.train(ins.config)
            parameters_res = self.get_parameters(
                GetParametersIns(ins.config.get("get_parameters", {}))
            )
            return FitRes(
                status=Status(Code.OK, ""),
                num_examples=n,
                parameters=parameters_res.parameters,
                metrics=metrics,
            )
        except NotImplementedError:
            return FitRes(
                status=Status(Code.FIT_NOT_IMPLEMENTED, "Train is not implemented"),
                num_examples=0,
                parameters=Parameters(tensor_type="", tensors=[]),
                metrics={},
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self._knowl.set_parameters(ins.parameters)
        try:
            n, metrics = self.epoch(ins.config)
            return EvaluateRes(
                status=Status(Code.OK, ""), loss=0.0, num_examples=n, metrics=metrics
            )
        except NotImplementedError:
            return EvaluateRes(
                status=Status(
                    code=Code.EVALUATE_NOT_IMPLEMENTED,
                    message="Client does not implement `epoch`",
                ),
                loss=0.0,
                num_examples=0,
                metrics={},
            )

    # =========
    # Overide these two methods
    # =========

    def train(self, config: Config) -> Tuple[int, Dict[str, Scalar]]:
        """Runs a reinforcement learning algorithm to update local knowledge

        Args:
            config: Training Configuration
        """
        raise NotImplementedError

    def epoch(self, config: Config) -> Tuple[int, Dict[str, Scalar]]:
        """An epoch represents a evaluation run

        Args:
            config (Config): Evaluation configuration
        """
        raise NotImplementedError


class GymClient(FlorlClient, ABC):
    """An RL client training within a gym environment"""

    def __init__(self, knowledge: Knowledge, env: gym.Env, seed: int | None = None):
        super().__init__(knowledge)
        self._env = env
        self._seed = seed

        self._env.reset(seed=self._seed)

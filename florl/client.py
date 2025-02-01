from abc import ABC, abstractmethod

import flwr as fl
from flwr.common import (
    Code,
    Config,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
    Metrics,
)
import torch.nn as nn
from torchrl.envs import EnvBase

from florl.common import StateDict, get_torch_parameters, load_torch_parameters
from florl.common.util import fit_ok


class FlorlClient(fl.client.Client, ABC):
    """A client interface specific to reinforcement learning"""

    # =========
    # Overide these methods
    # =========

    @abstractmethod
    @property
    def parameter_container(self) -> nn.Module:
        """Returns a module containing all parameters available in the client. This encapsulates policies, values, loss modules etc.

        Returns:
            nn.Module: A module containing any submodules used.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(
        self, parameters: StateDict, config: Config
    ) -> tuple[int, StateDict, Metrics]:
        """Trains the client's model using the provided parameters and configuration.

        This method is responsible for performing the training loop on the client's local data.
        It updates the model's parameters based on the provided `parameters` and returns
        the updated parameters along with training metrics.

        Args:
            parameters (StateDict): A dictionary containing the model parameters to be used
                                   for training. These parameters are typically provided by
                                   the server.
            config (Config): A dictionary containing configuration parameters for the training
                            process (e.g., learning rate, batch size, number of epochs).

        Returns:
            tuple[int, StateDict, Metrics]: A tuple containing:
                - `int`: The number of training examples used during the training process.
                - `StateDict`: The updated model parameters after training.
                - `Metrics`: A dictionary of metrics (e.g., loss, accuracy) collected during
                            the training process.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError()

    # =========

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        ignore_prefix = ins.config.get("ignore_prefix", ())
        assert isinstance(ignore_prefix, tuple)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=get_torch_parameters(self.parameter_container, ignore_prefix),
        )

    def fit(self, ins: FitIns) -> FitRes:
        try:
            ignore_prefix = ins.config.get("ignore_prefix", ())
            assert isinstance(ignore_prefix, tuple)
            state_dict = load_torch_parameters(ins.parameters)
            num_examples, state_dict, metrics = self.train(state_dict, ins.config)
            parameters = get_torch_parameters(state_dict, ignore_prefix=ignore_prefix)
            return fit_ok(num_examples, parameters, metrics)
        except NotImplementedError:
            return FitRes(
                status=Status(Code.FIT_NOT_IMPLEMENTED, "Not Implemented"),
                num_examples=0,
                parameters=Parameters(tensor_type="", tensors=[]),
                metrics={},
            )


class EnvironmentClient(FlorlClient):
    """A client with access to an RL client.

    Covers the majority of cases, possibly excluding offline reinforcement learning.
    """

    def __init__(self, env: EnvBase):
        super().__init__()
        self._env = env

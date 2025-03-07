from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from flwr.common import (
    Code,
    Config,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    EvaluateIns,
    EvaluateRes,
    Status,
)
from flwr.client import Client as FlwrClient
import torch.nn as nn
from torchrl.envs import EnvBase

from florl.common import (
    StateDict,
    JSONSerializable,
    Metrics,
    get_torch_parameters,
    load_torch_parameters,
    torch_to_numpy_parameters,
    numpy_to_torch_parameters,
)
from florl.common.res import evaluate_ok, fit_ok


class Client(FlwrClient, ABC):
    """A client interface specific to reinforcement learning"""

    # =========
    # Overide these methods
    # =========

    @property
    @abstractmethod
    def parameter_container(self) -> nn.Module:
        """Returns a module containing all parameters available in the client. This encapsulates policies, values, loss modules etc.

        Returns:
            nn.Module: A module containing any submodules used.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(
        self, parameters: StateDict, config: Config
    ) -> tuple[int, StateDict, JSONSerializable]:
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

    def evaluation(
        self, parameters: StateDict, config: Config
    ) -> tuple[int, JSONSerializable]:
        """Evaluates the client's model using the provided parameters and configuration.

        This method is responsible for evaluating the model's performance on the client's local data.
        It uses the provided `parameters` to update the model and returns evaluation metrics.

        Args:
            parameters (StateDict): A dictionary containing the model parameters to be used
                                for evaluation. These parameters are typically provided by
                                the server.
            config (Config): A dictionary containing configuration parameters for the evaluation
                            process (e.g., number of environment steps).

        Returns:
            tuple[int, Metrics]: A tuple containing:
                - `int`: The number of evaluation examples used during the evaluation process.
                - `Metrics`: A dictionary of metrics (e.g., loss, accuracy) collected during
                            the evaluation process.

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
            state_dict = load_torch_parameters(ins.parameters, ignore_prefix)
            num_examples, state_dict, metrics = self.train(state_dict, ins.config)
            parameters = get_torch_parameters(state_dict, ignore_prefix=ignore_prefix)
            return fit_ok(num_examples, parameters, Metrics(metrics).dump())
        except NotImplementedError:
            return super().fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        try:
            ignore_prefix = ins.config.get("ignore_prefix", ())
            assert isinstance(ignore_prefix, tuple)
            state_dict = load_torch_parameters(ins.parameters, ignore_prefix)
            num_examples, metrics = self.evaluation(state_dict, ins.config)
            return evaluate_ok(num_examples, Metrics(metrics).dump())
        except NotImplementedError:
            return super().evaluate(ins)

    def to_numpy(self) -> _NumPyFlorlWrapper[Self]:
        """Converts to a client which communicates with the server via Numpy array parameters.

        This is useful to interface with the default set of Flower strategies.

        Returns:
            _NumPyFlorlWrapper: A client which uses np.ndarray parameters.
        """
        return _NumPyFlorlWrapper(client=self)


class _NumPyFlorlWrapper[T: Client](FlwrClient):
    def __init__(self, client: T):
        super().__init__()
        self.client = client

    def get_properties(self, ins):
        return self.client.get_properties(ins)

    def get_parameters(self, ins):
        res = self.client.get_parameters(ins)
        res.parameters = torch_to_numpy_parameters(res.parameters)
        return res

    def fit(self, ins):
        ins.parameters = numpy_to_torch_parameters(
            ins.parameters, self.client.parameter_container.state_dict()
        )
        res = self.client.fit(ins)
        res.parameters = torch_to_numpy_parameters(res.parameters)
        return res

    def evaluate(self, ins):
        ins.parameters = numpy_to_torch_parameters(
            ins.parameters, self.client.parameter_container.state_dict()
        )
        return self.client.evaluate(ins)


class EnvironmentClient(Client):
    """A client with access to an RL client.

    Covers the majority of cases, possibly excluding offline reinforcement learning.
    """

    def __init__(self, env: EnvBase):
        super().__init__()
        self._env = env

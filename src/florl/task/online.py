from abc import abstractmethod
from typing import Literal

from flwr.common import Context
from flwr.client import Client, ClientApp
from flwr.server import ServerAppComponents, ServerApp
from torchrl.envs import EnvBase

from florl.client import EnvironmentClient
from florl.task.abc import Algorithm, Task, Experiment, SimulationExperiment


class OnlineAlgorithm(Algorithm):
    @abstractmethod
    def make_client(
        self, train_env: EnvBase, eval_env: EnvBase, reference_env: EnvBase
    ) -> EnvironmentClient:
        raise NotImplementedError()

    @abstractmethod
    def make_server(self) -> ServerAppComponents:
        raise NotImplementedError()


class OnlineTask(Task[OnlineAlgorithm]):
    @abstractmethod
    def create_env(self, mode: Literal["train", "evaluate", "reference"]) -> EnvBase:
        raise NotImplementedError()

    def compile(self, algorithm: OnlineAlgorithm) -> Experiment:
        def client_fn(context: Context) -> Client:
            train_env = self.create_env(mode="train")
            eval_env = self.create_env(mode="evaluate")
            reference_env = self.create_env(mode="reference")

            client = algorithm.make_client(
                train_env=train_env, eval_env=eval_env, reference_env=reference_env
            )
            return client.to_numpy()

        client_app = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
            return algorithm.make_server()

        server_app = ServerApp(server_fn=server_fn)

        return SimulationExperiment(
            client_app=client_app,
            server_app=server_app,
        )

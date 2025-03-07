from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from flwr.common import Context
from flwr.simulation import run_simulation
from flwr.client import Client, ClientApp
from flwr.server import Server, ServerApp
from torchrl.envs import EnvBase

from florl.client import EnvironmentClient

Config = BaseModel


class Experiment[C: Config](ABC):
    def __init__(self, client_app: ClientApp, server_app: ServerApp):
        super().__init__()
        self.client_app = client_app
        self.server_app = server_app

    @abstractmethod
    def run(self, config: C) -> None:
        raise NotImplementedError()


class _SimulationConfig(Config):
    num_supernodes: int


class SimulationConfig(Config):
    simulation: _SimulationConfig


class SimulationExperiment[C: SimulationConfig](Experiment[C]):
    def run(self, config: C) -> None:
        run_simulation(
            server_app=self.server_app,
            client_app=self.client_app,
            num_supernodes=config.simulation.num_supernodes,
            backend_name="ray",
        )


class Algorithm[C: Config](ABC):
    pass


class Task[A: Algorithm, C: Config](ABC):
    @abstractmethod
    def compile(self, algorithm: A) -> Experiment:
        raise NotImplementedError()


class OnlineAlgorithm[C: Config](Algorithm[C]):
    @abstractmethod
    def make_client(
        self, train_env: EnvBase, eval_env: EnvBase, reference_env: EnvBase
    ) -> EnvironmentClient:
        raise NotImplementedError()

    @abstractmethod
    def make_server(self) -> Server:
        raise NotImplementedError()


class OnlineTask[C: Config](Task[OnlineAlgorithm, C]):
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
        raise NotImplementedError()

from abc import ABC, abstractmethod

from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp

from florl.common import Config


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


class Algorithm(ABC):
    pass


class Task[A: Algorithm](ABC):
    @abstractmethod
    def compile(self, algorithm: A) -> Experiment:
        """
        Compiles the task using the provided algorithm and returns an Experiment instance.

        Args:
            algorithm (A): The algorithm instance that specifies how to compile the task.

        Returns:
            Experiment: A simulation experiment instance configured with client and server apps.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError()


class Benchmark[A: Algorithm]:
    def __init__(self, tasks: list[Task[A]]):
        super().__init__()
        self.tasks = tasks

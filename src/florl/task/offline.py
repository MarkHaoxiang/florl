from abc import abstractmethod
from typing import Any

from flwr.common import Context
from flwr.server import ServerAppComponents, ServerApp
from flwr.client import Client as FlwrClient, ClientApp
import torch
from torchrl.envs import EnvBase
from torchrl.data.datasets import BaseDatasetExperienceReplay
from torchrl.data.replay_buffers import Sampler, Storage

from florl.client import Client
from florl.task.abc import Algorithm, Task, Experiment, SimulationExperiment


type Partition = list[int] | torch.Tensor  # Indexes of the storage


class OfflineAlgorithm(Algorithm):
    @abstractmethod
    def make_client(
        self,
        dataset: BaseDatasetExperienceReplay,
        eval_env: EnvBase | None,
        reference_env: EnvBase,
    ) -> Client:
        raise NotImplementedError()

    @abstractmethod
    def make_server(self) -> ServerAppComponents:
        raise NotImplementedError()


class OfflineTask(Task[OfflineAlgorithm]):
    @abstractmethod
    def create_reference_env(self) -> EnvBase:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, node_id: int) -> BaseDatasetExperienceReplay:
        raise NotImplementedError()

    def create_eval_env(self) -> EnvBase | None:
        return None

    def compile(self, algorithm: OfflineAlgorithm) -> Experiment:
        def client_fn(context: Context) -> FlwrClient:
            eval_env = self.create_eval_env()
            reference_env = self.create_reference_env()

            dataset = self.load_dataset(context.node_id)

            client = algorithm.make_client(
                dataset=dataset, eval_env=eval_env, reference_env=reference_env
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


class RandomPartitionSamplerWithoutReplacement(Sampler):
    def __init__(self, partition: Partition):
        """Randomly sample from indices given in 'partition', without replacement.

        Since datasets are memory mapped, we can use the partition as a surrogate for separate datasets.

        Args:
            partition (Partition): A list of indices which belongs to the client.
        """
        super().__init__()
        if isinstance(partition, torch.Tensor):
            self._partition = partition.to(device="cpu")
        else:
            self._partition = torch.tensor(partition)
        self._n = len(self._partition)
        self._sample_list = torch.randperm(self._n, generator=self._rng)
        self._current_idx = 0

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        """Sample a batch of data from the provided storage.

        Automatically wraps around the sample list if the end is reached.

        Args:
            storage (Storage): The storage to sample from.
            batch_size (int): The number of samples to retrieve.

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing indices to sample from the storage and an empty dict.
        """
        if len(storage) == 0:
            raise ValueError("Storage is empty, cannot sample from it.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if batch_size > len(self._partition):
            raise ValueError("Batch size exceeds partition size.")

        if self._current_idx + batch_size >= self._n:
            # Wrap around
            prev_idxs = self._sample_list[self._current_idx :]
            new_idxs = self._sample_list[: self._current_idx + batch_size - self._n]
            idxs = torch.cat([prev_idxs, new_idxs])
            # Reinitialize the sample list
            self._sample_list = torch.randperm(self._n, generator=self._rng)
            self._current_idx = self._current_idx + batch_size - self._n
        else:
            idxs = self._sample_list[self._current_idx : self._current_idx + batch_size]
            self._current_idx += batch_size
        return self._partition[idxs], {}

    def __len__(self) -> int:
        return self._n

    def _empty(self):
        pass

    def dumps(self, path): ...

    def loads(self, path): ...

    def state_dict(self) -> dict[str, Any]:
        return {
            "partition": self._partition,
            "current_idx": self._current_idx,
            "sample_list": self._sample_list,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._partition = state_dict["partition"]
        self._current_idx = state_dict["current_idx"]
        self._sample_list = state_dict["sample_list"]
        self._n = len(self._partition)


# TODO: Slice sampler version

from abc import ABC, abstractmethod
import copy
from typing import Tuple, Dict
from flwr.common import Context, GetPropertiesIns, GetPropertiesRes

from flwr.common.typing import GetParametersIns, GetParametersRes, Scalar, Config
from gymnasium.core import Env
import torch
import kitten

from florl.client import GymClient
from florl.common import Knowledge


class KittenClient(GymClient, ABC):
    """A client conducting training and evaluation with the Kitten RL library"""

    def __init__(
        self,
        knowledge: Knowledge,
        env: Env,
        config: Config,
        seed: int | None = None,
        build_memory: bool = False,
        enable_evaluation: bool = True,
        device: str = "cpu",
    ):
        super().__init__(knowledge, env, seed, enable_evaluation)

        self._cfg = copy.deepcopy(config)
        self._device = device

        # Logging
        self._evaluator = kitten.logging.KittenEvaluator(
            env=self._env, device=self._device, **self._cfg.get("evaluation", {})
        )
        self._rng = kitten.common.global_seed(seed)

        # RL Modules
        self._step = 0
        self.build_algorithm()
        self._memory = None
        if build_memory:
            self._memory: kitten.experience.memory.ReplayBuffer = (
                kitten.experience.util.build_replay_buffer(
                    env=self._env, device=self._device, **self._cfg.get("memory", {})
                )
            )

        self._collector = kitten.experience.util.build_collector(
            policy=self.policy, env=self._env, memory=self._memory, device=self._device
        )
        self.early_start()

    def epoch(self, config: Config) -> Tuple[int, Dict[str, Scalar]]:
        torch.manual_seed(self._rng.numpy.integers(0, 65535))
        repeats = config.get("evaluation_repeats", self._evaluator.repeats)
        reward = self._evaluator.evaluate(self.policy, repeats)
        return repeats, {"reward": reward}

    @property
    def step(self) -> int:
        """Number of collected frames"""
        return self._step

    @abstractmethod
    def build_algorithm(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm(self) -> kitten.rl.Algorithm:
        raise NotImplementedError

    @property
    @abstractmethod
    def policy(self) -> kitten.policy.Policy:
        raise NotImplementedError

    def early_start(self):
        pass

class KittenClientWrapper(KittenClient):
    """ Utility to quickly wrap a client
    """
    def __init__(self, client: KittenClient):
        self._client = client
        self._knowl = self._client._knowl
        self._enable_evaluation = self._client._enable_evaluation

    def build_algorithm(self) -> None:
        return self._client.build_algorithm()
    
    def early_start(self) -> None:
        return self._client.early_start()

    def train(self, config: Config):
        return self._client.train(config)
 
    def epoch(self, config: Config):
        return self._client.epoch(config)

    def get_context(self) -> Context:
        return self._client.get_context()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._client.get_parameters(ins)

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return self._client.get_properties(ins)

    @property
    def policy(self) -> kitten.policy.Policy:
        return self._client.policy
    
    @property
    def algorithm(self) -> kitten.rl.Algorithm:
        return self._client.algorithm
    
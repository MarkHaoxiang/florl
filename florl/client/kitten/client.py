from abc import ABC, abstractmethod
import copy
from flwr.common.typing import Config, EvaluateRes

from gymnasium.core import Env
from florl.client import GymClient
from florl.common import Knowledge
import kitten

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
        device: str = "cpu",
    ):
        super().__init__(knowledge, env, seed)

        self._cfg = copy.deepcopy(config)
        self._device = device

        # Logging
        self._evaluator = kitten.logging.KittenEvaluator(
            env=self._env, device=self._device, **self._cfg.get("evaluation", {})
        )

        # RL Modules
        self._step = 0
        self.build_algorithm()
        self._memory = None
        if build_memory:
            self._memory = kitten.experience.util.build_replay_buffer(
                env=self._env, device=self._device, **self._cfg.get("memory", {})
            )

        self._collector = kitten.experience.util.build_collector(
            policy=self.policy, env=self._env, memory=self._memory, device=self._device
        )
        self.early_start()

    def epoch(self, config: Config) -> EvaluateRes:
        repeats = config.get("evaluation_repeats", self._evaluator.repeats)
        reward = self._evaluator.evaluate(self.policy, repeats)
        # TODO: What is loss under this framework? API shouldn't enforce returning this
        return 0.0, repeats, {"reward": reward}

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

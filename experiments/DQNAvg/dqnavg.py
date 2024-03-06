import copy

import gymnasium as gym
import torch
from flwr.common import Config
from flwr.common.typing import GetParametersIns, Parameters
import kitten
from kitten.rl.dqn import DQN
from kitten.common.util import build_env, build_critic

from florl.common import NumPyKnowledge
from florl.common.util import get_torch_parameters, set_torch_parameters
from florl.client.kitten import KittenClient

class DQNKnowledge(NumPyKnowledge):
    def __init__(self, critic: torch.nn.Module) -> None:
        super().__init__(["critic", "critic_target"])
        self.critic = kitten.nn.AddTargetNetwork(copy.deepcopy(critic))

    @property
    def torch_modules_registry(self):
        return {
            "critic": self.critic.net,
            "critic_target": self.critic.target
        }

class DQNClient(KittenClient):
    def __init__(
        self,
        knowledge: DQNKnowledge,
        env: gym.Env,
        config: Config,
        seed: int | None = None,
        device: str = "cpu",
    ):
        super().__init__(knowledge, env, config, seed, True, device)
        self._knowl: DQNKnowledge = self._knowl # Typing hints

    # Algorithm
    def build_algorithm(self) -> None:
        self._cfg.get("algorithm", {}).pop("critic", None)
        self._algorithm = DQN(
            critic=self._knowl.critic.net,
            device=self._device,
            **self._cfg.get("algorithm", {}),
        )
        self._policy = kitten.policy.EpsilonGreedyPolicy(
            fn=self.algorithm.policy_fn,
            action_space=self._env.action_space,
            rng=self._rng.numpy,
            device=self._device,
        )
        # Synchronisation
        self._algorithm._critic = self._knowl.critic

    @property
    def algorithm(self) -> kitten.rl.Algorithm:
        return self._algorithm

    @property
    def policy(self) -> kitten.policy.Policy:
        return self._policy

    # Training
    def early_start(self):
        self._collector.early_start(n=self._cfg["train"]["initial_collection_size"])

    def train(self, train_config: Config):
        metrics = {}
        # Synchronise critic net
        critic_loss = []
        # Training
        for _ in range(train_config["frames"]):
            self._step += 1
            # Collected Transitions
            self._collector.collect(n=1)
            batch, aux = self._memory.sample(self._cfg["train"]["minibatch_size"])
            batch = kitten.experience.Transition(*batch)
            # Algorithm Update
            critic_loss.append(self.algorithm.update(batch, aux, self.step))

        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return len(self._memory), metrics


class DQNClientFactory:
    def __init__(self, config: Config, device: str = "cpu") -> None:
        self.env = build_env(**config["rl"]["env"])
        self.net = build_critic(
            env=self.env, **config.get("rl", {}).get("algorithm", {}).get("critic", {})
        )
        self.device = device

    def create_dqn_client(self, cid: int, config: Config) -> DQNClient:
        env = copy.deepcopy(self.env)
        net = copy.deepcopy(self.net)

        knowledge = DQNKnowledge(net)
        client = DQNClient(
            knowledge=knowledge, env=env, config=config, seed=cid, device=self.device
        )
        return client

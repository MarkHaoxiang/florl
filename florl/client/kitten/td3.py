import copy

import torch
import gymnasium as gym
from flwr.common import Config
import kitten
from kitten.rl.td3 import TwinDelayedDeepDeterministicPolicyGradient
from kitten.common.util import build_env, build_critic, build_actor

from florl.client import FlorlFactory
from florl.common import Knowledge, NumPyKnowledge
from .client import KittenClient


class TD3Knowledge(NumPyKnowledge):
    def __init__(
        self,
        critic_1: kitten.nn.Critic,
        critic_2: kitten.nn.Critic,
        actor: kitten.nn.Actor,
    ) -> None:
        super().__init__(
            [
                "critic_1",
                "critic_target_1",
                "critic_2",
                "critic_target_2",
                "actor",
                "actor_target",
            ]
        )
        self.critic_1 = kitten.nn.AddTargetNetwork(copy.deepcopy(critic_1))
        self.critic_2 = kitten.nn.AddTargetNetwork(copy.deepcopy(critic_2))
        self.actor = kitten.nn.AddTargetNetwork(copy.deepcopy(actor))

    @property
    def torch_modules_registry(self):
        return {
            "critic_1": self.critic_1.net,
            "critic_target_1": self.critic_1.target,
            "critic_2": self.critic_2.net,
            "critic_target_2": self.critic_2.target,
            "actor": self.actor.net,
            "actor_target": self.actor.target,
        }


class TD3Client(KittenClient):
    def __init__(
        self,
        knowledge: TD3Knowledge,
        env: gym.Env,
        config: Config,
        seed: int | None = None,
        enable_evaluation: bool = True,
        device: str = "cpu",
    ):
        super().__init__(knowledge, env, config, seed, True, enable_evaluation, device)
        self._knowl: TD3Knowledge = self._knowl  # Type hints

    # Algorithm
    def build_algorithm(self) -> None:
        self._cfg.get("algorithm", {}).pop("critic", None)
        self._cfg.get("algorithm", {}).pop("actor", None)
        env_action_scale = (
            torch.tensor(
                self._env.action_space.high - self._env.action_space.low,
                device=self._device,
            )
            / 2.0
        )
        env_action_min = torch.tensor(
            self._env.action_space.low, dtype=torch.float32, device=self._device
        )
        env_action_max = torch.tensor(
            self._env.action_space.high, dtype=torch.float32, device=self._device
        )
        self._algorithm = TwinDelayedDeepDeterministicPolicyGradient(
            actor_network=self._knowl.actor.net,
            critic_1_network=self._knowl.critic_1.net,
            critic_2_network=self._knowl.critic_2.net,
            env_action_scale=env_action_scale,
            env_action_min=env_action_min,
            env_action_max=env_action_max,
            **self._cfg.get("algorithm", {}),
        )
        self._policy = kitten.policy.EpsilonGreedyPolicy(
            fn=self.algorithm.policy_fn,
            action_space=self._env.action_space,
            rng=self._rng.numpy,
            device=self._device,
        )
        # Synchronisation
        self._algorithm._critic_1 = self._knowl.critic_1
        self._algorithm._critic_2 = self._knowl.critic_2

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
        train_loss = []
        # Training
        for _ in range(train_config["frames"]):
            self._step += 1
            # Collected Transitions
            self._collector.collect(n=1)
            batch, aux = self._memory.sample(self._cfg["train"]["minibatch_size"])
            batch = kitten.experience.Transition(*batch)
            # Algorithm Update
            train_loss.append(sum(self.algorithm.update(batch, aux, self.step)))

        # Logging
        metrics["loss"] = sum(train_loss) / len(train_loss)
        return len(self._memory), metrics


class TD3ClientFactory(FlorlFactory):
    def __init__(self, config: Config, device: str = "cpu") -> None:
        self.env = build_env(**config["rl"]["env"])
        self.net_1 = build_critic(
            env=self.env, **config.get("rl", {}).get("algorithm", {}).get("critic", {})
        )
        self.net_2 = build_critic(
            env=self.env, **config.get("rl", {}).get("algorithm", {}).get("critic", {})
        )
        self.actor_net = build_actor(
            env=self.env, **config.get("rl", {}).get("algorithm", {}).get("actor", {})
        )
        self.device = device

    def create_default_knowledge(self, config: Config) -> Knowledge:
        net_1 = copy.deepcopy(self.net_1)
        net_2 = copy.deepcopy(self.net_2)
        net_3 = copy.deepcopy(self.actor_net)
        knowledge = TD3Knowledge(net_1, net_2, net_3)
        return knowledge

    def create_client(self, cid: str, config: Config, **kwargs) -> TD3Client:
        try:
            cid = int(cid)
        except:
            raise ValueError("cid should be an integer")
        env = copy.deepcopy(self.env)

        knowledge = self.create_default_knowledge(config)
        client = TD3Client(
            knowledge=knowledge,
            env=env,
            config=config,
            seed=cid,
            device=self.device,
            **kwargs,
        )
        return client

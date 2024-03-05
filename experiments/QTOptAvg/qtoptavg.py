import copy
from flwr.common.typing import GetParametersIns, Parameters

import gymnasium as gym
import torch
from flwr.common import Config
import kitten
from kitten.rl.qt_opt import QTOpt
from kitten.common.util import build_env, build_critic

from florl.common import NumPyKnowledge
from florl.common.util import get_torch_parameters, set_torch_parameters
from florl.client.kitten import KittenClient

class QTOptKnowledge(NumPyKnowledge):
    def __init__(self, critic_1: torch.nn.Module, critic_2: torch.nn.Module) -> None:
        super().__init__(["critic_1", "critic_target_1", "critic_2", "critic_target_2"])
        self.critic_1 = kitten.nn.AddTargetNetwork(copy.deepcopy(critic_1))
        self.critic_2 = kitten.nn.AddTargetNetwork(copy.deepcopy(critic_2))
    
    def _get_module_parameters_numpy(self, id_: str, ins: GetParametersIns):
        if id_ == "critic_1":
            return get_torch_parameters(self.critic_1.net)
        elif id_ == "critic_target_1":
            return get_torch_parameters(self.critic_1.target)
        elif id_ == "critic_2":
            return get_torch_parameters(self.critic_2.target)
        elif id_ == "critic_target_2":
            return get_torch_parameters(self.critic_2.target)
        else:
            raise ValueError(f"Unknown id {id_}")

    def _set_module_parameters_numpy(self, id_: str, ins: Parameters):
        if id_ == "critic_1":
            return set_torch_parameters(self.critic_1.net)
        elif id_ == "critic_target_1":
            return set_torch_parameters(self.critic_1.target)
        elif id_ == "critic_2":
            return set_torch_parameters(self.critic_2.target)
        elif id_ == "critic_target_2":
            return set_torch_parameters(self.critic_2.target)
        else:
            raise ValueError(f"Unknown id {id_}")

class QTOptClient(KittenClient):
    def __init__(self,
                 knowledge: QTOptKnowledge,
                 env: gym.Env,
                 config: Config,
                 seed: int | None = None,
                 device: str = "cpu"):
        super().__init__(knowledge, env, config, seed, True, device)

    # Algorithm
    def build_algorithm(self) -> None:
        self._cfg.get("algorithm", {}).pop("critic", None)
        self._algorithm = QTOpt(
            critic_1_network=self._knowl.critic_1.net,
            critic_2_network=self._knowl.critic_2.net,
            obs_space=self._env.observation_space,
            action_space=self._env.action_space,
            **self._cfg.get("algorithm", {})
        )
        self._policy = kitten.policy.EpsilonGreedyPolicy(
            fn=self.algorithm.policy_fn,
            action_space=self._env.action_space,
            rng=self._rng.numpy,
            device=self._device
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

class QTOptClientFactory:
    def __init__(self, config: Config, device: str = "cpu") -> None:
        self.env = build_env(**config["rl"]["env"])
        self.net_1 = build_critic(
            env=self.env,
            **config
                .get("rl",{})
                .get("algorithm", {})
                .get("critic", {})
        )
        self.net_2 = build_critic(
            env=self.env,
            **config
                .get("rl",{})
                .get("algorithm", {})
                .get("critic", {})
        )
        self.device = device

    def create_dqn_client(self, cid: int, config: Config) -> QTOptClient:
        env = copy.deepcopy(self.env)
        net_1 = copy.deepcopy(self.net_1)
        net_2 = copy.deepcopy(self.net_2)

        knowledge = QTOptKnowledge(net_1, net_2)
        client = QTOptClient(
            knowledge=knowledge,
            env=env,
            config=config,
            seed=cid,
            device=self.device
        )
        return client

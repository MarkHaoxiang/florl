import copy
from flwr.common.typing import GetParametersIns, NDArrays, Parameters

import gymnasium as gym
import torch.nn as nn
from flwr.common import Config
import kitten
from kitten.rl.dqn import DQN

from florl.common import NumPyKnowledge
from florl.common.util import get_torch_parameters, set_torch_parameters
from florl.client.kitten import KittenClient

class DQNKnowledge(NumPyKnowledge):
    def __init__(self, critic: nn.Module) -> None:
        super().__init__(["critic", "critic_target"])
        self.critic = kitten.nn.AddTargetNetwork(copy.deepcopy(critic))
    
    def _get_module_parameters_numpy(self, id_: str, ins: GetParametersIns):
        if id_ == "critic":
            return get_torch_parameters(self.critic.net)
        elif id_ == "critic_target":
            return get_torch_parameters(self.critic.target)
        else:
            raise ValueError(f"Unknown id {id_}")

    def _set_module_parameters_numpy(self, id_: str, ins: Parameters):
        if id_ == "critic":
            set_torch_parameters(self.critic.net, ins)
        elif id_ == "critic_target":
            set_torch_parameters(self.critic.target, ins)
        else:
            raise ValueError(f"Unknown id {id_}")

class DQNClient(KittenClient):
    def __init__(self,
                 knowledge: DQNKnowledge,
                 env: gym.Env,
                 config: Config,
                 seed: int | None = None,
                 device: str = "cpu"):
        super().__init__(knowledge, env, config, seed, True, device)

    # Algorithm
    def build_algorithm(self) -> None:
        self._cfg.get("algorithm", {}).pop("critic", None)
        self._algorithm = DQN(
            critic=self._knowl.critic.net,
            device=self._device,
            **self._cfg.get("algorithm", {})
        )
        self._policy = kitten.policy.EpsilonGreedyPolicy(
            fn=self.algorithm.policy_fn,
            action_space=self._env.action_space,
            device=self._device
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

    def train(self, net: nn.Module, train_config: Config):
        metrics = {}
        # Synchronise critic net
        self.algorithm.critic.net = net
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
        return metrics, train_config["frames"]

class DQNClientFactory:
    def __init__(self, config: Config, device: str = "cpu") -> None:
        self.env = kitten.util.build_env(**config["rl"]["env"])
        self.net = kitten.util.build_critic(
            env=self.env,
            **config
                .get("rl",{})
                .get("algorithm", {})
                .get("critic", {})
        )
        self.device = device
        
    def create_dqn_client(self, cid: int, config: Config) -> DQNClient:
        env = copy.deepcopy(self.env)
        net = copy.deepcopy(self.net)

        knowledge = DQNKnowledge(net)
        return DQNClient(
            knowledge=knowledge,
            env=env,
            config=config,
            seed=cid,
            device=self.device
        )

import copy
from types import MethodType

import torch
import gymnasium as gym
from flwr.common import Config
import kitten
from kitten.rl.dqn import DQN
from kitten.common.util import build_env, build_critic

from florl.client import FlorlFactory
from florl.common import Knowledge, NumPyKnowledge
from florl.client.kitten import KittenClient


class DQNKnowledge(NumPyKnowledge):
    def __init__(self, critic: kitten.nn.Critic) -> None:
        super().__init__(["critic", "critic_target"])
        self.critic = kitten.nn.AddTargetNetwork(copy.deepcopy(critic))

    @property
    def torch_modules_registry(self):
        return {"critic": self.critic.net, "critic_target": self.critic.target}


class DQNClient(KittenClient):
    def __init__(
        self,
        knowledge: DQNKnowledge,
        env: gym.Env,
        config: Config,
        seed: int | None = None,
        enable_evaluation: bool = True,
        device: str = "cpu",
    ):
        super().__init__(knowledge, env, config, seed, True, enable_evaluation, device)
        self._knowl: DQNKnowledge = self._knowl  # Typing hints

        # Used for FedProx injection
        self._global_knowl = copy.deepcopy(self._knowl)
        self._proximal_mu = 0.0

    # Algorithm
    def build_algorithm(self) -> None:
        self._cfg.get("algorithm", {}).pop("critic", None)
        self._algorithm = DQN(
            critic=self._knowl.critic.net,
            device=self._device,
            **self._cfg.get("algorithm", {}),
        )

        # Inject update to support FedProx
        def update(
            dqn_self: DQN,
            batch: kitten.experience.Transition,
            aux: kitten.experience.AuxiliaryMemoryData,
            step: int,
        ):
            # Kitten
            if step % dqn_self._update_frequency == 0:
                dqn_self._optim.zero_grad()
                loss = torch.mean((dqn_self.td_error(*batch) * aux.weights) ** 2)
                # ===== Proximal Term Here =====
                proximal_term = 0.0
                if self._global_knowl is not None and self._proximal_mu > 0.0:
                    for local_weights, global_weights in zip(
                        dqn_self._critic.net.parameters(),
                        self._global_knowl.critic.net.parameters(),
                    ):
                        proximal_term += torch.square(
                            (local_weights - global_weights).norm(2)
                        )
                proximal_term = proximal_term * (self._proximal_mu / 2)
                loss += proximal_term
                # Kitten
                dqn_self.loss_critic_value = loss.item()
                loss.backward()
                dqn_self._optim.step()
            if step % dqn_self._target_update_frequency == 0:
                dqn_self._critic.update_target_network()
            return dqn_self.loss_critic_value

        self._algorithm.update = MethodType(update, self._algorithm)

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
        if "proximal_mu" in train_config:
            self._proximal_mu = train_config["proximal_mu"]
        else:
            self._proximal_mu = 0
        self._global_knowl = copy.deepcopy(self._knowl)
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


class DQNClientFactory(FlorlFactory):
    def __init__(self, config: Config, device: str = "cpu") -> None:
        self.env = build_env(**config["rl"]["env"])
        self.net = build_critic(
            env=self.env, **config.get("rl", {}).get("algorithm", {}).get("critic", {})
        )
        self.device = device

    def create_default_knowledge(self, config: Config) -> Knowledge:
        net = copy.deepcopy(self.net)
        knowledge = DQNKnowledge(net)
        return knowledge

    def create_client(self, cid: str, config: Config, **kwargs) -> DQNClient:
        try:
            cid = int(cid)
        except:
            raise ValueError("cid should be an integer")
        env = copy.deepcopy(self.env)
        knowledge = self.create_default_knowledge(config)
        client = DQNClient(
            knowledge=knowledge,
            env=env,
            config=config,
            seed=cid,
            device=self.device,
            **kwargs,
        )
        return client

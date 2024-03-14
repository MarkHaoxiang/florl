import copy

import gymnasium as gym
import torch
from torch import nn
from flwr.common import Config
import kitten
from kitten.experience import AuxiliaryMemoryData, Transition
from kitten.rl.qt_opt import QTOpt
from kitten.common.util import build_env, build_critic


from florl.client import FlorlFactory
from florl.common import Knowledge, NumPyKnowledge
from .client import KittenClient

class QTOptKnowledge(NumPyKnowledge):
    def __init__(self, critic_1: kitten.nn.Critic, critic_2: kitten.nn.Critic) -> None:
        super().__init__(["critic_1", "critic_target_1", "critic_2", "critic_target_2"])
        self.critic_1 = kitten.nn.AddTargetNetwork(copy.deepcopy(critic_1))
        self.critic_2 = kitten.nn.AddTargetNetwork(copy.deepcopy(critic_2))

    @property
    def torch_modules_registry(self):
        return {
            "critic_1": self.critic_1.net,
            "critic_target_1": self.critic_1.target,
            "critic_2": self.critic_2.net,
            "critic_target_2": self.critic_2.target,
        }

# TODO: FedProxRL implementation breaks updating subset of parameters
class QTOptProx(QTOpt):
    """ QTOpt with optional proximal loss term
    """
    def _critic_update(self, batch: Transition, aux: AuxiliaryMemoryData, proximal_mu: float = 0, global_knowledge: QTOptKnowledge | None = None):
        # ===== Kitten Code =====
        x_1 = self._critic_1.q(batch.s_0, batch.a).squeeze()
        x_2 = self._critic_2.q(batch.s_0, batch.a).squeeze()
        with torch.no_grad():
            a_1 = self.policy_fn(batch.s_1, critic=self._critic_1.target)
            a_2 = self.policy_fn(batch.s_1, critic=self._critic_2.target)
            target_max_1 = self._critic_1.target.q(batch.s_1, a_1).squeeze()
            target_max_2 = self._critic_2.target.q(batch.s_1, a_2).squeeze()
            y = (
                batch.r
                + (~batch.d) * torch.minimum(target_max_1, target_max_2) * self._gamma
            )
        loss_critic = torch.mean((aux.weights * (y - x_1)) ** 2) + torch.mean(
            (aux.weights * (y - x_2)) ** 2
        )
        # ===== Proximal Term Here =====
        proximal_term = 0.0
        if global_knowledge is not None:
            for local_weights, global_weights in zip(self._critic_1.net.parameters(), global_knowledge.critic_1.net.parameters()):
                proximal_term += torch.square((local_weights - global_weights).norm(2))
            for local_weights, global_weights in zip(self._critic_2.net.parameters(), global_knowledge.critic_2.net.parameters()):
                proximal_term += torch.square((local_weights - global_weights).norm(2))
        proximal_term = proximal_term * (proximal_mu / 2)
        loss_critic += proximal_term
        # ===== Kitten Code =====
        loss_value = loss_critic.item()
        self._optim_critic.zero_grad()
        loss_critic.backward()
        if not self._clip_grad_norm is None:
            nn.utils.clip_grad_norm_(
                self._critic_1.net.parameters(), self._clip_grad_norm
            )
            nn.utils.clip_grad_norm_(
                self._critic_2.net.parameters(), self._clip_grad_norm
            )
        self._optim_critic.step()

        return loss_value

    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int, proximal_mu: float = 0, global_knowledge: QTOptKnowledge | None = None):
        if step % self._update_frequency == 0:
            self.loss_critic_value = self._critic_update(batch, aux, proximal_mu, global_knowledge)
            self._critic_1.update_target_network(tau=self._tau)
            self._critic_2.update_target_network(tau=self._tau)
        return self.loss_critic_value

class QTOptClient(KittenClient):
    def __init__(
        self,
        knowledge: QTOptKnowledge,
        env: gym.Env,
        config: Config,
        seed: int | None = None,
        enable_evaluation: bool = True,
        device: str = "cpu",
    ):
        super().__init__(knowledge, env, config, seed, True, enable_evaluation, device)
        self._knowl: QTOptKnowledge = self._knowl  # Type hints

    # Algorithm
    def build_algorithm(self) -> None:
        self._cfg.get("algorithm", {}).pop("critic", None)
        self._algorithm = QTOptProx(
            critic_1_network=self._knowl.critic_1.net,
            critic_2_network=self._knowl.critic_2.net,
            obs_space=self._env.observation_space,
            action_space=self._env.action_space,
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
        
        global_knowledge = copy.deepcopy(self._knowl)
        # Synchronise critic net
        critic_loss = []
        # Training
        for _ in range(train_config["frames"]):
            self._step += 1
            # Collected Transitions
            self._collector.collect(n=1)
            batch, aux = self._memory.sample(self._cfg["train"]["minibatch_size"])
            batch = kitten.experience.Transition(*batch)
            if "test" not in metrics:
                metrics["sample"] = float(batch.s_0.mean().item())
            # Algorithm Update
            if "proximal_mu" in train_config:
                critic_loss.append(self.algorithm.update(batch, aux, self.step, train_config["proximal_mu"], global_knowledge))
            else:
                critic_loss.append(self.algorithm.update(batch, aux, self.step))
        
        metrics["state_average_end"] = float(self._memory.storage[0].mean())
        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return len(self._memory), metrics


class QTOptClientFactory(FlorlFactory):
    def __init__(self, config: Config, device: str = "cpu") -> None:
        self.env = build_env(**config["rl"]["env"])
        self.net_1 = build_critic(
            env=self.env, **config.get("rl", {}).get("algorithm", {}).get("critic", {})
        )
        self.net_2 = build_critic(
            env=self.env, **config.get("rl", {}).get("algorithm", {}).get("critic", {})
        )
        self.device = device

    def create_default_knowledge(self, config: Config) -> Knowledge:
        net_1 = copy.deepcopy(self.net_1)
        net_2 = copy.deepcopy(self.net_2)
        knowledge = QTOptKnowledge(net_1, net_2)
        return knowledge

    def create_client(self, cid: str, config: Config, **kwargs) -> QTOptClient:
        try:
            cid = int(cid)
        except:
            raise ValueError("cid should be an integer")
        env = copy.deepcopy(self.env)

        knowledge = self.create_default_knowledge(config)
        client = QTOptClient(
            knowledge=knowledge,
            env=env,
            config=config,
            seed=cid,
            device=self.device,
            **kwargs,
        )
        return client
